from __future__ import annotations

import os, sys, json, random
from dataclasses import dataclass, field
from typing import Optional, Dict, Type, Tuple

import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor

from ..data.ohlcv_store import parquet_path
from ..env.features import make_features
from ..backtest.fees import FeeModel
from ..utils.imports import import_by_path


# =========================
# Config
# =========================


@dataclass
class TrainConfig:
    symbol: str
    timeframe: str = "1d"
    start: Optional[str] = None
    end: Optional[str] = None
    window: int = 128
    total_timesteps: int = 200_000
    save_path: Optional[str] = None

    # seed: None / -1 => рандом
    seed: Optional[int] = 42
    tb: bool = False

    # опциональный оверрайд авто-параметров RL
    n_steps: Optional[int] = None
    batch_size: Optional[int] = None
    log_interval: Optional[int] = None

    # режим тренинга: 'rl' или 'supervised'
    train_mode: str = "rl"

    # политика SB3 и её kwargs (если train_mode='rl')
    policy: str = "MlpPolicy"
    policy_kwargs: Dict = field(default_factory=dict)

    # Класс среды и его kwargs (для любых агентов)
    env_class: Optional[Type] = None
    env_kwargs: Dict = field(default_factory=dict)

    # W&B
    use_wandb: bool = False
    wandb_project: str = "rl-bybit-ppo"
    wandb_run_name: Optional[str] = None

    # путь класса агента (если хочешь восстанавливать из артефакта)
    agent_class_path: Optional[str] = None

    def __post_init__(self):
        if self.seed is None or self.seed == -1:
            self.seed = int.from_bytes(os.urandom(4), "little")


# =========================
# Trainer
# =========================


# наверх файла, рядом с импортами
def _json_safe(obj):
    """Рекурсивно делает объект JSON‑сериализуемым:
    классы/функции -> import path, dict/list/tuple -> поэлементно."""
    import inspect

    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    # class или функция -> строковый путь
    if (
        isinstance(obj, type)
        or inspect.isfunction(obj)
        or inspect.isclass(obj)
        or inspect.ismethod(obj)
    ):
        mod = getattr(obj, "__module__", None)
        name = getattr(obj, "__name__", None)
        if mod and name:
            return f"{mod}.{name}"
    return obj


class Trainer:
    """Универсальный тренер для RL и супервизии, с бэктестом и метаданными артефакта."""

    def __init__(self, cfg: TrainConfig, agent):
        self.cfg = cfg
        self.agent = agent
        self.df: Optional[pd.DataFrame] = None
        self.feats: Optional[pd.DataFrame] = None
        self.prices: Optional[pd.Series] = None
        self.effective_timesteps: int = 0
        self.observation_shape: Optional[Tuple[int, ...]] = None

    # ---------- data / env ----------

    @staticmethod
    def _load_price_df(
        symbol: str, timeframe: str, start: Optional[str], end: Optional[str]
    ) -> pd.DataFrame:
        path = parquet_path(symbol, timeframe)
        df = pd.read_parquet(path).sort_index()
        if start:
            df = df.loc[start:]
        if end:
            df = df.loc[:end]
        return df

    def _prepare_data(self):
        print(f"[trainer] cfg: {self.cfg}", flush=True)
        self.df = self._load_price_df(
            self.cfg.symbol, self.cfg.timeframe, self.cfg.start, self.cfg.end
        )
        if self.df.empty:
            print(
                "[trainer] после фильтра по датам данных нет",
                file=sys.stderr,
                flush=True,
            )
            sys.exit(1)
        print(
            f"[trainer] rows_raw={len(self.df)}  range=({self.df.index.min()} .. {self.df.index.max()})",
            flush=True,
        )

        self.feats = make_features(self.df, window=self.cfg.window)
        if self.feats.empty:
            print(
                f"[trainer] после make_features(window={self.cfg.window}) данных нет",
                file=sys.stderr,
                flush=True,
            )
            sys.exit(1)
        self.prices = self.df.loc[self.feats.index, "close"]

        max_env_steps = max(
            0, len(self.feats) - 1
        )  # один шаг на бар, последний бар — терминальный
        self.effective_timesteps = min(self.cfg.total_timesteps, max_env_steps)
        print(
            f"[trainer] rows_after_features={len(self.feats)}  "
            f"max_env_steps={max_env_steps} requested={self.cfg.total_timesteps} "
            f"effective={self.effective_timesteps}",
            flush=True,
        )
        if self.effective_timesteps <= 0:
            print(
                "[trainer] effective_timesteps=0 → увеличь диапазон дат или уменьшай window.",
                flush=True,
            )
            sys.exit(0)

    def _env_class_default(self):
        if self.cfg.env_class is not None:
            return self.cfg.env_class
        # по умолчанию — дневной env
        from ..env.trading_env import SingleAssetDailyEnv

        return SingleAssetDailyEnv

    def _make_env(self):
        Env = self._env_class_default()
        env = Env(
            features=self.feats,
            prices=self.prices,
            fee_model=FeeModel(),
            **self.cfg.env_kwargs,
        )
        # seed среды
        try:
            env.reset(seed=self.cfg.seed)
            if hasattr(env.action_space, "seed"):
                env.action_space.seed(self.cfg.seed)
            if hasattr(env.observation_space, "seed"):
                env.observation_space.seed(self.cfg.seed)
        except TypeError:
            try:
                env.seed(self.cfg.seed)
            except Exception:
                pass
        # сохраним форму наблюдения
        try:
            self.observation_shape = tuple(env.observation_space.shape)  # type: ignore[attr-defined]
        except Exception:
            self.observation_shape = None
        return env

    # ---------- RL auto-params ----------

    @staticmethod
    def _pick_n_steps(eff_steps: int) -> int:
        if eff_steps <= 64:
            return max(16, eff_steps // 2)
        cand = max(32, min(1024, eff_steps // 6))  # ≈6 итераций
        if cand >= eff_steps:
            cand = max(32, eff_steps - 1)
        return cand

    @staticmethod
    def _pick_batch_size(n_steps: int) -> int:
        for bs in (512, 256, 128, 64, 32, 16, 8):
            if n_steps % bs == 0 and bs <= n_steps:
                return bs
        return max(8, min(64, n_steps))

    # ---------- public API ----------

    def train(self):
        # сидинг
        np.random.seed(self.cfg.seed)
        random.seed(self.cfg.seed)
        torch.manual_seed(self.cfg.seed)

        # данные
        self._prepare_data()

        if getattr(self.agent, "uses_rl", False) or self.cfg.train_mode == "rl":
            # RL тренинг (SB3 PPO)
            env = self._make_env()
            vec_env = DummyVecEnv([lambda: env])
            vec_env = VecMonitor(vec_env, filename=None)

            n_steps = self.cfg.n_steps or self._pick_n_steps(self.effective_timesteps)
            batch_size = self.cfg.batch_size or self._pick_batch_size(n_steps)
            log_interval = self.cfg.log_interval or 1
            tb_log = "logs/ppo" if self.cfg.tb else None

            print(
                f"[trainer] RL auto params: n_steps={n_steps}, batch_size={batch_size}, log_interval={log_interval}",
                flush=True,
            )
            self.agent.fit(
                vec_env=vec_env,
                total_timesteps=self.effective_timesteps,
                n_steps=n_steps,
                batch_size=batch_size,
                seed=self.cfg.seed,
                verbose=1,
                tensorboard_log=tb_log,
                # policy=self.cfg.policy,  # на случай, если агент их читает
                # policy_kwargs=self.cfg.policy_kwargs,  # (SB3PPOAgent переопределит своими)
            )

        else:
            # Супервизия: соберём (X, y) из фич
            # По умолчанию — классификация знака ретерна следующего шага
            f = self.feats.to_numpy()
            # future return ~ log(prices[t+1]/prices[t])
            pr = self.prices
            fut_ret = np.log(pr.shift(-1) / pr).reindex(self.feats.index).to_numpy()
            mask = ~np.isnan(fut_ret)
            X, y = f[mask], (fut_ret[mask] > 0).astype(int)
            print(
                f"[trainer] supervised dataset: X={X.shape}, y_pos={y.mean():.3f}",
                flush=True,
            )
            self.agent.fit(X, y)

        # сохранение артефакта
        model_path = (
            self.cfg.save_path
            or f"models/agent-{self.cfg.symbol.replace('/','')}-{self.cfg.timeframe}.bin"
        )
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.agent.save(model_path)
        meta = self._build_meta()
        with open(
            model_path.replace(".zip", ".meta.json").replace(".bin", ".meta.json"), "w"
        ) as f:
            json.dump(meta, f, indent=2)
        print(f"[trainer] Saved model to {model_path}", flush=True)
        return model_path

    def backtest(
        self,
        model_or_path,
        start: str,
        end: str,
        deterministic: bool = True,
        return_trace: bool = False,
    ):
        """Out-of-sample прогон на указанном диапазоне (env берём из cfg)."""
        # агент
        if hasattr(model_or_path, "act"):
            agent = model_or_path
        else:
            # загрузка по метаданным
            meta_path = model_or_path.replace(".zip", ".meta.json").replace(
                ".bin", ".meta.json"
            )
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                AgentCls = import_by_path(meta["agent_class"])
                agent = AgentCls.load(model_or_path)
            else:
                # fallback: SB3 PPO .zip
                from ..agents.sb3_ppo_agent import SB3PPOAgent

                agent = SB3PPOAgent.load(model_or_path)

        # данные/фичи на val/oot
        df = self._load_price_df(self.cfg.symbol, self.cfg.timeframe, start, end)
        feats = make_features(df, window=self.cfg.window)
        if feats.empty or len(feats) < 2:
            raise ValueError(
                f"[backtest] недостаточно данных на {start}..{end} для window={self.cfg.window}"
            )
        prices = df.loc[feats.index, "close"]

        # одиночный env (не VecEnv)
        Env = self._env_class_default()
        env = Env(
            features=feats, prices=prices, fee_model=FeeModel(), **self.cfg.env_kwargs
        )
        try:
            env.reset(seed=self.cfg.seed)
        except TypeError:
            try:
                env.seed(self.cfg.seed)
            except Exception:
                pass

        # прогон
        if hasattr(agent, "reset"):
            agent.reset()

        reset_out = env.reset()
        obs = reset_out[0] if isinstance(reset_out, tuple) else reset_out

        rewards = []
        done = False
        while not done:
            action = agent.act(obs, deterministic=deterministic)
            step_out = env.step(action)
            if len(step_out) == 5:
                obs, r, term, trunc, info = step_out
                done = bool(term) or bool(trunc)
            else:
                obs, r, done, info = step_out
            rewards.append(float(r))

        rew = np.array(rewards, dtype=float)
        res = {
            "total_reward": float(rew.sum()),
            "mean_reward": float(rew.mean()) if rew.size else 0.0,
            "n_steps": int(rew.size),
        }
        if return_trace:
            idx = feats.index[1 : 1 + len(rew)]
            res["reward_trace"] = pd.Series(rew, index=idx, name="reward")
        return res

    # ---------- meta ----------

    def _build_meta(self) -> dict:
        Env = self._env_class_default()
        meta = {
            "version": 1,
            "symbol": self.cfg.symbol,
            "timeframe": self.cfg.timeframe,
            "window": self.cfg.window,
            "env_class": f"{Env.__module__}.{Env.__name__}",
            "env_kwargs": _json_safe(self.cfg.env_kwargs),  # <— здесь
            "observation_shape": self.observation_shape,
            "train_mode": self.cfg.train_mode,
            "agent_class": f"{self.agent.__class__.__module__}.{self.agent.__class__.__name__}",
            "agent_meta": _json_safe(
                getattr(self.agent, "to_meta", lambda: {})()
            ),  # <— здесь
            "policy": self.cfg.policy,
            "policy_kwargs": _json_safe(self.cfg.policy_kwargs),  # <— здесь
            "seed": self.cfg.seed,
        }
        return meta

    @staticmethod
    def load_agent_from_artifact(model_path: str):
        meta_path = model_path.replace(".zip", ".meta.json").replace(
            ".bin", ".meta.json"
        )
        meta = json.load(open(meta_path))
        AgentCls = import_by_path(meta["agent_class"])
        return AgentCls.load(model_path), meta
