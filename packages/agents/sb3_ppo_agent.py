from __future__ import annotations

from typing import Optional, Dict

import numpy as np

from stable_baselines3 import PPO


class SB3PPOAgent:
    """RL-агент на базе Stable-Baselines3 PPO."""

    uses_rl = True  # маркер для Trainer

    def __init__(self, policy: str = "MlpPolicy", policy_kwargs: Optional[Dict] = None):
        self.policy = policy
        self.policy_kwargs = policy_kwargs or {}
        self.model: PPO | None = None

    # RL-тренинг: ожидает vec_env и kwargs для PPO (n_steps, batch_size, seed, tensorboard_log, verbose,…)
    def fit(self, vec_env, total_timesteps: int, **ppo_kwargs):
        self.model = PPO(
            self.policy, vec_env, policy_kwargs=self.policy_kwargs, **ppo_kwargs
        )
        self.model.learn(total_timesteps=total_timesteps)

    def act(self, obs: np.ndarray, deterministic: bool = True) -> int:
        assert self.model is not None, "Model not trained/loaded"
        action, _ = self.model.predict(obs, deterministic=deterministic)
        return int(action)

    def save(self, path: str) -> None:
        assert self.model is not None, "Model not trained/loaded"
        self.model.save(path)

    @classmethod
    def load(cls, path: str) -> "SB3PPOAgent":
        agent = cls()
        agent.model = PPO.load(path)
        # policy_kwargs восстановятся из .zip; policy можно оставить по умолчанию
        return agent

    def to_meta(self) -> dict:
        return {
            "agent_class": "packages.agents.sb3_ppo_agent.SB3PPOAgent",
            "policy": self.policy,
            "policy_kwargs": self.policy_kwargs,
        }

    @classmethod
    def from_meta(cls, meta: dict) -> "SB3PPOAgent":
        return cls(
            policy=meta.get("policy", "MlpPolicy"),
            policy_kwargs=meta.get("policy_kwargs") or {},
        )
