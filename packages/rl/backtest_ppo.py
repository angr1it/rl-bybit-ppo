from __future__ import annotations
import argparse
import pandas as pd
from stable_baselines3 import PPO

from ..data.ohlcv_store import parquet_path
from ..env.features import make_features
from ..env.trading_env import SingleAssetDailyEnv
from ..backtest.fees import FeeModel


def load_price_df(symbol: str, timeframe: str, start: str, end: str):
    path = parquet_path(symbol, timeframe)
    df = pd.read_parquet(path).sort_index()
    return df.loc[start:end]


def build_env(df, window: int):
    feats = make_features(df, window=window)
    if feats.empty:
        raise ValueError(f"Нет данных для window={window} (bars={len(df)})")
    prices = df.loc[feats.index, "close"]
    return SingleAssetDailyEnv(features=feats, prices=prices, fee_model=FeeModel())


def evaluate_model(model, df: pd.DataFrame, window: int):
    env = build_env(df, window)
    obs, _ = env.reset()
    done = False
    total_reward = 0
    rewards = []
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward
        rewards.append(reward)

    return {
        "total_reward": total_reward,
        "mean_reward": sum(rewards) / len(rewards),
        "n_steps": len(rewards),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--timeframe", default="1d")
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True)
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--model_path", required=True)
    args = ap.parse_args()

    df = load_price_df(args.symbol, args.timeframe, args.start, args.end)
    model = PPO.load(args.model_path)

    results = evaluate_model(model, df, args.window)
    print(f"Validation results ({args.start} → {args.end}):")
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
