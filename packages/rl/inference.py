from __future__ import annotations
import argparse
import pandas as pd
import numpy as np
from stable_baselines3 import PPO

from ..data.ohlcv_store import parquet_path
from ..env.features import make_features
from ..env.trading_env import SingleAssetDailyEnv
from ..backtest.fees import FeeModel

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--timeframe", default="1d")
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--model_path", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(parquet_path(args.symbol, args.timeframe)).sort_index()
    feats = make_features(df, window=args.window)
    prices = df.loc[feats.index, "close"]

    env = SingleAssetDailyEnv(features=feats, prices=prices, fee_model=FeeModel())
    model = PPO.load(args.model_path)

    obs, _ = env.reset()
    positions = []
    rewards = []
    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        positions.append(info["position"])
        rewards.append(reward)
        if done or truncated:
            break

    import pandas as pd
    res = pd.DataFrame({
        "position": positions,
        "reward": rewards
    }, index=feats.index[1:len(positions)+1])

    # Простая equity-кривая
    res["equity"] = (1 + pd.Series(rewards, index=res.index)).cumprod()
    print(res.tail())
    out_path = f"models/signals-{args.symbol.replace('/','')}-{args.timeframe}.parquet"
    res.to_parquet(out_path)
    print(f"Saved signals to {out_path}")

if __name__ == "__main__":
    main()
