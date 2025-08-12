
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
import wandb

from ..data.ohlcv_store import parquet_path
from ..env.features import make_features
from ..env.trading_env import SingleAssetDailyEnv
from ..backtest.fees import FeeModel

class WandbCallback:
    def __init__(self): pass
    def __call.pretrain__(self): pass

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--timeframe", default="1d")
    ap.add_argument("--window", type=int, default=128)
    ap.add_argument("--total_timesteps", type=int, default=100_000)
    ap.add_argument("--wandb", type=str, default="false")
    args = ap.parse_args()

    use_wandb = args.wandb.lower() in ("1","true","yes")
    if use_wandb:
        wandb.init(project="rl-bybit-ppo", name=f"rppo-lstm-{args.symbol.replace('/','')}-{args.timeframe}")

    df = pd.read_parquet(parquet_path(args.symbol, args.timeframe)).sort_index()
    feats = make_features(df, window=args.window)
    prices = df.loc[feats.index, "close"]
    env = SingleAssetDailyEnv(features=feats, prices=prices, fee_model=FeeModel())
    env = Monitor(env)
    vec_env = DummyVecEnv([lambda: env])

    model = RecurrentPPO("MlpLstmPolicy", vec_env, verbose=1, tensorboard_log="logs/rppo")
    model.learn(total_timesteps=args.total_timesteps)

    out = f"models/rppo-lstm-{args.symbol.replace('/','')}-{args.timeframe}.zip"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    model.save(out)
    print(f"Saved model to {out}")

if __name__ == "__main__":
    main()
