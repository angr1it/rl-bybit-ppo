
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

from ..data.ohlcv_store import parquet_path
from ..backtest.vectorbt_runner import sma_crossover
from ..env.features import make_features

def actions_from_sma(df_close: pd.Series, fast=10, slow=50):
    # -1/0/+1 по пересечениям SMA
    pf = sma_crossover(df_close, fast=fast, slow=slow)
    pos = pf.positions().iloc[:,0]  # vectorbt stores columns
    # Преобразуем в {-1,0,+1}
    pos = pos.clip(-1,1).astype(int)
    return pos.reindex(df_close.index).fillna(0).astype(int)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--timeframe", default="1d")
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    df = pd.read_parquet(parquet_path(args.symbol, args.timeframe)).sort_index()
    feats = make_features(df, window=args.window)
    close = df.loc[feats.index, "close"]
    rets = close.pct_change().fillna(0.0)

    actions = actions_from_sma(close).loc[feats.index]
    # Награда как pos[t-1]*ret[t]
    reward = actions.shift(1).fillna(0) * rets
    done = np.zeros(len(feats), dtype=bool)
    done[-1] = True

    # Сохраним как npz для d3rlpy (s,a,r,s',done)
    obs = feats.values.astype("float32")
    next_obs = np.roll(obs, -1, axis=0)
    dataset = {
        "observations": obs[:-1],
        "actions": actions.values[:-1].astype("int64"),
        "rewards": reward.values[:-1].astype("float32"),
        "terminals": done[:-1].astype("bool"),
        "next_observations": next_obs[:-1],
    }
    out_path = args.out or f"data/offline/{args.symbol.replace('/','')}-{args.timeframe}-window{args.window}.npz"
    Path = __import__('pathlib').Path
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **dataset)
    print(f"Saved offline dataset to {out_path} (N={len(dataset['observations'])})")

if __name__ == "__main__":
    main()
