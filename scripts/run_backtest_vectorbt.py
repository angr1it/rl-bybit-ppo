from __future__ import annotations
import argparse
import pandas as pd
from packages.data.ohlcv_store import parquet_path
from packages.backtest.vectorbt_runner import sma_crossover
from packages.backtest.metrics import sharpe_ratio, max_drawdown

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--timeframe", default="1d")
    ap.add_argument("--fast", type=int, default=10)
    ap.add_argument("--slow", type=int, default=50)
    args = ap.parse_args()

    df = pd.read_parquet(parquet_path(args.symbol, args.timeframe)).sort_index()
    pf = sma_crossover(df["close"], fast=args.fast, slow=args.slow)
    stats = pf.stats()

    print(stats.tail(20))
    ret = pf.returns().rename("ret")
    print("Sharpe:", sharpe_ratio(ret))
    equity = pf.value()
    print("MaxDD:", max_drawdown(equity))

if __name__ == "__main__":
    main()
