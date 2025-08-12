from __future__ import annotations
import argparse
from datetime import datetime, timezone
import pandas as pd

from packages.data.bybit_client import BybitClient
from packages.data.ohlcv_store import ohlcv_to_df, save_parquet

def parse_date(s: str):
    return int(pd.Timestamp(s, tz='UTC').timestamp() * 1000)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", nargs="+", required=True, help="Примеры: BTC/USDT ETH/USDT")
    ap.add_argument("--timeframe", default="1d")
    ap.add_argument("--since", required=True, help="Дата начала, напр. 2021-01-01")
    ap.add_argument("--until", required=True, help="Дата конца, напр. 2025-01-01")
    ap.add_argument("--sandbox", type=str, default="true")
    args = ap.parse_args()

    client = BybitClient(sandbox=args.sandbox.lower() in ("1","true","yes"))
    since_ms = parse_date(args.since)
    until_ms = parse_date(args.until)

    for sym in args.symbols:
        rows = client.fetch_ohlcv_all(sym, timeframe=args.timeframe, since_ms=since_ms, until_ms=until_ms)
        if not rows:
            print(f"No data for {sym}")
            continue
        df = ohlcv_to_df(rows)
        path = save_parquet(df, symbol=sym, timeframe=args.timeframe)
        print(f"Saved {len(df)} bars to {path}")

if __name__ == "__main__":
    main()
