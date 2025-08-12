from __future__ import annotations
from pathlib import Path
import pandas as pd
from .config import SETTINGS

def ohlcv_to_df(rows):
    return pd.DataFrame(rows, columns=["timestamp","open","high","low","close","volume"]).assign(
        dt=lambda df: pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    ).set_index("dt")

def parquet_path(symbol: str, timeframe: str) -> Path:
    sym = symbol.replace("/", "")
    base = Path(SETTINGS.data_dir) / "bybit" / sym / timeframe
    base.mkdir(parents=True, exist_ok=True)
    return base / f"{sym}-{timeframe}.parquet"

def save_parquet(df: pd.DataFrame, symbol: str, timeframe: str) -> Path:
    path = parquet_path(symbol, timeframe)
    df.to_parquet(path)
    return path
