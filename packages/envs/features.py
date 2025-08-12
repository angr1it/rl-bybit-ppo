from __future__ import annotations
import pandas as pd
import numpy as np

def make_features(df: pd.DataFrame, window: int=64) -> pd.DataFrame:
    # df: index=dt, columns=[open,high,low,close,volume]
    out = pd.DataFrame(index=df.index.copy())
    out["ret1"] = df["close"].pct_change().fillna(0.0)
    out["vol"] = df["close"].pct_change().rolling(10).std().fillna(0.0)
    out["mom"] = df["close"].pct_change(10).fillna(0.0)
    out = out.dropna()
    return out.tail(len(out))  # simple passthrough
