from __future__ import annotations
import numpy as np
import pandas as pd

def sharpe_ratio(returns: pd.Series, risk_free: float=0.0, periods_per_year: int=252) -> float:
    ex_ret = returns - risk_free/periods_per_year
    mu = ex_ret.mean() * periods_per_year
    sd = ex_ret.std(ddof=1) * np.sqrt(periods_per_year)
    if sd == 0 or np.isnan(sd):
        return 0.0
    return float(mu / sd)

def max_drawdown(equity: pd.Series) -> float:
    cummax = equity.cummax()
    dd = equity / cummax - 1.0
    return float(dd.min())

def turnover(positions: pd.Series) -> float:
    # суммарный абсолютный оборот позиций
    return float((positions.diff().abs().fillna(0)).sum())
