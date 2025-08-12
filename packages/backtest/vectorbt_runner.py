from __future__ import annotations

import pandas as pd
import numpy as np
import vectorbt as vbt
from .fees import FeeModel


def sma_crossover(
    close: pd.Series,
    fast: int = 10,
    slow: int = 50,
    fee_model: FeeModel | None = None
):
    """
    SMA-кроссовер на vectorbt:
    - entries: fast SMA пересекает сверху slow SMA
    - exits:   fast SMA пересекает снизу slow SMA
    """

    close = close.sort_index().astype(float).dropna()

    fast_ind = vbt.MA.run(close, window=fast)
    slow_ind = vbt.MA.run(close, window=slow)

    # ВАЖНО: используем методы индикатора, а не Series
    entries = fast_ind.ma_crossed_above(slow_ind)
    exits   = fast_ind.ma_crossed_below(slow_ind)

    commission = fee_model.commission if fee_model else 0.0
    slippage   = fee_model.spread     if fee_model else 0.0

    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        fees=commission,
        slippage=slippage,
        freq="D",
        init_cash=10_000.0
    )
    return pf
