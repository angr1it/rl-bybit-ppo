# Live trading data contracts

This document describes data structures exchanged by the live trading layer.

## Core types

```python
from dataclasses import dataclass
from typing import Optional, Literal
from datetime import datetime

Side = Literal["buy", "sell"]
TIF = Literal["GTC", "IOC", "FOK"]

@dataclass
class SymbolSpec:
    symbol: str
    price_step: float
    qty_step: float
    min_qty: float
    max_qty: float
    base_ccy: str
    quote_ccy: str
    contract_type: Literal["spot", "linear", "inverse"]

@dataclass
class Quote:
    ts: datetime
    bid: float
    ask: float
    last: float

@dataclass
class Kline:
    ts_open: datetime
    ts_close: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    confirmed: bool  # closed candle

@dataclass
class OrderRequest:
    cl_id: str
    symbol: str
    side: Side
    qty: float
    price: Optional[float] = None
    tif: TIF = "IOC"
    reduce_only: bool = False
    post_only: bool = False

@dataclass
class OrderReport:
    cl_id: str
    order_id: str
    status: Literal["accepted", "rejected", "filled", "partially_filled", "canceled"]
    filled_qty: float
    avg_price: Optional[float]
    ts: datetime
    reason: Optional[str] = None

@dataclass
class Position:
    symbol: str
    qty: float  # >0 long, <0 short
    avg_price: float
    unrealized_pnl: float
```

These contracts mirror the structures used by the Bybit v5 API and are shared by both
paper and live brokers.
