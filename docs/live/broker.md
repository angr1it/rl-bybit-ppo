# Broker interface

The `Broker` abstraction hides REST and WebSocket details behind a unified API.

```python
from abc import ABC, abstractmethod
from typing import Callable, Optional, Dict
from .models import OrderRequest, OrderReport, Position, SymbolSpec, Quote, Kline

class Broker(ABC):
    @abstractmethod
    def ensure_symbol(self, symbol: str) -> SymbolSpec: ...

    @abstractmethod
    def get_positions(self) -> Dict[str, Position]: ...

    @abstractmethod
    def create_order(self, req: OrderRequest) -> OrderReport: ...

    @abstractmethod
    def cancel_all(self, symbol: Optional[str] = None) -> None: ...

    # subscriptions
    @abstractmethod
    def on_quotes(self, cb: Callable[[Quote], None]) -> None: ...

    @abstractmethod
    def on_kline(self, interval: str, cb: Callable[[Kline], None]) -> None: ...

    @abstractmethod
    def on_order_report(self, cb: Callable[[OrderReport], None]) -> None: ...

    @abstractmethod
    def on_position(self, cb: Callable[[Position], None]) -> None: ...

    # lifecycle
    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...
```

## Bybit implementation

- Uses REST for order management, `instruments-info` and position snapshots.
- Public WS feeds `kline` and `ticker` supply prices and closed-candle signals.
- Private WS feeds `order` and `position` report fills and position updates.
- Client order IDs (`cl_id`) ensure idempotency; reconnect logic resubscribes and
  reconciles state after disconnects.

## Paper broker

- Consumes quotes and klines from the local stream.
- Simulates fills using `FeeModel` for commissions and spread.
- Mirrors the interface of the live broker to enable dry runs.
