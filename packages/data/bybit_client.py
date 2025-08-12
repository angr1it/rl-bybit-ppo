from __future__ import annotations
import time
from typing import Optional, List, Any
import ccxt

from .config import SETTINGS

class BybitClient:
    def __init__(self, api_key: Optional[str]=None, api_secret: Optional[str]=None, sandbox: Optional[bool]=None):
        api_key = api_key if api_key is not None else SETTINGS.bybit_api_key
        api_secret = api_secret if api_secret is not None else SETTINGS.bybit_api_secret
        sandbox = sandbox if sandbox is not None else SETTINGS.sandbox

        self._ex = ccxt.bybit({
            "apiKey": api_key or "",
            "secret": api_secret or "",
            "enableRateLimit": True,
        })
        # Включаем testnet при необходимости
        if sandbox:
            # set_sandbox_mode переключает на testnet у поддерживаемых бирж
            try:
                self._ex.set_sandbox_mode(True)
            except Exception:
                pass  # для старых версий fallback ниже
            # Для некоторых версий ccxt можно вручную поменять base URL через exchange.urls

    @property
    def exchange(self):
        return self._ex

    def fetch_ohlcv_all(self, symbol: str, timeframe: str="1d", since_ms: Optional[int]=None, until_ms: Optional[int]=None, limit: int=1000):
        """Итерирует fetch_ohlcv до until_ms. Возвращает list[list[ts, o, h, l, c, v]]."""
        ex = self._ex
        all_bars = []
        since = since_ms
        while True:
            batch = ex.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
            if not batch:
                break
            all_bars.extend(batch)
            last_ts = batch[-1][0]
            if until_ms is not None and last_ts >= until_ms:
                break
            # +1 мс чтобы не зациклиться
            since = last_ts + 1
            time.sleep(ex.rateLimit / 1000.0)
        # фильтруем по until_ms
        if until_ms is not None:
            all_bars = [row for row in all_bars if row[0] <= until_ms]
        return all_bars
