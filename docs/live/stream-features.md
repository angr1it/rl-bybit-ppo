# Market data stream and online features

## Stream

`stream.py` delivers `Quote` and `Kline` events to subscribers.

- WebSocket is the primary source; REST polling acts as a fallback and
  reconciliation mechanism.
- Only closed candles (`Kline.confirmed=True`) are forwarded so that the
  agent acts on bar close for the next period.

## OnlineFeaturePipe

`features_live.py` recalculates features on each closed candle using the
same logic as the offline pipeline.

- Maintains a dataframe of OHLCV history.
- Recomputes features and normalizes them with `mu` and `sd` saved in the
  training artifact.
- Enforces the exact column order from `feature_stats['columns']`.
