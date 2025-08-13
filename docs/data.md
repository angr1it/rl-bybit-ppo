# Data and Backtesting Layer

This layer handles market data access and simple backtests.

## Components

- `packages/data/bybit_client.py` – wrappers around ccxt for fetching OHLCV data.
- `packages/data/ohlcv_store.py` – functions for storing and loading candles in Parquet.
- `packages/backtest/` – utilities for fee modelling and running quick vectorbt-based backtests.

## Responsibilities

1. Download OHLCV candles and persist them locally.
2. Provide feature engineering helpers used by environments.
3. Serve prices and features to the RL layer and offline evaluation tools.
4. Supply fee models and vectorbt runners for baseline backtests.
