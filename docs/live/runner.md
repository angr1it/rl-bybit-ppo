# Live runner

`runner.py` orchestrates the live trading loop.

1. Initializes broker, feature pipeline, position sizer and risk engine.
2. Subscribes to quote, kline, order and position streams.
3. On each confirmed candle:
   - updates features and obtains the latest observation;
   - queries the agent (supports RecurrentPPO with state and `episode_starts`);
   - checks risk limits and plans execution orders;
   - submits allowed orders via the broker.
4. Handles reconnects by reconciling positions and open orders.

The runner ensures that decisions are made on bar close and applied to the
next interval, preserving offline/live alignment.
