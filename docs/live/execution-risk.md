# Execution planning and risk management

## Position sizing

`PositionSizer` converts model actions {-1,0,+1} into target quantities.
Strategies may use fixed sizes, notional amounts or risk-adjusted sizing.

## Execution planner

`ExecutionPlanner` compares current and target positions and returns a list
of `OrderRequest` objects.

- Rounds quantities to `qty_step` and checks `min_qty`/`max_qty`.
- Uses `reduce_only` when closing positions and `post_only` for maker mode.
- Supports order attributes such as `TIF` and price band checks.

## Risk engine

`RiskEngine` validates actions and orders before submission.

- Enforces limits on position size, daily loss, order quantity and frequency.
- Applies a price band around the mid price.
- Provides a kill switch and circuit breaker on disconnects.
