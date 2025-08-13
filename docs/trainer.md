# Trainer

`Trainer` orchestrates data preparation, environment creation and agent training.

## Workflow

1. Load price data via `TrainConfig` parameters.
2. Generate technical features with `make_features` and align prices.
3. Build Gymnasium environments and wrap them into a `VecEnv`.
4. Delegate learning to the selected agent.
5. Save model weights and a `.meta.json` file describing the run.

## Key Methods

- `_load_price_df(symbol, timeframe, start, end)` – reads parquet data for the requested range.
- `_prepare_data()` – builds features, determines effective timesteps and observation shape.
- `train()` *(implicit in agent.fit)* – agent-specific training loop.
- Metadata helpers convert complex objects into JSON‑serializable form.
