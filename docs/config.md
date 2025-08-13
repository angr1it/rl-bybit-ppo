# TrainConfig

`TrainConfig` is a dataclass that captures all parameters needed for a training run.

## Fields

- `symbol`: trading pair, e.g. `"BTC/USDT"`.
- `timeframe`: candle timeframe, default `1d`.
- `start`, `end`: optional date range for training data.
- `window`: number of bars used for feature generation.
- `total_timesteps`: total RL training steps requested.
- `save_path`: where to store the trained model and metadata.
- `seed`: random seed; `None` or `-1` triggers a random value.
- `tb`: enable TensorBoard logging.
- `n_steps`, `batch_size`, `log_interval`: optional overrides for SB3 PPO settings.
- `train_mode`: `"rl"` or `"supervised"`.
- `policy`, `policy_kwargs`: policy class name and constructor kwargs.
- `env_class`, `env_kwargs`: environment class and its parameters.
- `use_wandb`, `wandb_project`, `wandb_run_name`: Weights & Biases settings.
- `agent_class_path`: import path to reconstruct the agent when loading artifacts.

The configuration is stored alongside model artifacts in `.meta.json` files.
