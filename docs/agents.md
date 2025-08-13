# Agents

Agents encapsulate policies and learning algorithms. They expose a common interface used by `Trainer`.

## Implementations

### SB3PPOAgent
- Wrapper around `stable_baselines3.PPO`.
- Methods: `fit(vec_env, total_timesteps, **ppo_kwargs)`, `act(obs)`, `save(path)`, `load(path)`, `to_meta()`, `from_meta()`.

### SB3RecurrentPPOAgent
- Uses `sb3_contrib.RecurrentPPO` with LSTM policies for partially observable setups.
- Shares the same interface as `SB3PPOAgent`.

### SklearnAgent
- Baseline supervised agent leveraging scikit-learn classifiers.
- Used for quick experiments outside RL.

All agents set a `uses_rl` flag indicating whether RL training is expected.
