
from __future__ import annotations
import argparse
import torch
import torch.nn as nn
import pandas as pd
from torchrl.envs.libs.gym import GymEnv
from torchrl.objectives.ppo import ClipPPOLoss
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data import TensorDict
from torchrl.objectives.value import GAE
from torch.optim import Adam

from ..data.ohlcv_store import parquet_path
from ..env.features import make_features
from ..env.trading_env_windowed import WindowedSingleAssetEnv
from ..backtest.fees import FeeModel

class TinyTransformerPolicy(nn.Module):
    def __init__(self, obs_shape, n_actions: int = 3, d_model: int = 32, nhead: int = 4, layers: int = 1):
        super().__init__()
        T, D = obs_shape
        self.in_proj = nn.Linear(D, d_model)
        layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.policy = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, n_actions))
        self.value = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

    def forward(self, obs):
        # obs: [B,T,D]
        x = self.in_proj(obs)
        cls = self.cls.expand(x.size(0), -1, -1)
        x = torch.cat([cls, x], dim=1)
        z = self.encoder(x)[:, 0]
        logits = self.policy(z)
        value = self.value(z)
        return logits, value

def make_env(df, window: int):
    feats = make_features(df, window=window)
    prices = df.loc[feats.index, "close"]
    return WindowedSingleAssetEnv(features=feats, prices=prices, window=window, fee_model=FeeModel())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbol", required=True)
    ap.add_argument("--timeframe", default="1d")
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--steps", type=int, default=5000)
    args = ap.parse_args()

    df = pd.read_parquet(parquet_path(args.symbol, args.timeframe)).sort_index()
    env = make_env(df, window=args.window)
    gym_env = GymEnv(env, device="cpu")  # TorchRL обёртка

    obs_spec = gym_env.observation_spec
    action_spec = gym_env.action_spec
    T, D = env.observation_space.shape
    model = TinyTransformerPolicy((T, D), n_actions=action_spec.space.n)

    optimizer = Adam(model.parameters(), lr=3e-4)
    advantage = GAE(gamma=0.99, lmbda=0.95, value_network=lambda td: model(td["observation"])[1])
    loss_module = ClipPPOLoss(
        actor_network=lambda td: model(td["observation"])[0],
        critic_network=lambda td: model(td["observation"])[1],
        advantage_module=advantage,
        entropy_eps=1e-3
    )

    collector = SyncDataCollector(
        gym_env,
        policy=lambda td: {"action": torch.distributions.Categorical(logits=model(td["observation"])[0]).sample()},
        frames_per_batch=64,
        total_frames=args.steps,
        device="cpu"
    )

    for batch in collector:
        # TorchRL возвращает tensordict; приведём поля к ожидаемым именам
        td = batch.clone()
        # Вычислим лосс PPO
        loss_vals = loss_module(td)
        loss = loss_vals["loss_objective"] + 0.5*loss_vals["loss_critic"] - 0.01*loss_vals["loss_entropy"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print("Finished TorchRL training demo.")

if __name__ == "__main__":
    main()
