
from __future__ import annotations
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """Небольшой LSTM-экстрактор для оконных наблюдений (T,D).
    Если используете RecurrentPPO, обычно LSTM уже в политике и экстрактор не нужен.
    Этот вариант на случай обычного PPO на оконном env.
    """
    def __init__(self, observation_space: spaces.Box, hidden_size: int = 64, out_dim: int = 64, num_layers: int = 1):
        assert len(observation_space.shape) == 2, "Expected observation shape (T, D)"
        super().__init__(observation_space, features_dim=out_dim)
        T, D = observation_space.shape
        self.lstm = nn.LSTM(input_size=D, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, out_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: [B, T, D]
        out, (h, c) = self.lstm(obs)
        # берём последний hidden state
        z = h[-1]  # [B, hidden]
        return self.head(z)
