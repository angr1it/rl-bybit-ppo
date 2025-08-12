
from __future__ import annotations
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from gymnasium import spaces

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)

class TransformerFeatureExtractor(BaseFeaturesExtractor):
    """Крошечный Transformer для оконных наблюдений (Box shape=(T, D)).
    Преобразует [B,T,D] -> [B, out_dim] через CLS-токен + пуллинг.
    """
    def __init__(self, observation_space: spaces.Box, d_model: int = 32, nhead: int = 4, num_layers: int = 2, out_dim: int = 64):
        assert len(observation_space.shape) == 2, "Expected observation shape (T, D)"
        super().__init__(observation_space, features_dim=out_dim)
        T, D = observation_space.shape
        self.in_proj = nn.Linear(D, d_model)
        self.pos = PositionalEncoding(d_model, max_len=T+1)
        enc_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))  # learnable [CLS]
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, out_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # obs: [B, T, D]
        B, T, D = obs.shape
        x = self.in_proj(obs)  # [B,T,d_model]
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, T+1, d_model]
        x = self.pos(x)
        z = self.encoder(x)[:, 0]  # CLS token
        return self.head(z)
