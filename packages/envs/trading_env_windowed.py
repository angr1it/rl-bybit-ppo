
from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from ..backtest.fees import FeeModel

class WindowedSingleAssetEnv(gym.Env):
    """Оконный вариант env для трансформеров.
    - Наблюдение: матрица [window, feat_dim]
    - Действие: {-1,0,+1}
    - Награда: pos[t-1]*ret[t] - cost(turnover)
    """
    metadata = {"render_modes": []}

    def __init__(self, features: pd.DataFrame, prices: pd.Series, window: int=64, fee_model: FeeModel|None=None, obs_norm: bool=True):
        assert features.index.equals(prices.index), "features and prices must be aligned"
        assert len(features) > window + 1, "Need enough bars for a windowed env"
        self.features = features.astype(np.float32)
        self.prices = prices.astype(np.float32)
        self.returns = self.prices.pct_change().fillna(0.0).astype(np.float32)
        self.fee_model = fee_model or FeeModel(commission=0.001, spread=0.0005)
        self.obs_norm = obs_norm
        self.window = int(window)

        self.feat_mu = self.features.rolling(self.window, min_periods=1).mean().fillna(0)
        self.feat_sd = self.features.rolling(self.window, min_periods=1).std().replace(0, 1).fillna(1)

        feat_dim = self.features.shape[1]
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.window, feat_dim), dtype=np.float32)

        self._t0 = self.window  # первый доступный бар
        self._t = self._t0
        self._pos = 0

    def reset(self, *, seed: int|None=None, options=None):
        super().reset(seed=seed)
        self._t = self._t0
        self._pos = 0
        return self._obs(self._t), {}

    def _obs(self, t: int):
        sl = slice(t - self.window, t)
        X = self.features.iloc[sl]
        if self.obs_norm:
            mu = self.feat_mu.iloc[sl]
            sd = self.feat_sd.iloc[sl]
            X = (X - mu) / sd
        return X.values.astype(np.float32)

    def step(self, action: int):
        action = int(action) - 1  # [0,1,2] -> [-1,0,1]
        prev_pos = self._pos
        self._pos = action
        r = float(prev_pos) * float(self.returns.iloc[self._t])
        turnover = abs(self._pos - prev_pos)
        cost = self.fee_model.cost_for_turnover(turnover)
        reward = r - cost
        self._t += 1
        terminated = self._t >= len(self.features)
        obs = self._obs(self._t) if not terminated else self._obs(len(self.features))
        info = {"position": self._pos, "turnover": turnover, "gross_return": r, "cost": cost}
        return obs, reward, terminated, False, info
