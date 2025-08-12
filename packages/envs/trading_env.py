from __future__ import annotations
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

from ..backtest.fees import FeeModel

class SingleAssetDailyEnv(gym.Env):
    """Простой дневной env:
    - Наблюдение: вектор фич за текущий день
    - Действие: {-1, 0, +1} (шорт / плоско / лонг)
    - Награда: позиция[t-1] * return[t] - издержки за изменение позиции
    """
    metadata = {"render_modes": []}

    def __init__(self, features: pd.DataFrame, prices: pd.Series, fee_model: FeeModel|None=None, obs_norm: bool=True):
        assert features.index.equals(prices.index), "features and prices must be aligned"
        self.features = features.astype(np.float32)
        self.prices = prices.astype(np.float32)
        self.returns = self.prices.pct_change().fillna(0.0).astype(np.float32)
        self.fee_model = fee_model or FeeModel(commission=0.001, spread=0.0005)
        self.obs_norm = obs_norm

        self.action_space = spaces.Discrete(3)  # -1, 0, +1
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.features.shape[1],), dtype=np.float32)

        self._t0 = 1  # начнем со второго бара (чтобы была доходность)
        self._t = self._t0
        self._pos = 0  # текущая позиция {-1,0,+1}

        self._mu = self.features.mean()
        self._sd = self.features.std().replace(0, 1)

    def reset(self, *, seed: int|None=None, options=None):
        super().reset(seed=seed)
        self._t = self._t0
        self._pos = 0
        obs = self._obs(self._t)
        return obs, {}

    def _obs(self, t: int):
        x = self.features.iloc[t]
        if self.obs_norm:
            x = (x - self._mu) / self._sd
        return x.values.astype(np.float32)

    def step(self, action: int):
        action = int(action) - 1  # [0,1,2] -> [-1,0,1]
        prev_pos = self._pos
        self._pos = action
        # Доход за текущий шаг базируется на предыдущей позиции
        r = float(prev_pos) * float(self.returns.iloc[self._t])
        turnover = abs(self._pos - prev_pos)
        cost = self.fee_model.cost_for_turnover(turnover)
        reward = r - cost
        self._t += 1
        terminated = self._t >= len(self.features)
        obs = self._obs(self._t-1) if not terminated else self._obs(len(self.features)-1)
        info = {"position": self._pos, "turnover": turnover, "gross_return": r, "cost": cost}
        return obs, reward, terminated, False, info
