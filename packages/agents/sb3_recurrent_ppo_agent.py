from __future__ import annotations
from typing import Optional, Dict, Any
import numpy as np

from sb3_contrib import RecurrentPPO


class SB3RecurrentPPOAgent:
    """Агент на RecurrentPPO (sb3-contrib) с LSTM-политикой."""

    uses_rl = True

    def __init__(
        self,
        policy: str = "MlpLstmPolicy",
        policy_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.policy = policy
        self.policy_kwargs = policy_kwargs or {}
        self.model: RecurrentPPO | None = None
        self._state = None
        self._episode_start = np.array([True], dtype=bool)

    def reset(self):
        """Сброс скрытого состояния для начала нового эпизода/бэктеста."""
        self._state = None
        self._episode_start = np.array([True], dtype=bool)

    def fit(self, vec_env, total_timesteps: int, **ppo_kwargs):
        ppo_kwargs.pop("policy", None)
        ppo_kwargs.pop("policy_kwargs", None)

        self.model = RecurrentPPO(
            policy=self.policy,
            env=vec_env,
            policy_kwargs=self.policy_kwargs,
            **ppo_kwargs,
        )
        self.model.learn(total_timesteps=total_timesteps)

    def act(self, obs, deterministic: bool = True) -> int:
        assert self.model is not None, "Model not trained/loaded"
        action, self._state = self.model.predict(
            obs,
            state=self._state,
            episode_start=self._episode_start,
            deterministic=deterministic,
        )
        # со второго шага эпизода флаг = False
        self._episode_start[:] = False
        return int(action)

    def save(self, path: str) -> None:
        assert self.model is not None, "Model not trained/loaded"
        self.model.save(path)

    @classmethod
    def load(cls, path: str) -> "SB3RecurrentPPOAgent":
        agent = cls()
        agent.model = RecurrentPPO.load(path)
        agent.reset()
        return agent

    def to_meta(self) -> dict:
        return {
            "agent_class": "packages.agents.sb3_recurrent_ppo_agent.SB3RecurrentPPOAgent",
            "policy": self.policy,
            "policy_kwargs": self.policy_kwargs,
        }

    @classmethod
    def from_meta(cls, meta: dict) -> "SB3RecurrentPPOAgent":
        return cls(
            policy=meta.get("policy", "MlpLstmPolicy"),
            policy_kwargs=meta.get("policy_kwargs") or {},
        )
