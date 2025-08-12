from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
import joblib


class SklearnAgent:
    """Пример: бинарная классификация → действие {-1, +1} по порогу."""

    uses_rl = False  # маркер для Trainer

    def __init__(self, threshold: float = 0.5):
        self.model = LogisticRegression(max_iter=1000)
        self.threshold = threshold

    # Супервизия: fit(X, y)
    def fit(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)

    def act(self, obs: np.ndarray, deterministic: bool = True) -> int:
        x = obs.reshape(-1) if obs.ndim > 1 else obs
        p = float(self.model.predict_proba([x])[0, 1])
        return int(p > self.threshold) * 2 - 1

    def save(self, path: str) -> None:
        joblib.dump((self.model, self.threshold), path)

    @classmethod
    def load(cls, path: str) -> "SklearnAgent":
        model, thr = joblib.load(path)
        agent = cls(threshold=thr)
        agent.model = model
        return agent

    def to_meta(self) -> dict:
        return {
            "agent_class": "packages.agents.sklearn_agent.SklearnAgent",
            "threshold": self.threshold,
        }

    @classmethod
    def from_meta(cls, meta: dict) -> "SklearnAgent":
        return cls(threshold=meta.get("threshold", 0.5))
