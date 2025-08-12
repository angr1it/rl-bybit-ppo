from __future__ import annotations

class FeeModel:
    """Простейшая модель издержек: комиссия как доля оборота + спред как половина bid-ask."""
    def __init__(self, commission: float=0.001, spread: float=0.0005):
        self.commission = commission
        self.spread = spread

    def cost_for_turnover(self, turnover: float) -> float:
        # Грубая оценка издержек за единицу оборота
        return turnover * (self.commission + self.spread)
