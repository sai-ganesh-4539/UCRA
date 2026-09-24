"""Stage 1 output assembly: quantile forecasts -> u_t, U_t, rho_t."""
from __future__ import annotations

import numpy as np


class UncertaintyState:
    """Rolling risk state of the network (feedback layer feeds it)."""

    def __init__(self, drift_window: int = 96):
        self.drift_window = drift_window
        self.violation_history: list[int] = []   # 1 = reservation missed demand
        self.recent_demand: list[float] = []
        self.baseline_mu: float | None = None    # train-scale stats
        self.baseline_sd: float = 1.0

    def set_reference(self, train_demand: np.ndarray):
        self.baseline_mu = float(np.mean(train_demand))
        self.baseline_sd = float(np.std(train_demand) + 1e-8)

    def observe(self, violated: bool, demand: float):
        self.violation_history.append(int(violated))
        self.recent_demand.append(float(demand))
        if len(self.violation_history) > self.drift_window:
            self.violation_history.pop(0)
        if len(self.recent_demand) > self.drift_window:
            self.recent_demand.pop(0)

    def rho(self) -> float:
        """Risk indicator rho_t in [0, 1.5]: recent violation rate + drift z."""
        if not self.violation_history:
            return 0.0
        viol_rate = float(np.mean(self.violation_history))
        z = 0.0
        if self.baseline_mu is not None and len(self.recent_demand) >= 8:
            recent = float(np.mean(self.recent_demand))
            z = abs(recent - self.baseline_mu) / self.baseline_sd
        return min(1.5, viol_rate + max(0.0, z - 2.0) / 4.0)


def extract_uncertainty(pred_q: np.ndarray, quantiles: list[float],
                        state: UncertaintyState) -> dict:
    """Turn (horizon, Q) quantile prediction into Stage 1 outputs at time t.

    Returns dict with:
      u_hat : point forecast (median) for the next slot
      spread: U_t = q99 - q50 (uncertainty width used by Phi)
      rho_t : risk indicator from the rolling state
    """
    qi = {round(q, 2): i for i, q in enumerate(quantiles)}
    q50 = pred_q[0, qi[0.5]]
    q99 = pred_q[0, qi[0.99]]
    return {
        "u_hat": float(q50),
        "spread": float(max(0.0, q99 - q50)),
        "rho_t": state.rho(),
        "q_vec": pred_q[0].tolist(),
    }
