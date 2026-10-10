"""Baseline reservation policies UCRA is compared against."""
from __future__ import annotations

import numpy as np


def static_peak(train_demand: np.ndarray, n_test: int) -> np.ndarray:
    """Reserve the historical peak everywhere (worst-case static sizing)."""
    return np.full(n_test, float(np.max(train_demand)))


def mean_forecast(pred_q: np.ndarray, quantiles: list[float]) -> np.ndarray:
    """Reserve the point forecast (median) only - ignores uncertainty."""
    qi = {round(q, 2): i for i, q in enumerate(quantiles)}
    return pred_q[:, 0, qi[0.5]].copy()


def oracle_quantile(test_demand: np.ndarray, horizon: int,
                    tau: float = 0.9) -> np.ndarray:
    """CHEATING upper bound: the true rolling quantile of the test window."""
    out = np.empty(len(test_demand))
    w = max(48, horizon * 4)
    for i in range(len(test_demand)):
        lo = max(0, i - w)
        out[i] = np.quantile(test_demand[lo:i + 1], tau) if i > 0 else test_demand[0]
    return out


def train_quantile_phi(train_demand: np.ndarray, n_test: int,
                       phi_cfg: dict) -> np.ndarray:
    """Train-only Phi reservation (static, uncertainty-aware).

    Applies the SAME kappa-buffer rule as Stage 2, but with u_hat, q_tau and
    spread taken from the TRAINING distribution only and frozen for the whole
    test window:   R = q90_train + kappa * (q99_train - q50_train).
    No online forecasts, no risk feedback, no EWMA. Isolates what the
    adaptive Stage 1-4 loop adds over a static uncertainty buffer.
    """
    q50 = float(np.quantile(train_demand, 0.5))
    q_tau = float(np.quantile(train_demand, phi_cfg["base_tau"]))
    q99 = float(np.quantile(train_demand, 0.99))
    spread = max(q99 - q50, 1e-9)
    r = q_tau + phi_cfg["kappa"] * spread
    return np.full(n_test, r)
