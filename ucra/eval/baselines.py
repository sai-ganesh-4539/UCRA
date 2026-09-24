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
