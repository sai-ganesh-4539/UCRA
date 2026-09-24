"""Feedback & Evaluation layer metrics (all computed on the TEST window)."""
from __future__ import annotations

import numpy as np


def reservation_error(R: np.ndarray, D: np.ndarray,
                      capacity: float) -> float:
    """Mean absolute gap between reservation and realized demand, normalised."""
    return float(np.mean(np.abs(R - D)) / max(capacity, 1e-9))


def violation_rate(R: np.ndarray, D: np.ndarray) -> float:
    """Risk exposure: fraction of slots where demand exceeded the reservation."""
    return float(np.mean(D > R))


def utilization(R: np.ndarray, D: np.ndarray) -> float:
    """Mean reserved-capacity utilization (capped at 1 per slot)."""
    return float(np.mean(np.minimum(D / np.maximum(R, 1e-9), 1.0)))


def waste(R: np.ndarray, D: np.ndarray, capacity: float) -> float:
    """Over-provisioned (unused reservation) share of capacity."""
    return float(np.mean(np.maximum(R - D, 0.0)) / max(capacity, 1e-9))


def summarize(R: np.ndarray, D: np.ndarray, capacity: float) -> dict:
    return {
        "reservation_error": reservation_error(R, D, capacity),
        "violation_rate": violation_rate(R, D),
        "utilization": utilization(R, D),
        "waste": waste(R, D, capacity),
    }
