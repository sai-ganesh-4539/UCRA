"""Feedback & Evaluation layer metrics (all computed on the TEST window)."""
from __future__ import annotations

import numpy as np
import pandas as pd


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


def qos_gap(sub: pd.DataFrame, R: np.ndarray) -> dict | None:
    """Service-performance feedback (architecture block: latency/reliability).

    Splits the evaluated snapshots by whether the policy's reservation was
    violated (offered > R) and compares the observed QoS columns of the CTTC
    dataset (``drops_ratio`` = packet-drop ratio, ``avg_delay`` = average
    delay) across the two groups. Returns None when either group is empty
    (the gap is then undefined, e.g. a policy with zero violations).
    """
    off = np.asarray(sub["offered"].values, dtype=float)
    R = np.asarray(R, dtype=float)
    viol = off > R
    if viol.all() or not viol.any():
        return None
    drops = np.asarray(sub["drops_ratio"].values, dtype=float)
    delay = np.asarray(sub["avg_delay"].values, dtype=float)
    return {
        "n_violated": int(viol.sum()),
        "n_ok": int((~viol).sum()),
        "drops_violated": float(drops[viol].mean()),
        "drops_ok": float(drops[~viol].mean()),
        "delay_violated": float(delay[viol].mean()),
        "delay_ok": float(delay[~viol].mean()),
    }
