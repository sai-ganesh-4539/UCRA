"""Windowing, chronological splits and scaling (Stage 1 preprocessing)."""
from __future__ import annotations

import numpy as np


def make_windows(series, seq_len: int, horizon: int):
    """Sliding windows from a 1-D series.

    Returns X (N, seq_len), Y (N, horizon) float32 arrays.
    """
    arr = np.asarray(series, dtype=np.float32)
    if len(arr) < seq_len + horizon:
        raise ValueError(f"series too short: {len(arr)} < {seq_len}+{horizon}")
    X, Y = [], []
    for i in range(len(arr) - seq_len - horizon + 1):
        X.append(arr[i:i + seq_len])
        Y.append(arr[i + seq_len:i + seq_len + horizon])
    return np.stack(X), np.stack(Y)


def chronological_split(n: int, fractions=(0.7, 0.15, 0.15)):
    """Return (train_idx, val_idx, test_idx) WITHOUT shuffling (time series!)."""
    a = int(n * fractions[0])
    b = int(n * (fractions[0] + fractions[1]))
    return np.arange(0, a), np.arange(a, b), np.arange(b, n)


class Scaler:
    """z-score scaler fitted on the TRAIN slice only (avoids leakage)."""

    def __init__(self):
        self.mu = 0.0
        self.sd = 1.0

    def fit(self, X: np.ndarray):
        self.mu = float(X.mean())
        self.sd = float(X.std() + 1e-8)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu) / self.sd

    def inverse(self, X: np.ndarray) -> np.ndarray:
        return X * self.sd + self.mu
