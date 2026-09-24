"""Synthetic bursty demand generator (UCRA smoke tests & demos).

Produces a 15-min-resolution multi-sector demand series with:
- diurnal + weekly seasonality
- bursty ON/OFF surges (PPBP-flavoured)
- AR(1) noise
so the full UCRA pipeline can run without the Zenodo downloads.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic(n_steps: int = 2016, n_sectors: int = 8,
                       seed: int = 42, base: float = 100.0) -> pd.DataFrame:
    """Return long-format DataFrame [ts, sector, demand] at 15-min resolution."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-01-01", periods=n_steps, freq="15min")
    t = np.arange(n_steps)

    diurnal = 1.0 + 0.45 * np.sin(2 * np.pi * t / 96 - np.pi / 2)      # daily
    weekly = 1.0 + 0.15 * np.sin(2 * np.pi * t / (96 * 7))             # weekly
    noise = np.zeros(n_steps)
    for i in range(1, n_steps):                                        # AR(1)
        noise[i] = 0.85 * noise[i - 1] + rng.normal(0, 0.05)

    frames = []
    for s in range(n_sectors):
        phase = rng.uniform(0, 2 * np.pi)
        sector_gain = rng.uniform(0.7, 1.3)
        bursts = np.zeros(n_steps)
        on = 0
        for i in range(n_steps):                                       # ON/OFF bursts
            if on == 0 and rng.random() < 0.01:
                on = int(rng.integers(2, 10))
            if on > 0:
                bursts[i] = rng.uniform(0.5, 1.4)
                on -= 1
        demand = base * sector_gain * diurnal * weekly * (
            1.0 + noise + 0.35 * np.sin(t / 96 + phase) + bursts)
        demand = np.clip(demand, 1.0, None)
        frames.append(pd.DataFrame({
            "ts": ts, "sector": f"sec{s:02d}", "demand": demand.round(3)}))

    return pd.concat(frames, ignore_index=True)


def synthetic_capacity(n_sectors: int = 8, per_sector: float = 250.0) -> float:
    """Network capacity used for reservation clipping (sum over sectors)."""
    return per_sector * n_sectors
