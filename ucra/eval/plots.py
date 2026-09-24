"""Paper-grade figures (PNG, 150 dpi, constrained layout)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plots] saved {path}")


def plot_demand_vs_reservation(ts, demand, R_ucra, R_static, path):
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    ax.plot(ts, demand, lw=1.0, label="Realized demand", color="#1f77b4")
    ax.plot(ts, R_ucra, lw=1.2, label="UCRA reservation", color="#d62728")
    ax.plot(ts, R_static, lw=1.0, ls="--", label="Static-peak baseline", color="#7f7f7f")
    ax.set_xlabel("time")
    ax.set_ylabel("demand / reservation")
    ax.set_title("UCRA reservation tracks uncertain demand")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _save(fig, path)


def plot_quantile_band(ts, realized, pred_q, quantiles, path, n=240):
    """Predicted quantile fan chart vs realized demand (first n test slots)."""
    qi = {round(q, 2): i for i, q in enumerate(quantiles)}
    lo, hi, med = qi[0.05], qi[0.99], qi[0.5]
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    tt = ts[:n]
    ax.fill_between(tt, pred_q[:n, 0, lo], pred_q[:n, 0, hi],
                    alpha=0.25, color="#1f77b4", label="q05-q99 band (U_t)")
    ax.plot(tt, pred_q[:n, 0, med], color="#1f77b4", lw=1.0, label="median forecast")
    ax.plot(tt, realized[:n], color="#ff7f0e", lw=0.9, label="realized")
    ax.set_xlabel("time")
    ax.set_ylabel("demand")
    ax.set_title("Stage 1: quantile uncertainty band vs realized demand")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _save(fig, path)


def plot_kappa_sweep(kappas, viol, util, path):
    """Risk-utility trade-off: kappa sweep (Phi risk aversion)."""
    fig, ax1 = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax1.plot(kappas, viol, "o-", color="#d62728", label="violation rate")
    ax1.set_xlabel("kappa (risk aversion in Phi)")
    ax1.set_ylabel("violation rate", color="#d62728")
    ax2 = ax1.twinx()
    ax2.plot(kappas, util, "s--", color="#1f77b4", label="utilization")
    ax2.set_ylabel("utilization", color="#1f77b4")
    ax1.set_title("Phi risk-utility trade-off")
    _save(fig, path)


def plot_training_history(hist: dict, path: str):
    fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    ax.plot(hist["train"], label="train pinball loss")
    ax.plot(hist["val"], label="val pinball loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Quantile LSTM training")
    ax.legend()
    _save(fig, path)


def plot_evolution(ts, viol_rolling, update_points, path, window=96):
    """Rolling violation rate with markers where self-evolution fired."""
    fig, ax = plt.subplots(figsize=(11, 3.2), constrained_layout=True)
    ax.plot(ts, viol_rolling, color="#9467bd", lw=1.2,
            label=f"rolling violation rate ({window}-slot)")
    for p in update_points:
        if 0 <= p < len(ts):
            ax.axvline(ts[p], color="#2ca02c", ls=":", lw=0.8)
    ax.set_xlabel("time")
    ax.set_ylabel("violation rate")
    ax.set_title("Stage 5: self-evolving fine-tuning events (green lines)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _save(fig, path)
