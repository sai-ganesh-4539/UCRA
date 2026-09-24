#!/usr/bin/env python3
"""Data audit: load the configured dataset, print stats, save overview plots.

Usage:  python scripts/audit_data.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from ucra.data import load_dataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))

    series, extras = load_dataset(cfg)
    cap = extras.get("capacity", float("nan"))
    print("=" * 60)
    print(f"dataset       : {cfg['data']['dataset']}")
    print(f"length        : {len(series)} slots")
    print(f"span          : {series.index.min()} -> {series.index.max()}")
    print(f"capacity est. : {cap:.1f}")
    print(f"demand min/max/mean : {series.min():.1f} / {series.max():.1f} "
          f"/ {series.mean():.1f}")
    for q in (0.5, 0.9, 0.99):
        print(f"demand q{int(q*100):02d}          : {series.quantile(q):.1f}")
    print(f"missing slots : {int(series.isna().sum())}")
    print("=" * 60)

    out = Path(cfg["paths"]["outputs"])
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), constrained_layout=True)
    axes[0].plot(series.index, series.values, lw=0.9,
                 color="#1f77b4", label="network demand")
    axes[0].axhline(cap, color="#d62728", ls="--", lw=1, label=f"capacity ~{cap:.0f}")
    axes[0].set_title(f"Demand series — {cfg['data']['dataset']}")
    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    axes[1].hist(series.values, bins=50, color="#1f77b4", alpha=0.8)
    axes[1].set_title("Demand distribution")
    axes[1].set_xlabel("demand per 15-min slot")
    p = out / "audit_demand.png"
    fig.savefig(p, dpi=150)
    print(f"[audit] saved {p}")


if __name__ == "__main__":
    main()
