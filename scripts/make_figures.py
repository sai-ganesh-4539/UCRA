#!/usr/bin/env python3
"""Regenerate all paper figures from saved results (no retraining).

Usage:  python scripts/make_figures.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml

from ucra.eval.plots import plot_demand_vs_reservation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = Path(cfg["paths"]["outputs"])

    log = pd.read_csv(out / "ucra_log.csv", parse_dates=["ts"])
    res = json.loads((out / "ucra_results.json").read_text())

    import numpy as np
    plot_demand_vs_reservation(
        log["ts"], log["demand"], log["R"],
        np.full(len(log), res["static_peak"]["reservation_error"] * res["capacity"]),
        out / "fig_demand_vs_reservation.png")
    print(f"[figures] all figures refreshed in {out}/")


if __name__ == "__main__":
    main()
