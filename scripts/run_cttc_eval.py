#!/usr/bin/env python3
"""Stage 2-3 evaluation on the CTTC B5G slicing dataset (Zenodo 10610616).

The CTTC dataset provides independent steady-state snapshots of slice
reservations: each (sample, slice) row carries the operator's reservation
(delta * link capacity), the offered traffic, and the observed QoS
(packet-drop ratio, average delay).

Question answered here: if reservations were sized by UCRA's Phi transform
on the empirical offered-load distribution per slice type, would we get
fewer violations than the operator's static delta sizing, at similar
over-provisioning?

Policies (per slice type; quantiles from the TRAIN half of snapshots):
  operator   : reservation as shipped in the dataset (delta * link_cap)
  static_q90 : empirical q90 of offered load (Phi with kappa = 0)
  ucra_phi   : q90 + kappa * spread * (1 + rho_weight * rho_t)   (Stage 2)

Usage:
  python scripts/run_cttc_eval.py --config configs/default.yaml
  python scripts/run_cttc_eval.py --max-samples 100    # quick pass
Saves: outputs/cttc_results.json, outputs/fig_cttc_slices.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from ucra.data.cttc_loader import load_cttc


def phi_empirical(offered_hist: np.ndarray, offered_now: float,
                  phi_cfg: dict) -> float:
    """Stage-2 transform applied to the empirical offered-load distribution."""
    tau = phi_cfg["base_tau"]
    q50 = float(np.quantile(offered_hist, 0.5))
    q_tau = float(np.quantile(offered_hist, tau))
    q99 = float(np.quantile(offered_hist, 0.99))
    spread = max(q99 - q50, 1e-9)
    rho = float(np.clip((offered_now - q50) / spread, 0.0, 1.0))
    return q_tau + phi_cfg["kappa"] * spread * (1.0 + phi_cfg["rho_weight"] * rho)


def eval_policy(sub: pd.DataFrame, R: np.ndarray) -> dict:
    off = sub["offered"].values
    viol = off > R
    util = np.minimum(off / np.maximum(R, 1e-9), 1.0)
    over = (R - off) / np.maximum(R, 1e-9)
    return {
        "n": int(len(sub)),
        "violation_rate": float(np.mean(viol)),
        "utilization": float(np.mean(util)),
        "over_provision": float(np.mean(np.maximum(over, 0.0))),
    }


def main():
    ap = argparse.ArgumentParser(description="UCRA Stage 2-3 eval on CTTC")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--max-samples", type=int, default=250,
                    help="how many CTTC snapshots to parse (250 is plenty)")
    ap.add_argument("--train-frac", type=float, default=0.7)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = Path(cfg["paths"]["outputs"])
    out.mkdir(exist_ok=True)
    phi_cfg = cfg["phi"]

    samples_dir = cfg["data"]["cttc"]["samples_dir"]
    df = load_cttc(samples_dir, max_samples=args.max_samples)
    if df.empty:
        raise SystemExit("[cttc-eval] no samples parsed - fetch the dataset "
                         "first: python scripts/download_data.py --cttc")

    df = df[df["offered"] > 0].reset_index(drop=True)
    cut = int(df["sample"].max() * args.train_frac)
    train_mask = df["sample"] <= cut
    test = df[~train_mask]

    sweep = cfg["eval"]["sweep_kappa"]
    results = {}
    for stype, sub in test.groupby("slice_type"):
        hist = df.loc[train_mask & (df["slice_type"] == stype),
                      "offered"].values
        if len(hist) < 10:
            continue
        R_op = sub["reserved"].values
        R_q90 = np.full(len(sub), float(np.quantile(hist, phi_cfg["base_tau"])))
        R_phi = np.array([phi_empirical(hist, o, phi_cfg)
                          for o in sub["offered"]])
        entry = {
            "operator": eval_policy(sub, R_op),
            "static_q90": eval_policy(sub, R_q90),
            "ucra_phi": eval_policy(sub, R_phi),
        }
        sweep_res = {}
        for k in sweep:
            pc = dict(phi_cfg, kappa=k)
            Rk = np.array([phi_empirical(hist, o, pc) for o in sub["offered"]])
            sweep_res[str(k)] = eval_policy(sub, Rk)
        entry["kappa_sweep"] = sweep_res

        off = sub["offered"].values
        viol = off > R_op
        if viol.any() and (~viol).any():
            entry["operator_qos_gap"] = {
                "drops_violated": float(sub.loc[viol, "drops_ratio"].mean()),
                "drops_ok": float(sub.loc[~viol, "drops_ratio"].mean()),
                "delay_violated": float(sub.loc[viol, "avg_delay"].mean()),
                "delay_ok": float(sub.loc[~viol, "avg_delay"].mean()),
            }
        results[str(stype)] = entry

    (out / "cttc_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: {p: v[p] for p in ("operator", "static_q90", "ucra_phi")}
                      for k, v in results.items()}, indent=2))

    types = sorted(results)
    x = np.arange(len(types))
    w = 0.27
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), constrained_layout=True)
    for ax, key, title in (
            (axes[0], "violation_rate", "Reservation violations by policy"),
            (axes[1], "over_provision", "Over-provisioning by policy")):
        for i, (pol, color) in enumerate((("operator", "#7f7f7f"),
                                          ("static_q90", "#1f77b4"),
                                          ("ucra_phi", "#d62728"))):
            vals = [results[t][pol][key] for t in types]
            ax.bar(x + (i - 1) * w, vals, w, label=pol, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(types)
        ax.set_ylabel(key.replace("_", " "))
        ax.set_title(title)
        ax.legend()
    p = out / "fig_cttc_slices.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("[cttc-eval] saved {} and cttc_results.json".format(p))


if __name__ == "__main__":
    main()
