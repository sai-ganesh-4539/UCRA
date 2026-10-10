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
  operator        : reservation as shipped in the dataset (delta * link_cap)
  static_q90      : empirical q90 of offered load (Phi with kappa = 0)
  phi_train_only  : q90_train + kappa * (q99_train - q50_train), constant
                    (Stage-2 rule with a train-only, frozen uncertainty)
  ucra_phi        : Stage-2 rule with a CAUSAL online risk indicator:
                    rho_t is computed ONLY from previously served test
                    samples (violation history + demand z, exactly like the
                    RAN loop's UncertaintyState). The current sample's
                    offered load is observed AFTER its reservation is fixed.

CAUSALITY (verified): earlier versions fed `offered_now` of the sample being
served into rho - outcome leakage. The sequential loop below fixes that.

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

from ucra.core.uncertainty import UncertaintyState
from ucra.data.cttc_loader import load_cttc


def phi_empirical(offered_hist: np.ndarray, offered_now: float,
                  phi_cfg: dict) -> float:
    """Stage-2 transform on the empirical train distribution, rho = 0.

    NOTE: no per-sample online term here. The online (causal) variant is
    phi_empirical_causal below; passing offered_now only sizes the base
    quantiles from the train histogram, never from the current outcome.
    """
    tau = phi_cfg["base_tau"]
    q_tau = float(np.quantile(offered_hist, tau))
    q99 = float(np.quantile(offered_hist, 0.99))
    q50 = float(np.quantile(offered_hist, 0.5))
    spread = max(q99 - q50, 1e-9)
    return q_tau + phi_cfg["kappa"] * spread


def phi_empirical_causal(offered_hist: np.ndarray, offered_now: float,
                         phi_cfg: dict, state: UncertaintyState) -> float:
    """Stage-2 rule with a CAUSAL online risk indicator rho_t.

    rho_t comes exclusively from the UncertaintyState feedback of PREVIOUSLY
    served samples (rolling violation rate + demand z-score). offered_now is
    an argument only to keep the call signature uniform - it is NOT used for
    sizing, and the state observes it only after the caller commits R.
    """
    del offered_now  # must not influence the reservation (no leakage)
    tau = phi_cfg["base_tau"]
    q50 = float(np.quantile(offered_hist, 0.5))
    q_tau = float(np.quantile(offered_hist, tau))
    q99 = float(np.quantile(offered_hist, 0.99))
    spread = max(q99 - q50, 1e-9)
    rho = state.rho()          # feedback from past slots only
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


def evaluate_policies(df: pd.DataFrame, cfg: dict, out: Path,
                      train_frac: float = 0.7,
                      make_figures: bool = True) -> dict:
    """Run the 4-policy comparison + kappa sweep on a parsed slice frame.
    Writes cttc_results.json (+ figure) into out; returns the results dict."""
    phi_cfg = cfg["phi"]
    df = df[df["offered"] > 0].reset_index(drop=True)
    cut = int(df["sample"].max() * train_frac)
    train_mask = df["sample"] <= cut
    test = df[~train_mask]

    sweep = cfg["eval"]["sweep_kappa"]
    drift_window = cfg["update"]["drift_window"]
    results = {}
    for stype, sub in test.groupby("slice_type"):
        hist = df.loc[train_mask & (df["slice_type"] == stype),
                      "offered"].values
        if len(hist) < 10:
            continue
        R_op = sub["reserved"].values
        R_q90 = np.full(len(sub), float(np.quantile(hist, phi_cfg["base_tau"])))
        # train-only Phi: kappa buffer on train quantiles, frozen, no online term
        R_tr = np.full(len(sub), phi_empirical(hist, 0.0, phi_cfg))
        # causal UCRA Phi: sequential loop, rho from previously served samples
        state = UncertaintyState(drift_window)
        state.set_reference(hist)
        R_phi = []
        for o in sub["offered"].values:
            R_phi.append(phi_empirical_causal(hist, o, phi_cfg, state))
            state.observe(bool(o > R_phi[-1]), float(o))   # feedback AFTER serving
        R_phi = np.array(R_phi)
        entry = {
            "operator": eval_policy(sub, R_op),
            "static_q90": eval_policy(sub, R_q90),
            "phi_train_only": eval_policy(sub, R_tr),
            "ucra_phi": eval_policy(sub, R_phi),
            "causal_rho": True,
            "n_train": int(train_mask.sum()),
            "n_test": int(len(sub)),
        }
        sweep_res = {}
        for k in sweep:
            pc = dict(phi_cfg, kappa=k)
            st = UncertaintyState(drift_window)
            st.set_reference(hist)
            Rk = []
            for o in sub["offered"].values:
                Rk.append(phi_empirical_causal(hist, o, pc, st))
                st.observe(bool(o > Rk[-1]), float(o))
            sweep_res[str(k)] = eval_policy(sub, np.array(Rk))
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

    results["_meta"] = {"n_samples_parsed": int(df["sample"].nunique()),
                        "train_frac": train_frac, "causal_rho": True}
    (out / "cttc_results.json").write_text(json.dumps(results, indent=2))
    print(json.dumps({k: {p: v[p] for p in ("operator", "static_q90",
                                            "phi_train_only", "ucra_phi")}
                      for k, v in results.items() if not k.startswith("_")},
                     indent=2))

    if not make_figures:
        return results

    types = sorted(k for k in results if not k.startswith("_"))
    x = np.arange(len(types))
    w = 0.2
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2), constrained_layout=True)
    for ax, key, title in (
            (axes[0], "violation_rate", "Reservation violations by policy"),
            (axes[1], "over_provision", "Over-provisioning by policy")):
        for i, (pol, color) in enumerate((("operator", "#7f7f7f"),
                                          ("static_q90", "#1f77b4"),
                                          ("phi_train_only", "#9467bd"),
                                          ("ucra_phi", "#d62728"))):
            vals = [results[t][pol][key] for t in types]
            ax.bar(x + (i - 1.5) * w, vals, w, label=pol, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(types)
        ax.set_ylabel(key.replace("_", " "))
        ax.set_title(title)
        ax.legend(fontsize=8)
    p = out / "fig_cttc_slices.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("[cttc-eval] saved {} and cttc_results.json".format(p))
    return results


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

    samples_dir = cfg["data"]["cttc"]["samples_dir"]
    df = load_cttc(samples_dir, max_samples=args.max_samples)
    if df.empty:
        raise SystemExit("[cttc-eval] no samples parsed - fetch the dataset "
                         "first: python scripts/download_data.py --cttc")
    evaluate_policies(df, cfg, out, train_frac=args.train_frac)


if __name__ == "__main__":
    main()
