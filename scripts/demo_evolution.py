#!/usr/bin/env python3
"""Stage-5 self-evolution demo: inject a demand surge into the TEST window
and show that UCRA detects the drift and adapts (Stage 5), while a frozen
model keeps violating.

Two passes over the same drifted window, both starting from outputs/model.pt:

  frozen   : Stages 1-4 only (no evolution)     -> violations stay high
  evolving : full UCRA loop (Stage 5 enabled)   -> fine-tune fires when the
             rolling violation rate crosses the trigger, then violations fall

The surge is a step multiplier on realized demand (simulates a mass event /
flash crowd on top of the real RAN traces). Nothing else changes.

Usage:
  python scripts/demo_evolution.py --config configs/default.yaml
  python scripts/demo_evolution.py --shift 0.5 --at 0.15 --eval-every 24
Saves: outputs/fig_drift_demo.png, outputs/drift_demo.json
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
import torch
import yaml

from ucra.core.evolve import DriftMonitor, EvolutionEngine
from ucra.core.transform import phi_transform, update_reservation
from ucra.core.uncertainty import UncertaintyState, extract_uncertainty
from ucra.data import load_dataset
from ucra.eval.metrics import summarize
from ucra.features.windowing import Scaler, chronological_split, make_windows
from ucra.models.quantile_lstm import (QuantileLSTM, make_loss,
                                       predict_quantiles)


def fresh_model(mc: dict, ckpt: Path) -> QuantileLSTM:
    m = QuantileLSTM(mc["seq_len"], mc["horizon"], mc["quantiles"],
                     mc["hidden_size"], mc["num_layers"], mc["dropout"])
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    m.load_state_dict(state["state_dict"])
    return m


def run_pass(model, pred_q, Xz, Yz, D, capacity, cfg, mc, scY, train_ref,
             evolving: bool, eval_every: int):
    """One closed-loop pass. Xz/Yz: z-space test windows; D: realized demand."""
    state = UncertaintyState(cfg["update"]["drift_window"])
    state.set_reference(train_ref)
    drift = DriftMonitor(cfg["evolution"])
    drift.set_reference(train_ref)
    evo = EvolutionEngine(model, cfg["evolution"], mc)
    loss_fn = make_loss(mc["quantiles"])

    n = len(D)
    R = np.empty(n)
    viol = np.zeros(n)
    updates = []
    r_prev = float(np.quantile(train_ref, 0.9))

    for t in range(n):
        unc = extract_uncertainty(pred_q[t], mc["quantiles"], state)
        r_target = phi_transform(unc["u_hat"], unc["spread"], unc["rho_t"],
                                 capacity, cfg["phi"])
        r_t = update_reservation(r_prev, r_target, D[t - 1] if t else r_target,
                                 capacity, cfg["update"])
        R[t] = r_t
        v = float(D[t] > R[t])
        viol[t] = v
        state.observe(bool(v), D[t])
        stats = drift.observe(bool(v), D[t])

        if evolving and t > 0 and t % eval_every == 0 and drift.triggered(stats):
            if evo.buffer.X:
                res = evo.finetune(loss_fn)
                if res.get("updated"):
                    updates.append(t)
                    pred_q = scY.inverse(predict_quantiles(model, Xz))
        if evolving:
            evo.remember(Xz[t], Yz[t])
        r_prev = r_t
    return R, viol, updates


def main():
    ap = argparse.ArgumentParser(description="UCRA Stage-5 drift demo")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--shift", type=float, default=0.35,
                    help="surge size; 0.35 = demand multiplied by 1.35")
    ap.add_argument("--at", type=float, default=0.15,
                    help="fraction of the TEST window where the surge starts")
    ap.add_argument("--eval-every", type=int, default=24,
                    help="Stage-5 trigger check cadence (slots)")
    ap.add_argument("--ft-epochs", type=int, default=None,
                    help="override evolution.finetune_epochs for the demo")
    ap.add_argument("--lr-scale", type=float, default=1.0,
                    help="multiply model lr for fine-tuning (demo only)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    mc = dict(cfg["model"])
    if args.ft_epochs:
        cfg["evolution"] = dict(cfg["evolution"], finetune_epochs=args.ft_epochs)
    mc["lr"] = mc["lr"] * args.lr_scale
    out = Path(cfg["paths"]["outputs"])
    out.mkdir(exist_ok=True)
    ckpt = out / "model.pt"
    if not ckpt.exists():
        raise SystemExit("no outputs/model.pt - run scripts/train.py first")

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    series, extras = load_dataset(cfg, verbose=False)
    capacity = extras["capacity"]
    X, Y = make_windows(series.values, mc["seq_len"], mc["horizon"])
    itr, iva, ite = chronological_split(len(X), tuple(mc["train_val_test"]))
    train_ref = series.values[itr[0]:itr[-1] + mc["seq_len"]]

    # ---- inject the surge into the TEST segment of the series ----
    s = series.copy()
    vals = s.values.copy()
    t0 = ite[0] + mc["seq_len"] + int(args.at * len(ite))
    vals[t0:] *= (1.0 + args.shift)
    s[:] = vals

    # windows from the DRIFTED series (the model must forecast what it sees)
    Xd, Yd = make_windows(s.values, mc["seq_len"], mc["horizon"])
    D = Yd[ite][:, 0]
    ts = s.index[ite[0] + mc["seq_len"]: ite[0] + mc["seq_len"] + len(ite)]

    scaler = Scaler()
    z = np.load(out / "scaler.npz")
    scaler.mu, scaler.sd = float(z["mu"]), float(z["sd"])
    scY = Scaler()
    scY.mu, scY.sd = float(z["mu_y"]), float(z["sd_y"])
    Xz = scaler.transform(Xd[ite])
    Yz = scY.transform(Yd[ite])

    # ---------- pass A: frozen (Stages 1-4 only) ----------
    model_a = fresh_model(mc, ckpt)
    pred_q_a = scY.inverse(predict_quantiles(model_a, Xz))
    R_f, viol_f, _ = run_pass(model_a, pred_q_a, Xz, Yz, D, capacity, cfg, mc,
                              scY, train_ref, evolving=False,
                              eval_every=args.eval_every)

    # ---------- pass B: self-evolving (Stage 5 on) ----------
    model_b = fresh_model(mc, ckpt)
    pred_q_b = scY.inverse(predict_quantiles(model_b, Xz))
    R_e, viol_e, upd_e = run_pass(model_b, pred_q_b, Xz, Yz, D, capacity, cfg,
                                  mc, scY, train_ref, evolving=True,
                                  eval_every=args.eval_every)

    d0 = int(args.at * len(ite))

    def post(v):
        return float(np.mean(v[d0:]))

    def util_post(R):
        return float(np.mean(np.minimum(D[d0:] / np.maximum(R[d0:], 1e-9), 1.0)))

    res = {
        "shift": args.shift,
        "surge_start_slot": d0,
        "n_test": int(len(ite)),
        "capacity": float(capacity),
        "frozen": {
            "viol_overall": float(np.mean(viol_f)),
            "viol_post_surge": post(viol_f),
            "utilization_post_surge": util_post(R_f),
            "updates": 0,
        },
        "evolving": {
            "viol_overall": float(np.mean(viol_e)),
            "viol_post_surge": post(viol_e),
            "utilization_post_surge": util_post(R_e),
            "updates": len(upd_e),
            "update_points": upd_e,
        },
    }
    (out / "drift_demo.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))

    W = cfg["update"]["drift_window"]
    roll_f = pd.Series(viol_f).rolling(W, min_periods=8).mean()
    roll_e = pd.Series(viol_e).rolling(W, min_periods=8).mean()

    fig, axes = plt.subplots(2, 1, figsize=(11, 7), constrained_layout=True,
                             sharex=True)
    ax = axes[0]
    ax.plot(ts, D, lw=1.0, color="#1f77b4", label="realized demand (+surge)")
    ax.plot(ts, R_f, lw=1.1, color="#d62728", label="reservation (frozen)")
    ax.plot(ts, R_e, lw=1.1, color="#2ca02c", label="reservation (self-evolving)")
    ax.axhline(capacity, color="#444444", ls=":", lw=0.8)
    ax.axvline(ts[d0], color="k", ls="--", lw=0.8)
    ax.annotate("surge starts", (ts[d0], capacity), rotation=90,
                va="top", ha="right", fontsize=8)
    ax.set_ylabel("demand / reservation")
    ax.set_title("Stage-5 demo: +{:.0f}% demand surge on the test window"
                 .format(100 * args.shift))
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))

    ax = axes[1]
    ax.plot(ts, roll_f, lw=1.2, color="#d62728", label="frozen")
    ax.plot(ts, roll_e, lw=1.2, color="#2ca02c", label="self-evolving")
    ax.axhline(cfg["evolution"]["violation_trigger"], color="#7f7f7f",
               ls=":", lw=1, label="Stage-5 trigger")
    for p in upd_e:
        ax.axvline(ts[p], color="#2ca02c", ls=":", lw=0.9)
    ax.set_xlabel("time")
    ax.set_ylabel("rolling violation rate")
    ax.set_title("Self-evolution: trigger fires, model fine-tunes, violations fall")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))

    p = out / "fig_drift_demo.png"
    fig.savefig(p, dpi=150)
    plt.close(fig)
    print("[demo] saved {}".format(p))


if __name__ == "__main__":
    main()
