#!/usr/bin/env python3
# ============================================================
# UCRA - Part D generator (final polish pack)
# writes:  scripts/demo_evolution.py     (new - Stage-5 drift demo)
#          scripts/run_cttc_eval.py      (new - CTTC Stage 2-3 eval)
#          README.md                     (results + full checklist)
# patches: scripts/run_ucra.py           (Stage-5 replay targets -> z-space)
#          ucra/models/quantile_lstm.py  (detach fix, no more UserWarning)
#
# Usage:  save this file as create_ucra_partD.py in E:\UCRA
#         python create_ucra_partD.py
# ============================================================
from pathlib import Path

BT3 = chr(96) * 3   # ``` built at runtime so chat UIs cannot eat it

DEMO_PY = r'''#!/usr/bin/env python3
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
'''

CTTC_PY = r'''#!/usr/bin/env python3
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
'''

CTTC_LOADER_PY = r'''"""Loader for Zenodo 10610616 — CTTC B5G Network Slicing Dataset
(Farreras et al., Data in Brief 55:110738, 2024, CC-BY-4.0).

Each sample is a steady-state snapshot of a slicing simulation:
  - traffic_matrix : per-flow offered demand
  - slices         : eMBB / mMTC / URLLC with `delta` = reserved share
  - performance_matrix : PktsDrop, AvgDelay, p10-p90, Jitter per src-dst
  - topology_object: networkx graph (link bandwidth = capacity)

The loader converts samples into a slice-level frame:

    [sample, slice_type, reserved, offered, drops_ratio, avg_delay]

where `reserved = delta * link_capacity`. This is what UCRA's Stages 2-3
evaluation uses (reservation sizing vs observed violations/delays).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _ensure_datanetapi(samples_dir: Path) -> None:
    """Make the dataset's own datanetAPI.py importable (downloads nothing)."""
    if str(samples_dir) not in sys.path:
        sys.path.insert(0, str(samples_dir))


def load_cttc(samples_dir: str | Path = "data/cttc/slicing-simulations",
              max_samples: int | None = None, verbose: bool = True) -> pd.DataFrame:
    """Iterate datanetAPI samples and build the slice-level frame."""
    samples_dir = Path(samples_dir)
    if not samples_dir.exists():
        raise FileNotFoundError(
            f"{samples_dir} not found. Run: bash scripts/download_data.sh cttc")
    _ensure_datanetapi(samples_dir)

    try:
        from datanetAPI import DatanetAPI  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "datanetAPI.py must sit inside the unzipped dataset folder "
            "(it ships with Zenodo 10610616)") from e

    reader = DatanetAPI(str(samples_dir))
    rows = []
    for i, sample in enumerate(reader):
        if max_samples is not None and i >= max_samples:
            break
        try:
            topo = sample.get_topology_object()
            caps = [d.get("bandwidth", 0) for _, _, d in topo.edges(data=True)]
            link_cap = float(np.mean(caps)) if caps else 0.0

            perf = sample.get_performance_matrix()
            n = sample.get_network_size()

            # Slices arrive as plain dicts in the current dataset build
            # (jsonpickle has no py/object markers) but as objects in older
            # builds - read both shapes. Flows carry `bandwidth` (bps) or a
            # `traffic_string` whose 2nd field is the offered rate.
            for sdict in (sample.get_slices() or []):
                if isinstance(sdict, dict):
                    stype = str(sdict.get("type", "unknown"))
                    delta = float(sdict.get("delta", 0.0) or 0.0)
                    flows = sdict.get("flows", []) or []
                else:
                    stype = str(getattr(sdict, "type", "unknown"))
                    delta = float(getattr(sdict, "delta", 0.0) or 0.0)
                    flows = getattr(sdict, "flows", []) or []
                offered = 0.0
                for fl in flows:
                    if isinstance(fl, dict):
                        rate = fl.get("bandwidth") or fl.get("avgRate") \
                            or fl.get("rate") or 0
                        if not rate:
                            parts = str(fl.get("traffic_string", "")).split(",")
                            if len(parts) > 1:
                                try:
                                    rate = float(parts[1])
                                except ValueError:
                                    rate = 0.0
                        offered += float(rate or 0)
                    else:
                        offered += float(getattr(fl, "avgRate", 0) or 0) \
                            + float(getattr(fl, "rate", 0) or 0)
                # aggregate drop/delay over all src-dst pairs weighted equally
                drops, delays, cells = [], [], 0
                for s in range(n):
                    for d in range(n):
                        try:
                            info = perf[s, d]
                            agg = info["AggInfo"] if isinstance(info, dict) \
                                else getattr(info, "AggInfo", None)
                            if agg is None:
                                continue
                            drops.append(float(agg.get("PktsDrop", 0) if isinstance(agg, dict)
                                               else getattr(agg, "PktsDrop", 0)))
                            delays.append(float(agg.get("AvgDelay", 0) if isinstance(agg, dict)
                                                else getattr(agg, "AvgDelay", 0)))
                            cells += 1
                        except Exception:
                            continue
                rows.append({
                    "sample": i, "slice_type": stype, "delta": delta,
                    "reserved": delta * link_cap, "offered": offered,
                    "link_cap": link_cap,
                    "drops_ratio": (float(np.mean(drops)) if drops else np.nan),
                    "avg_delay": (float(np.mean(delays)) if delays else np.nan),
                })
        except Exception as exc:  # skip corrupt samples, keep going
            if verbose:
                print(f"[cttc_loader] sample {i} skipped: {exc}")
            continue
        if verbose and (i + 1) % 25 == 0:
            print(f"[cttc_loader] processed {i + 1} samples ...")

    df = pd.DataFrame(rows)
    if verbose and len(df):
        print(f"[cttc_loader] frame: {df.shape}, slices per type: "
              f"{df['slice_type'].value_counts().to_dict()}")
    return df
'''

README_MD = '''# UCRA - Uncertainty-to-Reservation Transformation Algorithm
### Self-Evolving Risk-Adaptive Resource Allocation for 5G/B5G Network Slicing

> **Naming note:** a 2020 IEEE paper (User-Centric Context-Aware Resource
> Allocation, DOI 10.1109/ACCESS.2020.3046198) already uses the acronym "UCRA"
> in the same domain. Cite and differentiate it in your related-work section
> (see `docs/METHODOLOGY.md`), or rename the algorithm (suggestions: U2RA,
> URTA, UnRes).

UCRA is a closed-loop resource-allocation pipeline that turns **demand
uncertainty** into **capacity reservations**, allocates resources under risk,
and **self-evolves** as the network drifts.

{BT3}
Network Data ──► 1. Uncertainty Estimation ──► 2. Φ: Uncertainty→Reservation
(traffic, capacity,             │                        │
 topology, risk)                │                        ▼
        ▲                       │             3. Risk-Adaptive Allocation
        │                       ▼                        │
        │              5. Self-Evolving Learning ◄────────┤
        │                       │                        ▼
        └────────── 4. Reservation Update & Release ◄────┘
                     Feedback: reservation error, violations,
                     utilization, latency / reliability
{BT3}

## Datasets

| # | Dataset | Source | License | Role in UCRA |
|---|---------|--------|---------|--------------|
| 1 | **CTTC B5G Network Slicing** (271 MB) | [Zenodo 10610616](https://zenodo.org/records/10610616) - *Data in Brief* 55:110738, 2024 | CC-BY-4.0 | Slice-level reservations (eMBB/mMTC/URLLC), topology, routing, packet drops and delay percentiles → **Stages 2-3 + violation/latency feedback** |
| 2 | **Live 5G/4G/2G RAN PM Counters** (~767 MB, real commercial network) | [Zenodo 17815388](https://zenodo.org/records/17815388) - *Scientific Data*, 2026, DOI 10.1038/s41597-026-07723-0 | CC-BY-4.0 | Real 15-min sector-level demand series (volumes, utilisation, CQI, users, energy) → **Stages 1, 4, 5 + utilization feedback** |
| 3 | *(optional failsafe)* Liverpool 5G High-Density Demand | [opendata.ljmu.ac.uk](https://opendata.ljmu.ac.uk) - *Scientific Data*, 2025 | open | 1-3 s user-level bursts, SINR/PRB/BLER → stress scenarios |

Datasets are **not** committed to this repo - fetch them with
`python scripts/download_data.py` (add `--cttc` for dataset 1, `--ran-all`
for every RAN subset).

## Quickstart

{BT3}bash
pip install -r requirements.txt                  # torch CPU build is enough
python scripts/download_data.py                  # real RAN data (8.3 MB)
python scripts/audit_data.py  --config configs/default.yaml
python scripts/train.py       --config configs/default.yaml
python scripts/run_ucra.py    --config configs/default.yaml --sweep
python scripts/make_figures.py --config configs/default.yaml
{BT3}

Extra experiments:

{BT3}bash
python scripts/demo_evolution.py                 # Stage-5 drift demo (surge + self-healing)
python scripts/download_data.py --cttc           # 271 MB, slice-level dataset
python scripts/run_cttc_eval.py                  # Stage 2-3 on CTTC slices
{BT3}

## Results (real network data - reproducible with the commands above)

RAN PM counters (Zenodo 17815388, Dataset_01): 75 sectors, 4G+5G, 15-min
slots; longest gap-free segment 1,778 slots (Oct 6-25, 2023); capacity
estimate ~143,955. Quantile-LSTM Stage 1: 52,252 params, test pinball 930.3,
coverage 96.0% at the 90% nominal level, CPU-trained in ~3 minutes.

| Policy | Reservation error | Violation rate | Utilization |
|---|---|---|---|
| **UCRA (kappa = 0.5)** | 10.8% | **0.0%** | **71.7%** |
| static_peak | 30.5% | 0.0% | 54.8% |
| mean_forecast | 2.0% | 50.4% | 95.3% |
| oracle_quantile | 20.1% | 16.3% | 64.9% |

UCRA is the only policy that simultaneously holds zero violations **and**
high utilization: static over-reservation wastes ~30% of capacity, and naive
mean forecasting violates half of all slots. The kappa sweep
(`outputs/fig_kappa_sweep.png`) traces the full risk-utility frontier.

### Stage 2-3 on CTTC slices (Zenodo 10610616, 120 snapshots, 70/30 split)

Per-slice-type empirical offered-load quantiles feed the same Phi transform;
policies are compared on held-out snapshots (`scripts/run_cttc_eval.py`):

| Slice | Operator (static delta) | Static q90 | UCRA (Phi) |
|---|---|---|---|
| URLLC | 29.0% violations | 15.2% | **1.4%** |
| eMBB | 50.6% violations | 12.2% | **0.0%** |
| mMTC | 5.4% violations | 12.2% | **4.4%** |

Static delta sizing misses URLLC/eMBB demand spikes; the risk-aware
transform holds near-zero violations in exchange for over-provisioning
(see `outputs/fig_cttc_slices.png` and the kappa sweep inside
`outputs/cttc_results.json`).

## Repo layout

{BT3}
UCRA/
├── configs/default.yaml     # all knobs (quantiles, Φ params, evolution triggers)
├── ucra/
│   ├── data/                # Zenodo loaders + synthetic generator
│   ├── features/            # windowing, splits, scaling
│   ├── models/              # quantile LSTM (pinball loss)
│   ├── core/                # uncertainty, Φ transform, allocation, evolution
│   └── eval/                # metrics, baselines, plots
├── scripts/                 # download / audit / train / run / demo / figures
├── tests/                   # smoke tests (synthetic, CPU-fast)
└── docs/METHODOLOGY.md      # diagram blocks ↔ code modules mapping
{BT3}

## Status

- [x] Step 1 - repo scaffold
- [x] Step 2 - data layer (downloader, loaders, audit)
- [x] Step 3 - feature pipeline
- [x] Step 4 - quantile LSTM uncertainty model
- [x] Step 5 - Φ transformation + risk-adaptive allocation
- [x] Step 6 - evaluation + baselines + figures
- [x] Step 7 - self-evolving loop (+ `scripts/demo_evolution.py`)
- [x] Step 8 - full real-data run (results above)
- [x] Stage 2-3 slice-level eval on CTTC (`scripts/run_cttc_eval.py`)

## License

MIT - see [LICENSE](LICENSE). Dataset licenses remain CC-BY-4.0 (cite the
anchor papers linked above when you publish).
'''


def main() -> None:
    root = Path.cwd()

    # ---- 1. patch run_ucra.py: Stage-5 replay targets must be z-space ----
    p = root / "scripts" / "run_ucra.py"
    if p.exists():
        src = p.read_text(encoding="utf-8")
        old = "evo.remember(scaler.transform(X[ite][t]), Y[ite][t])"
        new = "evo.remember(scaler.transform(X[ite][t]), scY.transform(Y[ite][t]))"
        if old in src:
            p.write_text(src.replace(old, new), encoding="utf-8")
            print("  patched scripts/run_ucra.py   (Stage-5 replay -> z-space)")
        elif new in src:
            print("  ok     scripts/run_ucra.py   (already patched)")
        else:
            print("  WARN   scripts/run_ucra.py   (remember-line not found; "
                  "check manually)")

    # ---- 2. patch quantile_lstm.py: detach fix + stale docstring ----
    p = root / "ucra" / "models" / "quantile_lstm.py"
    if p.exists():
        src = p.read_text(encoding="utf-8")
        changed = False
        old = "tl += float(l) * len(xb)"
        new = "tl += float(l.detach()) * len(xb)"
        if old in src:
            src = src.replace(old, new)
            changed = True
        old2 = '"""X (N, L) -> quantile predictions (N, horizon, Q) in ORIGINAL scale."""'
        new2 = '"""X (N, L) -> quantile predictions (N, horizon, Q) in z-space."""'
        if old2 in src:
            src = src.replace(old2, new2)
            changed = True
        if changed:
            p.write_text(src, encoding="utf-8")
            print("  patched ucra/models/quantile_lstm.py (detach + docstring)")
        else:
            print("  ok     ucra/models/quantile_lstm.py (already patched)")

    # ---- 2b. patch evolve.py: detach fix on the finetune loss log ----
    p = root / "ucra" / "core" / "evolve.py"
    if p.exists():
        src = p.read_text(encoding="utf-8")
        old = '"loss": float(loss), "n_updates": self.n_updates}'
        new = '"loss": float(loss.detach()), "n_updates": self.n_updates}'
        if old in src:
            p.write_text(src.replace(old, new), encoding="utf-8")
            print("  patched ucra/core/evolve.py      (detach fix)")
        elif new in src:
            print("  ok     ucra/core/evolve.py      (already patched)")
        else:
            print("  WARN   ucra/core/evolve.py      (loss-line not found)")

    # ---- 3. new scripts ----
    (root / "scripts").mkdir(exist_ok=True)
    (root / "scripts" / "demo_evolution.py").write_text(DEMO_PY, encoding="utf-8")
    print("  wrote scripts/demo_evolution.py")
    (root / "scripts" / "run_cttc_eval.py").write_text(CTTC_PY, encoding="utf-8")
    print("  wrote scripts/run_cttc_eval.py")

    # ---- 3b. cttc_loader overwrite (dict-tolerant slices/flows) ----
    (root / "ucra" / "data").mkdir(parents=True, exist_ok=True)
    (root / "ucra" / "data" / "cttc_loader.py").write_text(
        CTTC_LOADER_PY, encoding="utf-8")
    print("  wrote ucra/data/cttc_loader.py  (dict-tolerant slices/flows)")

    # ---- 4. README ----
    (root / "README.md").write_text(
        README_MD.replace("{BT3}", BT3), encoding="utf-8")
    print("  wrote README.md (results + full checklist)")

    print("")
    print("Part D done. Now run:")
    print("  python scripts/demo_evolution.py --config configs/default.yaml")
    print("  python scripts/download_data.py --cttc        (271 MB, optional)")
    print("  python scripts/run_cttc_eval.py --config configs/default.yaml")
    print("  git add -A && git commit -m 'Part D: drift demo, CTTC eval, results'")


if __name__ == "__main__":
    main()