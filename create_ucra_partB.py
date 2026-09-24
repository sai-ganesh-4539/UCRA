#!/usr/bin/env python3
# ============================================================
# UCRA — Part B/2 — scripts and tests (paste-safe generator)
# Usage: run INSIDE your existing UCRA folder:  python create_ucra_partB.py
# It writes/overwrites the files listed below, creating dirs.
# ============================================================
from pathlib import Path

FILES = {}

FILES["scripts/download_data.sh"] = r'''#!/usr/bin/env bash
# UCRA dataset downloader (resumable). Usage:
#   bash scripts/download_data.sh          # RAN Dataset_01 (8 MB) + CTTC dataset (271 MB)
#   RAN_ALL=1 bash scripts/download_data.sh  # all three RAN subsets (~767 MB)
#   SKIP_CTTC=1 bash scripts/download_data.sh
set -e
cd "$(dirname "$0")/.."

ZENODO_RAN="https://zenodo.org/records/17815388/files"
ZENODO_CTTC="https://zenodo.org/records/10610616/files"

dl() {  # dl <url> <out>
  echo "[download] $2"
  curl -L -C - --retry 5 --retry-delay 3 -o "$2" "$1"
}

mkdir -p data/ran data/cttc

# ---- Dataset 2: Live RAN PM counters (real commercial network) ----
dl "${ZENODO_RAN}/Dataset_01.zip" data/ran/Dataset_01.zip
if [ "${RAN_ALL:-0}" = "1" ]; then
  dl "${ZENODO_RAN}/Dataset_02.zip" data/ran/Dataset_02.zip
  dl "${ZENODO_RAN}/Dataset_03.zip" data/ran/Dataset_03.zip
fi
dl "${ZENODO_RAN}/README.md" data/ran/README_dataset.md || true

# ---- Dataset 1: CTTC B5G slicing (simulated, slice-level) ----
if [ "${SKIP_CTTC=0}" != "1" ]; then
  dl "${ZENODO_CTTC}/slicing-simulations.zip" data/cttc/slicing-simulations.zip
  # unzip; datanetAPI.py ships inside the archive, if not, fetch it separately
  (cd data/cttc && unzip -n -q slicing-simulations.zip -d slicing-simulations)
  if [ ! -f data/cttc/slicing-simulations/datanetAPI.py ]; then
    dl "${ZENODO_CTTC}/datanetAPI.py" data/cttc/slicing-simulations/datanetAPI.py
  fi
  pip install jsonpickle --quiet || true
fi

echo "[download] done. Files:"
du -sh data/ran data/cttc 2>/dev/null || true
'''

FILES["scripts/audit_data.py"] = r'''#!/usr/bin/env python3
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
'''

FILES["scripts/train.py"] = r'''#!/usr/bin/env python3
"""Train the quantile LSTM (Stage 1 - Uncertainty Estimation).

Usage:  python scripts/train.py --config configs/default.yaml
Saves:  outputs/model.pt, outputs/scaler.npz, outputs/history.png
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
import yaml

from ucra.data import load_dataset
from ucra.features.windowing import Scaler, chronological_split, make_windows
from ucra.eval.plots import plot_training_history
from ucra.models.quantile_lstm import QuantileLSTM, predict_quantiles, train_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    mc = cfg["model"]

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    print(f"[train] loading dataset '{cfg['data']['dataset']}' ...")
    series, extras = load_dataset(cfg)
    print(f"[train] {len(series)} slots | capacity ~{extras['capacity']:.1f}")

    X, Y = make_windows(series.values, mc["seq_len"], mc["horizon"])
    itr, iva, ite = chronological_split(len(X), tuple(mc["train_val_test"]))

    scaler = Scaler().fit(X[itr])
    scY = Scaler().fit(Y[itr])   # targets ALSO z-scored: pinball gradients
                                 # are bounded (~tau), so original-scale targets
                                 # (tens of thousands) make training stall.
    Xtr, Xva, Xte = scaler.transform(X[itr]), scaler.transform(X[iva]), scaler.transform(X[ite])
    Ytr, Yva, Yte = scY.transform(Y[itr]), scY.transform(Y[iva]), scY.transform(Y[ite])

    model = QuantileLSTM(mc["seq_len"], mc["horizon"], mc["quantiles"],
                         mc["hidden_size"], mc["num_layers"], mc["dropout"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model params: {n_params:,}")

    hist = train_model(model, Xtr, Ytr, Xva, Yva, mc)

    out = Path(cfg["paths"]["outputs"])
    out.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "cfg": {"seq_len": mc["seq_len"], "horizon": mc["horizon"],
                        "quantiles": mc["quantiles"], "hidden_size": mc["hidden_size"],
                        "num_layers": mc["num_layers"], "dropout": mc["dropout"]}},
               out / "model.pt")
    np.savez(out / "scaler.npz", mu=scaler.mu, sd=scaler.sd,
             mu_y=scY.mu, sd_y=scY.sd)

    # quick test-set quantile check (back in ORIGINAL scale)
    from ucra.models.quantile_lstm import make_loss
    qt = scY.inverse(predict_quantiles(model, Xte))
    Y_orig = scY.inverse(Yte)  # targets back in original units for metrics
    q05, q50, q99 = (mc["quantiles"].index(0.05), mc["quantiles"].index(0.5),
                     mc["quantiles"].index(0.99))
    cover = float(np.mean((Y_orig[:, 0] >= qt[:, 0, q05]) & (Y_orig[:, 0] <= qt[:, 0, q99])))
    metrics = {"test_pinball": float(make_loss(mc["quantiles"])(
        torch.tensor(qt), torch.tensor(Y_orig))),
        "coverage_90pct": cover, "n_train": len(itr), "n_test": len(ite)}
    (out / "train_metrics.json").write_text(json.dumps(metrics, indent=2))
    plot_training_history(hist, out / "history.png")

    print("[train] done:", json.dumps(metrics, indent=2))
    print(f"[train] saved {out/'model.pt'}, {out/'scaler.npz'}, history.png")


if __name__ == "__main__":
    main()
'''

FILES["scripts/run_ucra.py"] = r'''#!/usr/bin/env python3
"""Run the full UCRA closed loop on the TEST window (Steps 2-6 in one pass):

  Stage 1  quantile LSTM -> u_hat, U_t, rho_t
  Stage 2  Phi           -> R_t (then Stage 4 EWMA/hysteresis update)
  Stage 3  risk-adaptive allocation across sectors (or single pool)
  Stage 5  self-evolving fine-tuning when drift/violation triggers fire
  Feedback metrics + baselines + figures

Usage:  python scripts/run_ucra.py --config configs/default.yaml
Saves:  outputs/ucra_results.json, outputs/ucra_log.csv, PNG figures.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import yaml

from ucra.core.allocate import risk_adaptive_allocate
from ucra.core.evolve import DriftMonitor, EvolutionEngine
from ucra.core.transform import phi_transform, update_reservation
from ucra.core.uncertainty import UncertaintyState, extract_uncertainty
from ucra.data import load_dataset
from ucra.eval.baselines import mean_forecast, oracle_quantile, static_peak
from ucra.eval.metrics import summarize
from ucra.eval.plots import (plot_demand_vs_reservation, plot_evolution,
                             plot_kappa_sweep, plot_quantile_band)
from ucra.features.windowing import Scaler, chronological_split, make_windows
from ucra.models.quantile_lstm import (QuantileLSTM, make_loss,
                                       predict_quantiles, train_model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--sweep", action="store_true",
                    help="also run the kappa risk-utility sweep")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    mc, phi_cfg = cfg["model"], cfg["phi"]
    out = Path(cfg["paths"]["outputs"])
    out.mkdir(exist_ok=True)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # ---------- data + windows ----------
    series, extras = load_dataset(cfg)
    capacity = extras["capacity"]
    X, Y = make_windows(series.values, mc["seq_len"], mc["horizon"])
    itr, iva, ite = chronological_split(len(X), tuple(mc["train_val_test"]))

    # ---------- Stage 1 model (train if no checkpoint yet) ----------
    model = QuantileLSTM(mc["seq_len"], mc["horizon"], mc["quantiles"],
                         mc["hidden_size"], mc["num_layers"], mc["dropout"])
    ckpt = out / "model.pt"
    if ckpt.exists():
        state = torch.load(ckpt, map_location="cpu", weights_only=False)
        model.load_state_dict(state["state_dict"])
        z = np.load(out / "scaler.npz")
        scaler = Scaler(); scaler.mu, scaler.sd = float(z["mu"]), float(z["sd"])
        scY = Scaler(); scY.mu, scY.sd = float(z["mu_y"]), float(z["sd_y"])
        print("[ucra] loaded trained model from outputs/model.pt")
    else:
        print("[ucra] no checkpoint found -> training first "
              "(or run scripts/train.py)")
        scaler = Scaler().fit(X[itr])
        scY = Scaler().fit(Y[itr])
        hist = train_model(model, scaler.transform(X[itr]), scY.transform(Y[itr]),
                           scaler.transform(X[iva]), scY.transform(Y[iva]), mc)
        torch.save({"state_dict": model.state_dict(),
                    "cfg": {"seq_len": mc["seq_len"], "horizon": mc["horizon"],
                            "quantiles": mc["quantiles"],
                            "hidden_size": mc["hidden_size"],
                            "num_layers": mc["num_layers"],
                            "dropout": mc["dropout"]}}, ckpt)
        np.savez(out / "scaler.npz", mu=scaler.mu, sd=scaler.sd,
                 mu_y=scY.mu, sd_y=scY.sd)

    # (N, horizon, Q) back in ORIGINAL demand scale
    pred_q = scY.inverse(predict_quantiles(model, scaler.transform(X[ite])))
    D_test = Y[ite][:, 0]                                        # realized next-slot demand
    ts_test = series.index[ite[0] + mc["seq_len"]: ite[0] + mc["seq_len"] + len(ite)]

    # ---------- closed loop over the test window ----------
    state = UncertaintyState(cfg["update"]["drift_window"])
    state.set_reference(series.values[itr[0]:itr[-1] + mc["seq_len"]])
    drift = DriftMonitor(cfg["evolution"])
    drift.set_reference(series.values[itr[0]:itr[-1] + mc["seq_len"]])
    evo = EvolutionEngine(model, cfg["evolution"], mc)
    loss_fn = make_loss(mc["quantiles"])

    sector_frame = extras.get("sector_frame")
    R = np.empty(len(ite)); viol = np.zeros(len(ite))
    log_rows = []; update_points = []; r_prev = float(np.quantile(
        series.values[itr], 0.9))  # start from train q90

    for t in range(len(ite)):
        unc = extract_uncertainty(pred_q[t], mc["quantiles"], state)
        r_target = phi_transform(unc["u_hat"], unc["spread"], unc["rho_t"],
                                 capacity, phi_cfg)
        r_t = update_reservation(r_prev, r_target, D_test[t - 1] if t else r_target,
                                 capacity, cfg["update"])
        R[t] = r_t

        # Stage 3 across sectors for THIS slot (if sector data available)
        if sector_frame is not None:
            slot = sector_frame[sector_frame["ts"] == ts_test[t]]
            col = "demand" if "demand" in slot.columns else "data_volume"
            dem = slot[col].values if len(slot) else np.array([D_test[t]])
        else:
            dem = np.array([D_test[t]])
        alloc = risk_adaptive_allocate(r_t, dem, risks=None,
                                       alloc_cfg=cfg["allocation"])
        v = float(D_test[t] > r_t)
        viol[t] = v
        state.observe(bool(v), D_test[t])
        stats = drift.observe(bool(v), D_test[t])

        # Stage 5 trigger check
        if t % cfg["evolution"]["eval_every"] == 0 and t > 0 and drift.triggered(stats):
            if evo.buffer.X:
                res = evo.finetune(loss_fn)
                if res.get("updated"):
                    update_points.append(t)
                    # refresh quantile forecasts with evolved theta
                    pred_q = predict_quantiles(model, scaler.transform(X[ite]))
        evo.remember(scaler.transform(X[ite][t]), Y[ite][t])

        log_rows.append({"t": t, "ts": ts_test[t], "demand": D_test[t],
                         "R": r_t, "R_target": r_target, "u_hat": unc["u_hat"],
                         "spread": unc["spread"], "rho_t": unc["rho_t"],
                         "violated": int(v), "viol_rate": stats["viol_rate"],
                         "z": stats["z"]})
        r_prev = r_t

    # ---------- metrics + baselines ----------
    res_ucra = summarize(R, D_test, capacity)
    R_static = static_peak(series.values[itr], len(ite))
    R_mean = mean_forecast(pred_q, mc["quantiles"])
    R_oracle = oracle_quantile(D_test, mc["horizon"], phi_cfg["base_tau"])
    res = {
        "ucra": res_ucra,
        "static_peak": summarize(R_static, D_test, capacity),
        "mean_forecast": summarize(R_mean, D_test, capacity),
        "oracle_quantile": summarize(R_oracle, D_test, capacity),
        "capacity": capacity,
        "evolution_updates": len(update_points),
    }
    (out / "ucra_results.json").write_text(json.dumps(res, indent=2))
    pd.DataFrame(log_rows).to_csv(out / "ucra_log.csv", index=False)
    print(json.dumps(res, indent=2))

    # ---------- figures ----------
    plot_demand_vs_reservation(ts_test, D_test, R, R_static,
                               out / "fig_demand_vs_reservation.png")
    plot_quantile_band(ts_test, D_test, pred_q, mc["quantiles"],
                       out / "fig_quantile_band.png")
    roll = pd.Series(viol).rolling(cfg["update"]["drift_window"], min_periods=8).mean()
    plot_evolution(ts_test, roll.values, update_points,
                   out / "fig_self_evolution.png")
    if args.sweep:
        ks, vs, us = [], [], []
        for k in cfg["eval"]["sweep_kappa"]:
            pcfg = dict(phi_cfg, kappa=k)
            Rs = [phi_transform(extract_uncertainty(pred_q[t], mc["quantiles"],
                                                    state)["u_hat"],
                                extract_uncertainty(pred_q[t], mc["quantiles"],
                                                    state)["spread"],
                                0.0, capacity, pcfg)
                  for t in range(len(ite))]
            Rs = np.array(Rs)
            s = summarize(Rs, D_test, capacity)
            ks.append(k); vs.append(s["violation_rate"]); us.append(s["utilization"])
        plot_kappa_sweep(ks, vs, us, out / "fig_kappa_sweep.png")

    print(f"[ucra] results in {out}/ (json, csv, 3-4 PNGs)")


if __name__ == "__main__":
    main()
'''

FILES["scripts/make_figures.py"] = r'''#!/usr/bin/env python3
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
'''

FILES["tests/__init__.py"] = r'''"""UCRA test suite."""
'''

FILES["tests/test_smoke.py"] = r'''"""UCRA end-to-end smoke test (synthetic data, tiny epochs, CPU-fast).

Run:  python -m pytest tests/test_smoke.py -q   (or python tests/test_smoke.py)
"""
import numpy as np
import torch
import yaml


def _cfg(tmp_path):
    cfg = yaml.safe_load(open("configs/default.yaml"))
    cfg["data"]["dataset"] = "synthetic"
    cfg["data"]["synthetic"] = {"n_steps": 500, "n_sectors": 3}
    mc = cfg["model"]
    mc.update({"seq_len": 48, "horizon": 2, "hidden_size": 8, "num_layers": 1,
               "max_epochs": 2, "patience": 2, "batch_size": 32})
    cfg["paths"]["outputs"] = str(tmp_path / "outputs")
    cfg["update"]["drift_window"] = 48
    cfg["evolution"].update({"eval_every": 20, "finetune_epochs": 1,
                             "replay_size": 32})
    return cfg


def test_full_pipeline(tmp_path):
    from ucra.data import load_dataset
    from ucra.features.windowing import Scaler, chronological_split, make_windows
    from ucra.models.quantile_lstm import (QuantileLSTM, make_loss,
                                           predict_quantiles, train_model)
    from ucra.core.uncertainty import UncertaintyState, extract_uncertainty
    from ucra.core.transform import phi_transform, update_reservation
    from ucra.core.allocate import risk_adaptive_allocate
    from ucra.core.evolve import DriftMonitor, EvolutionEngine
    from ucra.eval.metrics import summarize

    cfg = _cfg(tmp_path)
    torch.manual_seed(0)
    np.random.seed(0)

    series, extras = load_dataset(cfg)
    assert len(series) == 500 and extras["capacity"] > 0

    X, Y = make_windows(series.values, 48, 2)
    itr, iva, ite = chronological_split(len(X))
    scaler = Scaler().fit(X[itr])
    model = QuantileLSTM(48, 2, cfg["model"]["quantiles"], 8, 1, 0.0)
    hist = train_model(model, scaler.transform(X[itr]), Y[itr],
                       scaler.transform(X[iva]), Y[iva], cfg["model"],
                       verbose=False)
    assert len(hist["val"]) >= 1

    pred_q = predict_quantiles(model, scaler.transform(X[ite]))
    assert pred_q.shape == (len(ite), 2, 7)

    # one closed-loop step
    state = UncertaintyState(96)
    state.set_reference(series.values[itr])
    unc = extract_uncertainty(pred_q[0], cfg["model"]["quantiles"], state)
    assert unc["spread"] >= 0 and 0.0 <= unc["rho_t"] <= 1.5

    R = phi_transform(unc["u_hat"], unc["spread"], 0.2,
                      extras["capacity"], cfg["phi"])
    assert 0 < R <= extras["capacity"] * cfg["phi"]["max_reserve_ratio"]

    R2 = update_reservation(R, R * 0.5, D := float(Y[ite][0, 0]),
                            extras["capacity"], cfg["update"])
    assert R2 >= 0

    alloc = risk_adaptive_allocate(R, np.array([D * 0.4, D * 0.9]),
                                  alloc_cfg=cfg["allocation"])
    assert alloc["alloc"].sum() <= R + 1e-6

    m = summarize(np.full(10, R), Y[ite][:10, 0], extras["capacity"])
    assert set(m) == {"reservation_error", "violation_rate", "utilization", "waste"}

    # evolution engine
    drift = DriftMonitor(cfg["evolution"])
    drift.set_reference(series.values[itr])
    evo = EvolutionEngine(model, cfg["evolution"], cfg["model"])
    for i in range(len(ite)):
        evo.remember(scaler.transform(X[ite][i]), Y[ite][i])
    res = evo.finetune(make_loss(cfg["model"]["quantiles"]))
    assert res["updated"] is True


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        test_full_pipeline(td)
    print("SMOKE TEST PASSED")
'''

def main() -> None:
    for rel, content in FILES.items():
        target = Path(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        print("  wrote", rel)
    print("\nDone. Now run:")
    print("  python scripts/audit_data.py --config configs/default.yaml")
    print("  python scripts/train.py --config configs/default.yaml")
    print("  python scripts/run_ucra.py --config configs/default.yaml --sweep")

if __name__ == "__main__":
    main()
