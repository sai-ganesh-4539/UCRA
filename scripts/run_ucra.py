#!/usr/bin/env python3
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
