#!/usr/bin/env python3
"""Run the full UCRA closed loop on the TEST window (Steps 2-6 in one pass):

  Stage 1  quantile LSTM -> u_hat, U_t, rho_t
  Stage 2  Phi           -> R_t (then Stage 4 EWMA/hysteresis update)
  Stage 3  risk-adaptive allocation across sectors (or single pool)
  Stage 5  self-evolving fine-tuning when drift/violation triggers fire
  Feedback metrics + baselines + figures

CAUSALITY (verified): Stage-5 fine-tuning only ever sees labels that were
observable at update time. Windows reach the replay buffer through
DelayedReplay, which releases a window's H-step target no earlier than
t+H-1 - the first instant all of its labels exist. After an update the
forecast table is refreshed ONLY for slots not yet served (t' > t); slots
<= t keep the forecast that was actually used, and all baselines are
computed from the frozen pre-evolution forecast.

Usage:  python scripts/run_ucra.py --config configs/default.yaml [--sweep]
                              [--seed 42] [--tag NAME]
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
from ucra.core.evolve import DelayedReplay, DriftMonitor, EvolutionEngine
from ucra.core.transform import phi_transform, update_reservation
from ucra.core.uncertainty import UncertaintyState, extract_uncertainty
from ucra.data import load_dataset
from ucra.eval.baselines import (mean_forecast, oracle_quantile,
                                 static_peak, train_quantile_phi)
from ucra.eval.metrics import summarize
from ucra.eval.plots import (plot_demand_vs_reservation, plot_evolution,
                             plot_kappa_sweep, plot_quantile_band)
from ucra.features.windowing import Scaler, chronological_split, make_windows
from ucra.models.quantile_lstm import (QuantileLSTM, make_loss,
                                       predict_quantiles, train_model)


def run_closed_loop(cfg: dict, make_figures: bool = True, verbose: bool = True):
    """Full UCRA loop. Returns (results_dict, log_dataframe). Writes outputs."""
    mc, phi_cfg = cfg["model"], cfg["phi"]
    out = Path(cfg["paths"]["outputs"])
    out.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    # ---------- data + windows ----------
    series, extras = load_dataset(cfg, verbose=verbose)
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
        if verbose:
            print("[ucra] loaded trained model from {}".format(ckpt))
    else:
        if verbose:
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

    # (N, horizon, Q) back in ORIGINAL demand scale.
    # pred_q_frozen : the Stage-1 model as trained - NEVER evolves. Used for
    #                 baselines and the kappa sweep (so they are unaffected
    #                 by any Stage-5 update).
    # pred_q_used   : the forecast actually served at each slot. Identical to
    #                 the frozen table until a Stage-5 update fires; after an
    #                 update at t only slots > t are refreshed (causal).
    pred_q_frozen = scY.inverse(predict_quantiles(model, scaler.transform(X[ite])))
    pred_q_used = pred_q_frozen.copy()
    D_test = Y[ite][:, 0]                                        # realized next-slot demand
    ts_test = series.index[ite[0] + mc["seq_len"]: ite[0] + mc["seq_len"] + len(ite)]

    # ---------- closed loop over the test window ----------
    state = UncertaintyState(cfg["update"]["drift_window"])
    state.set_reference(series.values[itr[0]:itr[-1] + mc["seq_len"]])
    drift = DriftMonitor(cfg["evolution"])
    drift.set_reference(series.values[itr[0]:itr[-1] + mc["seq_len"]])
    evo = EvolutionEngine(model, cfg["evolution"], mc)
    causal = DelayedReplay(evo.buffer, mc["horizon"])   # <-- causal intake
    loss_fn = make_loss(mc["quantiles"])

    sector_frame = extras.get("sector_frame")
    R = np.empty(len(ite)); viol = np.zeros(len(ite))
    log_rows = []; update_points = []; r_prev = float(np.quantile(
        series.values[itr], 0.9))  # start from train q90
    H = mc["horizon"]

    for t in range(len(ite)):
        unc = extract_uncertainty(pred_q_used[t], mc["quantiles"], state)
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
        state.observe(bool(v), D_test[t])          # feedback: slot t is now history
        stats = drift.observe(bool(v), D_test[t])

        # release every replay offer whose labels are fully observed by t
        causal.flush(t)

        # Stage 5 trigger check (only on data available up to t)
        if t % cfg["evolution"]["eval_every"] == 0 and t > 0 and drift.triggered(stats):
            if evo.buffer.X:
                res = evo.finetune(loss_fn)
                if res.get("updated"):
                    update_points.append(t)
                    # causal refresh: only slots not yet served (t+1..) may
                    # see the evolved weights
                    refreshed = scY.inverse(
                        predict_quantiles(model, scaler.transform(X[ite])))
                    pred_q_used[t + 1:] = refreshed[t + 1:]

        # offer this window for future replay; it becomes trainable at t+H-1
        causal.offer(t, scaler.transform(X[ite][t]), scY.transform(Y[ite][t]))

        log_rows.append({"t": t, "ts": ts_test[t], "demand": D_test[t],
                         "R": r_t, "R_target": r_target, "u_hat": unc["u_hat"],
                         "spread": unc["spread"], "rho_t": unc["rho_t"],
                         "violated": int(v), "viol_rate": stats["viol_rate"],
                         "z": stats["z"]})
        r_prev = r_t

    # ---------- metrics + baselines (all vs the SAME D_test, capacity) ----------
    res_ucra = summarize(R, D_test, capacity)
    R_static = static_peak(series.values[itr], len(ite))
    R_mean = mean_forecast(pred_q_frozen, mc["quantiles"])
    R_oracle = oracle_quantile(D_test, mc["horizon"], phi_cfg["base_tau"])
    R_trainq = train_quantile_phi(series.values[itr], len(ite), phi_cfg)
    res = {
        "ucra": res_ucra,
        "static_peak": summarize(R_static, D_test, capacity),
        "mean_forecast": summarize(R_mean, D_test, capacity),
        "train_quantile_phi": summarize(R_trainq, D_test, capacity),
        "oracle_quantile": summarize(R_oracle, D_test, capacity),
        "capacity": capacity,
        "evolution_updates": len(update_points),
        "update_points": update_points,
        "causal_replay": True,
        "label_delay_slots": H - 1,
        "replay_offered": causal.n_offered,
        "replay_released": causal.n_released,
        "n_test": len(ite),
    }
    (out / "ucra_results.json").write_text(json.dumps(res, indent=2))
    pd.DataFrame(log_rows).to_csv(out / "ucra_log.csv", index=False)
    if verbose:
        print(json.dumps(res, indent=2))

    # ---------- figures ----------
    if make_figures:
        plot_demand_vs_reservation(ts_test, D_test, R, R_static,
                                   out / "fig_demand_vs_reservation.png")
        plot_quantile_band(ts_test, D_test, pred_q_used, mc["quantiles"],
                           out / "fig_quantile_band.png")
        roll = pd.Series(viol).rolling(cfg["update"]["drift_window"],
                                       min_periods=8).mean()
        plot_evolution(ts_test, roll.values, update_points,
                       out / "fig_self_evolution.png")
    # ---------- kappa sweep (from the FROZEN forecast; independent of figures) ----------
    if cfg.get("_do_sweep", False):
        ks, vs, us = [], [], []
        for k in cfg["eval"]["sweep_kappa"]:
            pcfg = dict(phi_cfg, kappa=k)
            Rs = np.array([
                phi_transform(
                    extract_uncertainty(pred_q_frozen[t], mc["quantiles"],
                                        state)["u_hat"],
                    extract_uncertainty(pred_q_frozen[t], mc["quantiles"],
                                        state)["spread"],
                    0.0, capacity, pcfg)
                for t in range(len(ite))])
            s = summarize(Rs, D_test, capacity)
            ks.append(k); vs.append(s["violation_rate"]); us.append(s["utilization"])
        sweep_json = {"kappas": ks, "violation_rate": vs,
                      "utilization": us, "source": "frozen_pred_q",
                      "rho_weight": 0.0}
        (out / "kappa_sweep.json").write_text(json.dumps(sweep_json, indent=2))
        if make_figures:
            plot_kappa_sweep(ks, vs, us, out / "fig_kappa_sweep.png")

    return res, pd.DataFrame(log_rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--sweep", action="store_true",
                    help="also run the kappa risk-utility sweep")
    ap.add_argument("--seed", type=int, default=None,
                    help="override cfg seed (defaults to config value)")
    ap.add_argument("--tag", default=None,
                    help="suffix for the output directory (e.g. seed43)")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.seed is not None:
        cfg["seed"] = args.seed
    if args.tag:
        cfg["paths"]["outputs"] = str(Path(cfg["paths"]["outputs"]) / args.tag)
    cfg["_do_sweep"] = args.sweep
    run_closed_loop(cfg)
    print(f"[ucra] results in {cfg['paths']['outputs']}/ (json, csv, PNGs)")


if __name__ == "__main__":
    main()
