#!/usr/bin/env python3
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
