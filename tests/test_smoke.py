"""UCRA end-to-end smoke test (synthetic data, tiny epochs, CPU-fast).

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
