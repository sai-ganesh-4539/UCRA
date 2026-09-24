# 10 -- Code Reference: Every Module, Class and Function

Signatures are given as written in the source. Read this together with the
stage docs (03-06) for the "why".

## ucra/features/windowing.py

- `make_windows(series, seq_len, horizon) -> (X, Y)`
  Slides over a 1-D array; X (N, seq_len) float32, Y (N, horizon). Raises
  ValueError if the series is shorter than seq_len + horizon. No stride
  parameter (stride 1 by design: more windows from short segments).
- `chronological_split(n, fractions=(0.7, 0.15, 0.15)) -> (train_idx,
  val_idx, test_idx)`
  Contiguous np.arange ranges; NEVER shuffles.
- `class Scaler` -- `fit(X)` (stores mean/std + 1e-8 guard), `transform`,
  `inverse`. Fitted on TRAIN slice only. Two instances in practice: one for
  inputs, one (scY) for targets; both persisted in scaler.npz as
  mu/sd/mu_y/sd_y.

## ucra/models/quantile_lstm.py

- `class QuantileLSTM(seq_len, horizon, quantiles, hidden_size=64,
  num_layers=2, dropout=0.1)` -- nn.Module. forward: (B, L) ->
  unsqueeze(-1) -> LSTM -> last step -> Linear -> (B, horizon, Q). seq_len
  is stored but unused at runtime (LSTM infers length); it documents intent.
- `make_loss(quantiles) -> loss_fn(y_pred (B,H,Q), y_true (B,H))`
  Pinball via `torch.maximum(taus * e, (taus - 1.0) * e)`, mean over
  everything. taus shaped (1, 1, Q) for broadcasting.
- `train_model(model, Xtr, Ytr, Xva, Yva, cfg, verbose=True) -> hist`
  Adam(lr=cfg lr), batch 64, up to max_epochs, early stop patience 6 on val
  pinball, restores best-val state at the end. NOTE the `float(l.detach())`
  in the running-train-loss accumulation (Part D fix for the PyTorch
  UserWarning about graph retention).
- `predict_quantiles(model, X, batch_size=256) -> (N, horizon, Q)`
  In WHATEVER space X is in (z-space in the pipeline); batched, no-grad,
  model.eval().

## ucra/core/uncertainty.py

- `class UncertaintyState(drift_window=96)` -- ring buffers of violation
  flags and demand; `set_reference(train_demand)` stores mu/sd;
  `observe(violated, demand)`; `rho()` per doc 03 section 10
  (needs >= 8 recent demand points for the z term; capped at 1.5).
- `extract_uncertainty(pred_q, quantiles, state) -> dict`
  pred_q here is ONE slot's (horizon, Q) matrix; uses step 0 only. The
  quantile-index dict `{round(q, 2): i}` is the reason quantile values in
  config must stay 2-decimal-friendly (0.5, 0.99 -- fine; 0.333 would
  break the lookups).

## ucra/core/transform.py

- `phi_transform(u_hat, spread, rho_t, capacity, phi_cfg) -> float`
  Implements doc 04 section 1 verbatim; np.clip between the 1.05*u_hat
  floor and 0.95*C ceiling.
- `update_reservation(r_prev, r_target, realized, capacity, upd_cfg) ->
  float`
  EWMA alpha 0.3; release branch only when headroom_ratio > 0.10 AND
  r_target < r_prev; never releases below `realized`; floors at 0.

## ucra/core/allocate.py

- `risk_adaptive_allocate(r_t, demands, risks=None, alloc_cfg=None) ->
  {"alloc", "unmet", "best_effort"}`
  Doc 05 verbatim. Edge cases: empty demands or r_t <= 0 -> everything
  unmet; all-zero weights -> uniform shares; leftover redistributed among
  the still-hungry proportionally to their shares (capped by their
  remaining demand). `risks=None` means all boosts 1.0.

## ucra/core/evolve.py

- `class ReplayBuffer(capacity=512)` -- reservoir sampling (Vitter R);
  `arrays()` stacks or returns (None, None).
- `class DriftMonitor(cfg)` -- reads violation_trigger, drift_zscore,
  drift_window from the evolution config section; `observe()` returns
  {"viol_rate", "z"}; `triggered()` is OR over the two thresholds.
- `class EvolutionEngine(model, cfg, train_cfg)` -- cfg = evolution
  section, train_cfg = model section (only `lr` is used, x 0.3);
  `remember(x, y)` pushes to the buffer; `finetune(loss_fn)` runs
  finetune_epochs full-batch steps and returns a result dict; counts
  n_updates.

## ucra/data/

- `__init__.py: load_dataset(cfg, verbose=True) -> (series, extras)`
  Dispatch on cfg["data"]["dataset"]:
  - "synthetic": generated frame + sector_frame; capacity = 250 * n_sectors.
  - "ran": ran_loader + load_demand_series; capacity = 1.3 * series.max().
    extras["sector_frame"] = canonical long frame (used by Stage 3).
  - "cttc": load_cttc -> per-sample summed offered load as the series
    (RangeIndex); capacity = max link_cap. extras["slice_frame"].
- `ran_loader.py`: `load_ran(data_dir, subset, verbose)` (auto-extract,
  tech-aware column mapping, canonical frame) and `load_demand_series(df,
  agg, freq, keep="longest")` (gap splitting + interpolation limit 4).
- `cttc_loader.py`: `load_cttc(samples_dir, max_samples, verbose)` --
  dict-tolerant slice/flow reading (doc 07 section 3), corrupt-sample skip,
  per-25 progress prints.
- `synthetic.py`: `generate_synthetic(n_steps, n_sectors, seed, base)` and
  `synthetic_capacity(n_sectors, per_sector=250)`.

## ucra/eval/

- `metrics.py`: `reservation_error(R, D, C)`, `violation_rate(R, D)`,
  `utilization(R, D)`, `waste(R, D, C)`, and `summarize(R, D, C)` returning
  all four in one dict. All pure numpy, all guarded with max(..., 1e-9).
- `baselines.py`: `static_peak(train_demand, n_test)` (constant array),
  `mean_forecast(pred_q, quantiles)` (step-0 median),
  `oracle_quantile(test_demand, horizon, tau=0.9)` (rolling window
  w = max(48, horizon*4) = 48; slot 0 seeded with test_demand[0]).
- `plots.py`: `plot_demand_vs_reservation`, `plot_quantile_band` (first n
  = 240 test slots fan chart), `plot_kappa_sweep` (dual-axis),
  `plot_training_history`, `plot_evolution`. All matplotlib Agg, 150 dpi,
  constrained_layout, legends anchored outside via bbox_to_anchor.

## scripts/ (entry points)

| Script | One-liner | Key flags |
|---|---|---|
| download_data.py | stdlib Zenodo fetcher/verifier/unzipper | --cttc, --ran-all, --list, --root |
| audit_data.py | dataset sanity stats + overview figure | --config |
| train.py | Stage 1 training + calibration check | --config |
| run_ucra.py | full closed loop + baselines (+ sweep) | --config, --sweep |
| demo_evolution.py | Stage 5 A/B surge demo | --shift, --at, --eval-every, --ft-epochs, --lr-scale |
| run_cttc_eval.py | Stage 2-3 policies on CTTC slices | --config, --max-samples, --train-frac |
| make_figures.py | refresh figures from saved CSV/JSON | --config |

All scripts insert the repo root into sys.path (`Path(__file__)...parents[1]`)
so they run from anywhere, but outputs land relative to `paths.outputs` in
the config -- run from the repo root to keep everything predictable.

## tests/test_smoke.py

End-to-end on synthetic data with a tiny model (seq 48, horizon 2, hidden
8, 1 layer, 2 epochs). Asserts: series length and capacity; window shapes;
pred_q shape (N, 2, 7); spread >= 0 and rho in [0, 1.5]; phi output in
(0, 0.95C]; update_reservation >= 0; allocation sum <= R; the four metric
keys; and a full evolution cycle (remember x N then finetune ->
updated=True). Run: `python -m pytest tests/test_smoke.py -q` or directly
`python tests/test_smoke.py`.
