# DEEP_CODE.md — code reference, config, runbook, failure modes

The "where is everything and what happens when I run it" file. Every module
with its key functions, every config knob with its effect, the run order
with expected output, and the failure modes we actually hit and fixed.

---

## 1. Repo map (annotated)

```
E:\UCRA\
  README.md                  results tables + quickstart
  DEEP_PIPELINE.md           stages 1-5 math + code      (root docs, this set)
  DEEP_DATA.md               datasets + loaders + leakage
  DEEP_RESULTS.md            every number decoded
  DEEP_CODE.md               this file
  configs/default.yaml       EVERY tunable in one file
  ucra/
    data/      ran_loader.py      Zenodo 17815388 -> long frame -> demand series
               cttc_loader.py     Zenodo 10610616 -> slice-level frame
               synthetic.py       config-driven fake data (tests/smoke)
               __init__.py        load_dataset(cfg) dispatch -> (series, extras)
    features/  windowing.py       make_windows, chronological_split, Scaler
    models/    quantile_lstm.py   QuantileLSTM, make_loss, train_model,
                                  predict_quantiles            (Stage 1)
    core/      uncertainty.py     UncertaintyState, extract_uncertainty
               transform.py       phi_transform (Stage 2), update_reservation (Stage 4)
               allocate.py        risk_adaptive_allocate       (Stage 3)
               evolve.py          DriftMonitor, ReplayBuffer,
                                  EvolutionEngine              (Stage 5)
    eval/      metrics.py         reservation_error, violation_rate,
                                  utilization, waste, summarize
               baselines.py       static_peak, mean_forecast, oracle_quantile
               plots.py           the 6 figure builders
  scripts/
    download_data.py   stdlib downloader: resume + retries + md5 manifest
    audit_data.py      series sanity + histogram -> outputs/audit_demand.png
    train.py           Stage-1 training -> model.pt, scaler.npz, metrics
    run_ucra.py        full closed loop + baselines + figures (--sweep)
    demo_evolution.py  Stage-5 drift demo (frozen vs evolving)
    run_cttc_eval.py   Stage 2-3 on CTTC (operator vs static vs phi)
    make_figures.py    refresh figures from saved log + results
  tests/test_smoke.py    end-to-end pipeline test on synthetic data
  outputs/               committed results (13 artifacts + README manifest)
  docs/                  16-guide documentation set
```

---

## 2. Module API reference

### ucra/data/__init__.py
```python
load_dataset(cfg, verbose=True) -> (pd.Series, extras: dict)
# series: float demand per 15-min slot (ran/cttc/synthetic dispatch)
# extras: {"capacity": float, "sector_frame": df}   (ran)
#         {"capacity": float, "slice_frame": df}    (cttc)
```

### ucra/features/windowing.py
```python
make_windows(series, seq_len, horizon) -> (X (N, L), Y (N, H)) float32
chronological_split(n, (0.7, 0.15, 0.15)) -> (itr, iva, ite)  # no shuffle
Scaler().fit(X) / .transform(X) / .inverse(X)                 # z-score, eps 1e-8
```

### ucra/models/quantile_lstm.py
```python
QuantileLSTM(seq_len, horizon, quantiles, hidden_size=64,
             num_layers=2, dropout=0.1)      # forward: (B, L) -> (B, H, Q)
make_loss(quantiles) -> loss_fn              # pinball, broadcast (B, H, Q)
train_model(model, Xtr, Ytr, Xva, Yva, cfg)  # Adam, early stop -> history dict
predict_quantiles(model, X, batch_size=256)  # (N, horizon, Q), no_grad
```

### ucra/core/uncertainty.py
```python
UncertaintyState(drift_window=96)            # .set_reference(train), .observe(v, D)
extract_uncertainty(pred_q[t], quantiles, state) -> dict
#   {"u_hat": q50, "spread": q99 - q50, "rho_t": in [0, 1.5], "q_vec": [...]}
```

### ucra/core/transform.py
```python
phi_transform(u_hat, spread, rho_t, capacity, phi_cfg) -> float
#   clip(q90 + kappa*spread*(1 + 0.3*rho), 1.05*q50, 0.95*C)
update_reservation(r_prev, r_target, realized, capacity, upd_cfg) -> float
#   EWMA alpha=0.3; release only under >10% headroom hysteresis
```

### ucra/core/allocate.py
```python
risk_adaptive_allocate(r_t, demands, risks=None, alloc_cfg=None) -> dict
#   {"alloc": (n,), "unmet": (n,), "best_effort": float}
#   10% best-effort pool; w = demand^1 * (1+risk)^0.5; leftover redistribution
```

### ucra/core/evolve.py
```python
DriftMonitor(cfg)                # .set_reference, .observe -> {"viol_rate", "z"},
                                 # .triggered: viol_rate>0.15 or z>3.0
ReplayBuffer(capacity=512)       # reservoir sampling; .push, .arrays
EvolutionEngine(model, cfg, train_cfg)
#   .remember(x, y)  (z-space!)  .finetune(loss_fn): Adam lr*0.3, 3 epochs
```

### ucra/eval/*
```python
summarize(R, D, capacity) -> {"reservation_error", "violation_rate",
                              "utilization", "waste"}
static_peak(train_demand, n_test) -> (n_test,)     # constant train max
mean_forecast(pred_q, quantiles) -> (N,)           # per-slot q50
oracle_quantile(test_demand, horizon, tau) -> (N,) # CHEATING rolling q90, w=48
```

---

## 3. configs/default.yaml — every knob and its effect

| section.key | default | effect if changed |
|---|---|---|
| seed | 42 | all torch/np randomness |
| data.dataset | ran | `ran` / `cttc` / `synthetic` — one switch swaps the whole pipeline input |
| data.ran.subset | dataset01 | which Zenodo archive to use (01 = 8 MB, 02 = 49 MB, 03 = 709 MB) |
| data.ran.target / agg | data_volume / sum | which counter becomes demand; sector aggregation |
| data.cttc.samples_dir | data/cttc/... | where the unzipped CTTC dataset lives |
| data.synthetic.n_steps / n_sectors | 2016 / 8 | smoke-test data size |
| model.seq_len / horizon | 96 / 4 | lookback (24 h) and predict-ahead (1 h) in 15-min slots |
| model.hidden_size / num_layers / dropout | 64 / 2 / 0.1 | capacity of the LSTM (52,252 params at defaults) |
| model.quantiles | [0.05...0.99] | the 7 heads; spread uses 0.5 and 0.99 |
| model.batch_size / lr | 64 / 0.001 | optimization |
| model.max_epochs / patience | 50 / 6 | training length + early stop |
| model.train_val_test | [.7 .15 .15] | chronological split fractions |
| phi.base_tau | 0.9 | the quantile the reservation follows (q_tau) |
| phi.kappa | 0.5 | THE risk knob: buffer = kappa x spread |
| phi.rho_weight | 0.3 | how strongly feedback (rho_t) inflates the buffer |
| phi.min_headroom | 0.05 | floor: never reserve < 5% above q50 |
| phi.max_reserve_ratio | 0.95 | ceiling: never reserve > 95% of capacity |
| allocation.pressure_exponent | 1.0 | how aggressively demand-heavy entities get more |
| allocation.risk_exponent | 0.5 | sqrt damping of risk weighting |
| allocation.best_effort_share | 0.1 | share of R kept out of promises |
| update.ewma_alpha | 0.3 | reservation smoothing (higher = snappier) |
| update.release_hysteresis | 0.1 | release only when >10% of capacity unused |
| update.drift_window | 96 | rolling window for rho + drift stats |
| evolution.violation_trigger | 0.15 | Stage-5 fires above this rolling violation rate |
| evolution.drift_zscore | 3.0 | ...or above this demand z-score |
| evolution.replay_size | 512 | continual-learning buffer size |
| evolution.finetune_epochs | 3 | epochs per Stage-5 update |
| evolution.eval_every | 96 | trigger check cadence (demo uses 24) |
| eval.sweep_kappa | [0 ... 2.0] | frontier grid for --sweep |

---

## 4. Runbook (exact order + what you should see)

```
pip install -r requirements.txt          # torch CPU, pandas, yaml, matplotlib
python scripts/download_data.py          # RAN; add --cttc for dataset B
python scripts/audit_data.py
#   -> "length: 1778 slots", "capacity est.: 143954.5", missing slots: 0
python scripts/train.py
#   -> "model params: 52,252", epoch lines, early stop, then
#      test_pinball / coverage_90pct (~0.93-0.96) / n_train 1175 / n_test 252
python scripts/run_ucra.py --sweep
#   -> "[ucra] loaded trained model from outputs/model.pt"
#      results JSON: ucra 0.0 viol / 0.72 util; static 0.30 waste;
#      mean_forecast ~0.50 viol; oracle 0.16 viol
python scripts/demo_evolution.py
#   -> frozen vs evolving JSON; updates [96, 120, 144]
python scripts/run_cttc_eval.py          # needs --cttc download first
#   -> per-slice table: operator 29/50.6/5.4 vs ucra_phi 1.4/0.0/4.4
python scripts/make_figures.py           # optional: refresh figs from log
python -m pytest tests/test_smoke.py -q  # green = pipeline intact
```

Total wall time after download: ~5 minutes on a plain laptop CPU.

---

## 5. Failure modes we actually hit (and their fixes)

| symptom | cause | fix |
|---|---|---|
| `The token '&&' is not a valid statement separator` | Windows PowerShell 5.1 has no `&&` | run commands one per line, or `python x.py; if ($?) { ... }` |
| training loss stuck constant (~66) | pinball gradients bounded; raw-scale targets too large | already fixed: train in z-space (scY), invert after |
| `'dict' object has no attribute 'type'` in CTTC load | dataset v2.2.1 ships dicts, old API expects objects | already fixed: cttc_loader reads both shapes |
| fine-tune would corrupt model | replay buffer stored raw-scale Y while model is z-space | already fixed: buffer stores scY.transform(Y) |
| `[warn]` UserWarning in train loop | `float(l)` on tensor with grad | already fixed: `float(l.detach())` — cosmetic anyway |
| run_ucra uses OLD results after config change | `outputs/model.pt` exists -> loads checkpoint silently | delete `outputs/model.pt` (+ `scaler.npz`) after changing model/phi knobs |
| `No dataset files under data/ran` | download not run / wrong cwd | `python scripts/download_data.py` from repo root |
| CTTC eval exits "no samples parsed" | 271 MB zip not downloaded/unzipped | `python scripts/download_data.py --cttc` |
| coverage differs from a friend's run | torch CPU nondeterminism across platforms | expected; conclusions unchanged (docs/13) |
| git push DNS failure | transient network | retry; pip working means network is back |

---

## 6. Tests

`tests/test_smoke.py` runs the ENTIRE five-stage loop on tiny synthetic
data (500 steps, 3 sectors, seq_len 48, hidden 8, 2 epochs, ~seconds):
asserts series shape, window shapes (N, 2, 7), spread >= 0, rho in
[0, 1.5], Phi respects the ceiling, EWMA >= 0, allocation never exceeds
R, the four metric keys exist, and the EvolutionEngine updates. If it
passes, every layer of the stack is wired correctly — run it after any
code or config surgery.

---

## 7. Extension recipes (for the paper / future work)

- **New dataset**: add a loader returning a 15-min demand series + extras,
  register it in `load_dataset()`, set `data.dataset: <name>`. Zero
  changes to stages 1-5.
- **New baseline**: one function `(train_demand | pred_q, ...) -> R` in
  `baselines.py`, add `summarize()` call in `run_ucra.py`.
- **Tune the SLA stance**: move `phi.kappa` (or pick from the sweep curve).
- **Faster adaptation**: lower `evolution.eval_every` (demo showed 24
  works) or raise `finetune_epochs`.
- **Per-slice deployment**: run the Phi stage per slice type with
  per-type kappa — the CTTC eval script is the template.
