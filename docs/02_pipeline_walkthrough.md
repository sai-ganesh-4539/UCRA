# 02 -- Pipeline Walkthrough: Commands, Files, and Data Flow

This doc follows the exact path your data took, with the real numbers from
your machine. Keep `configs/default.yaml` open beside it (doc 11 explains
every line of it).

## 1. Repository map

```
UCRA/
+-- configs/default.yaml        all knobs (doc 11)
+-- ucra/                       the library (importable package)
|   +-- data/                   loaders: ran_loader, cttc_loader, synthetic, dispatch
|   +-- features/windowing.py   sliding windows, chronological split, z-score Scaler
|   +-- models/quantile_lstm.py Stage 1 network + pinball loss + training loop
|   +-- core/uncertainty.py     Stage 1 output assembly (u_hat, U_t, rho_t)
|   +-- core/transform.py       Stage 2 Phi + Stage 4 EWMA/hysteresis
|   +-- core/allocate.py        Stage 3 risk-adaptive split
|   +-- core/evolve.py          Stage 5 drift monitor + replay + fine-tune
|   +-- eval/metrics.py         the 4 metrics
|   +-- eval/baselines.py       static_peak, mean_forecast, oracle_quantile
|   +-- eval/plots.py           all PNG figures
+-- scripts/                    one entry point per experiment (doc 10)
+-- tests/test_smoke.py         end-to-end smoke test on synthetic data
+-- docs/                       these guides
+-- outputs/                    EVERYTHING the pipeline produces (git-ignored)
+-- data/                       downloaded Zenodo archives (git-ignored)
```

## 2. The data flow, with your actual numbers

```
Dataset_01.zip (8.33 MB, Zenodo 17815388)
        |  ran_loader.py: unzip, tech-aware column mapping, 15-min rows
        v
canonical frame  302,052 rows x 6 cols   [ts, sector, data_volume,
        |                                 utilization, active_users, tech]
        |  75 sectors, techs 4G+5G, two campaigns (Mar 2023, Oct 2023)
        |  load_demand_series(): sum sectors per timestamp -> strict 15-min grid
        v
demand series, split at every gap > 15 min -> LONGEST gap-free segment
        1,778 slots  (2023-10-06 21:15 -> 2023-10-25 09:30)
        |
        |  capacity estimate C = 1.3 x max(series) = 1.3 x ~110,734
        |                       = 143,954.5      <-- your printed capacity
        v
make_windows(seq_len=96, horizon=4):  1,779 - 96 - 4 = 1,679 windows
        X (1679, 96), Y (1679, 4)
        |
        |  chronological_split 0.70/0.15/0.15 (NO shuffle)
        v
train 1,175 | val 252 | test 252 windows
        |
        |  Stage 1 training (z-space, early stopping)  ~3 min CPU
        v
model.pt (52,252 params) + scaler.npz (mu, sd, mu_y, sd_y)
        |
        |  predict on test -> pred_q (252, 4, 7) in ORIGINAL units
        v
closed loop over 252 slots (run_ucra.py): Phi -> EWMA -> allocate -> monitor
        |
        v
ucra_results.json + ucra_log.csv + 4 PNGs
```

Useful conversions: 1 slot = 15 min; the 252-slot test window = 63 hours
(approx. Oct 22 18:00 -> Oct 25 08:45, 2023 -- verify with the first/last
`ts` in `outputs/ucra_log.csv`).

## 3. The commands, one by one

### `python scripts/download_data.py`
Pure-stdlib downloader (no wget/curl needed on Windows). Fetches RAN
Dataset_01 (8,333,223 bytes) from Zenodo 17815388, verifies md5, resumes
interrupted downloads with HTTP Range requests, retries 6 times, skips
anything already complete, and extracts zips with a marker so it is
idempotent. Add `--cttc` for the 271 MB CTTC archive (also installs the
`jsonpickle` package the dataset's own `datanetAPI.py` needs). `--list`
prints what it knows about; `--ran-all` fetches Dataset_02/03 too (not
needed for the paper numbers).

### `python scripts/audit_data.py --config configs/default.yaml`
Loads the configured dataset through the same loader the pipeline uses and
prints: length 1,778 slots, span, capacity 143,954.5, demand min/max/mean
(mean ~53,000 on this segment), q50/q90/q99, missing-slot count. Writes
`outputs/audit_demand.png` (series + histogram). **Always run this first** --
if these numbers look wrong, everything downstream is wrong.

### `python scripts/train.py --config configs/default.yaml`
Stage 1 only. Windows the series, fits the two z-score scalers, trains the
quantile LSTM (Adam, batch 64, early stopping patience 6, best-state
restore), saves `outputs/model.pt` + `outputs/scaler.npz` +
`outputs/history.png` + `outputs/train_metrics.json`. Prints parameter
count (52,252) and the test-set quantile check: coverage 96.0% of realized
values inside the [q05, q99] band (nominal for that band is 94% -- see doc
03 section 7 about the metric's name) and test pinball 930.3 in original
units. Re-running **skips nothing**: train.py always retrains. The other
scripts, however, reuse the saved checkpoint.

### `python scripts/run_ucra.py --config configs/default.yaml --sweep`
The full closed loop (doc 01 section 4), plus baselines and the kappa sweep.
Loads `outputs/model.pt` if present (yours did), otherwise trains first.
Per test slot: predict -> Phi -> EWMA -> allocate -> monitor -> (Stage 5
armed). Writes `outputs/ucra_results.json`, `outputs/ucra_log.csv` (one row
per slot: demand, R, R_target, u_hat, spread, rho_t, violated, rolling
rate, z), and figures `fig_demand_vs_reservation.png`,
`fig_quantile_band.png`, `fig_self_evolution.png`, and with `--sweep`
`fig_kappa_sweep.png`. The sweep re-runs Phi for kappa in
[0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0] with rho forced to 0 (kappa-only
effect) and plots the risk-utility frontier.

### `python scripts/demo_evolution.py --config configs/default.yaml`
Stage 5 demonstration (doc 06 has the full decode of your output). Injects
a +35% demand surge 15% into the test window, then runs the loop twice from
the same checkpoint: frozen (Stages 1-4) vs self-evolving (Stage 5 on).
Writes `outputs/fig_drift_demo.png` + `outputs/drift_demo.json`.

### `python scripts/download_data.py --cttc` then
### `python scripts/run_cttc_eval.py --config configs/default.yaml --max-samples 120`
Parses 120 CTTC steady-state snapshots (271 MB dataset, ~2,908 slice rows),
splits snapshots 70/30, and compares three reservation policies per slice
type on held-out snapshots: `operator` (the dataset's own delta sizing),
`static_q90` (train-half q90 of offered load), `ucra_phi` (the Phi
transform on the empirical distribution). Writes
`outputs/cttc_results.json` (includes a per-type kappa sweep and QoS-gap
fields) + `outputs/fig_cttc_slices.png`.

### `python scripts/make_figures.py --config configs/default.yaml`
Refreshes `fig_demand_vs_reservation.png` from the saved CSV/JSON without
re-running the loop.

### `python -m pytest tests/test_smoke.py -q`
End-to-end smoke test on synthetic data (tiny model, 2 epochs, seconds on
CPU). Asserts shapes and invariants: Phi output within (0, 0.95C],
allocation sum <= R, all 4 metrics present, replay fine-tune returns
`updated: True`.

## 4. Console output -> code mapping

| What you saw | Who printed it | Where in code |
|---|---|---|
| `[ran_loader] N CSV file(s)` / `canonical frame: (302052, 6) ...` | ran_loader | `load_ran()` final print |
| `[cttc_loader] processed 25/50/... samples` | cttc_loader | loop progress print |
| `[cttc_loader] frame: (2908, 8), slices per type: ...` | cttc_loader | end of `load_cttc` |
| `[train] model params: 52,252` | train.py | `sum(p.numel() ...)` |
| `epoch 07 train 0.0213 val 0.0189` | quantile_lstm | `train_model` per-epoch print |
| `[ucra] loaded trained model from outputs/model.pt` | run_ucra | checkpoint branch |
| the big JSON with ucra/static_peak/mean_forecast/oracle | run_ucra | `print(json.dumps(res ...))` |
| demo JSON with `frozen` / `evolving` | demo_evolution | after both passes |
| `[demo] saved outputs\fig_drift_demo.png` | demo_evolution | figure save |

## 5. Everything in outputs/ and who writes it

| File | Writer | Contents |
|---|---|---|
| audit_demand.png | audit_data.py | series + histogram |
| history.png | train.py | train/val pinball per epoch |
| model.pt | train.py or run_ucra.py | state_dict + model cfg |
| scaler.npz | train.py or run_ucra.py | mu, sd (inputs) + mu_y, sd_y (targets) |
| train_metrics.json | train.py | test pinball, coverage, n_train, n_test |
| ucra_results.json | run_ucra.py | 4 policies x 4 metrics + capacity + evolution_updates |
| ucra_log.csv | run_ucra.py | per-slot loop trace (the forensic record) |
| fig_demand_vs_reservation.png | run_ucra.py | demand vs UCRA R vs static-peak |
| fig_quantile_band.png | run_ucra.py | q05-q99 fan chart vs realized |
| fig_self_evolution.png | run_ucra.py | rolling violation rate + update markers |
| fig_kappa_sweep.png | run_ucra.py --sweep | violations & utilization vs kappa |
| drift_demo.json | demo_evolution.py | frozen vs evolving comparison |
| fig_drift_demo.png | demo_evolution.py | 2-panel surge figure |
| cttc_results.json | run_cttc_eval.py | 3 policies x 3 slice types + kappa sweep |
| fig_cttc_slices.png | run_cttc_eval.py | violation + over-provision bars |

## 6. Runtimes on your machine (CPU)

| Step | Time |
|---|---|
| download RAN | < 1 min (8 MB) |
| audit | ~10 s |
| train | ~3 min (50 epochs max, early stop) |
| run_ucra (+sweep) | ~20 s |
| demo_evolution | ~1 min |
| download CTTC | one-time 271 MB (resumable) |
| run_cttc_eval --max-samples 120 | ~2-4 min (parsing dominates) |
