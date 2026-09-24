# ==================================================================
# UCRA documentation generator -- PART 1 of 4
# Writes 3 file(s) into docs/: README.md, 01_big_picture.md, 02_pipeline_walkthrough.md
# Paste-safe by construction: this source contains NO triple-backtick
# runs; markdown fences are stored as @@FENCE@@ and restored on write.
# Save as create_ucra_docs_1of4.py in E:\UCRA and run:
#     python create_ucra_docs_1of4.py
# Run parts 1..4 in order from the UCRA root folder.
# ==================================================================
from pathlib import Path
import hashlib
import sys

_TOK = "@@FENCE@@"   # markdown-fence placeholder
_BT = chr(96) * 3    # the real markdown fence

FILES = {}

FILES['README.md'] = r'''# UCRA Documentation -- Complete Guide Set

This folder explains **everything** in the UCRA repository: what the algorithm
is, how every stage works, why each design decision was made, what every file
does, what every number in your results means, and how to extend or defend the
work. Start here and follow one of the reading orders below.

The companion file [`METHODOLOGY.md`](METHODOLOGY.md) is the compact
diagram-block-to-code mapping table used when writing the paper.

## Suggested reading orders

**"I ran it -- now I want to understand what I built"** (recommended first pass)

    01_big_picture  ->  02_pipeline_walkthrough  ->  09_results_interpretation
    (glossary 15 open in a second tab at all times)

**"I need to defend this in a paper / to a reviewer"**

    01 -> 03 (Stage 1 math) -> 04 (Phi math) -> 08 (metrics + baselines)
    -> 09 (results) -> 14 (paper map + limitations) -> 07 (dataset provenance)

**"I want to modify or extend the code"**

    10 (code reference) -> 11 (config guide) -> 03/04/05/06 (stage internals)
    -> 12 (experiments guide)

**"Something broke"**

    13_faq_troubleshooting  -- every error we hit during the build is listed
    there with the exact fix.

## The 16 guides

| # | File | What it covers |
|---|------|----------------|
| 01 | [01_big_picture.md](01_big_picture.md) | The problem, the idea, the 5-stage loop, a slot-by-slot narrative walk-through |
| 02 | [02_pipeline_walkthrough.md](02_pipeline_walkthrough.md) | Every command, every output file, data-flow with the real numbers from your run |
| 03 | [03_stage1_uncertainty.md](03_stage1_uncertainty.md) | Quantile LSTM, pinball loss, z-space training, coverage, rho_t |
| 04 | [04_stage2_phi_stage4_update.md](04_stage2_phi_stage4_update.md) | The Phi equation term by term, worked example, EWMA + hysteresis release |
| 05 | [05_stage3_allocation.md](05_stage3_allocation.md) | Risk-adaptive allocation weights, best-effort pool, leftover recycling |
| 06 | [06_stage5_self_evolution.md](06_stage5_self_evolution.md) | Drift triggers, reservoir replay, fine-tuning, your drift-demo JSON decoded |
| 07 | [07_data_sources.md](07_data_sources.md) | Both Zenodo datasets: schemas, loaders, capacity estimate, citations, BibTeX |
| 08 | [08_metrics_baselines.md](08_metrics_baselines.md) | The 4 metrics and 3 baselines: formulas, intuition, what "good" looks like |
| 09 | [09_results_interpretation.md](09_results_interpretation.md) | YOUR numbers (RAN + drift demo + CTTC) explained cell by cell |
| 10 | [10_code_reference.md](10_code_reference.md) | Every module, class and function with signatures and gotchas |
| 11 | [11_config_guide.md](11_config_guide.md) | Every knob in configs/default.yaml + tuning recipes |
| 12 | [12_experiments_guide.md](12_experiments_guide.md) | Reproduction checklist, ablations, sweep grids, seed variance |
| 13 | [13_faq_troubleshooting.md](13_faq_troubleshooting.md) | All real errors hit during the build and their fixes |
| 14 | [14_paper_map.md](14_paper_map.md) | Repo artifacts mapped to paper sections, claims, limitations, naming note |
| 15 | [15_glossary.md](15_glossary.md) | Every term used anywhere, in plain language |

## How the docs were written

Every equation, file name, function signature and result number in these
guides was verified directly against the repository source code and the actual
run outputs (RAN pipeline, drift demo, CTTC evaluation) on the date of
generation. If you change the code, re-check docs 03-06 and 09 first.
'''

FILES['01_big_picture.md'] = r'''# 01 -- The Big Picture: What UCRA Is and Why It Exists

## 1. The problem in one paragraph

A 5G operator sells "network slices" -- guaranteed levels of service (URLLC
for factory robots, eMBB for video, mMTC for millions of tiny sensors). Every
15 minutes the network must decide **how much capacity to reserve**. This is a
bet placed *before* demand is known:

- Reserve **too little** -> demand exceeds the reservation -> **violations**:
  dropped packets, missed latency deadlines, angry customers, contractual
  penalties.
- Reserve **too much** -> the extra capacity sits idle -> **waste**: you pay
  for spectrum/energy/servers that carry no traffic.

Classic approaches pick one static number (historical peak, or a forecast
mean) and hope. Both fail: the peak wastes ~30% of capacity all the time, the
mean violates roughly half the time. The core difficulty is that demand is
*uncertain* (you never know the next value exactly) and *non-stationary*
(behavior drifts -- new events, new apps, new days). UCRA is a closed-loop
answer to exactly these two difficulties.

## 2. The idea in one sentence

> Forecast the **whole probability distribution** of future demand (not just
> one number), transform that **uncertainty** into a single reservation size
> through an explicit risk formula (Phi), split that reservation across
> competing consumers under risk weighting, and **keep re-learning** whenever
> reality drifts away from what the model expects.

## 3. The five stages (your framework diagram, annotated)

@@FENCE@@
        Network Data  (traffic, capacity, topology, risk)
             |
             v
  +---------------------+
  | 1. UNCERTAINTY      |  quantile LSTM: one input window ->
  |    ESTIMATION       |  7 quantiles of future demand
  +---------------------+  outputs: u_hat (point), U_t (spread), rho_t (risk)
             |
             v
  +---------------------+
  | 2. Phi TRANSFORM    |  R_t = q_tau + kappa * spread * (1 + rho_w * rho_t)
  |    uncertainty ->   |  (floored and ceilinged by guardrails)
  |    reservation      |
  +---------------------+
             |
             v
  +---------------------+          +--------------------------------+
  | 3. RISK-ADAPTIVE    |          | Feedback: reservation error,   |
  |    ALLOCATION       |--------->| violations, utilization,       |
  +---------------------+          | drops / delay (CTTC)           |
             |                     +--------------------------------+
             v                              |            |
  +---------------------+                   |            |
  | 4. RESERVATION      |<------------------+            |
  |    UPDATE & RELEASE |   (EWMA smoothing + release hysteresis)
  +---------------------+                                |
             |                                           |
             v                                           v
  +-----------------------------------------------------------+
  | 5. SELF-EVOLVING LEARNING                                  |
  |  drift monitor -> trigger -> fine-tune on replay buffer    |
  +-----------------------------------------------------------+
@@FENCE@@

Stage roles in one line each:

| Stage | Question it answers | Code |
|---|---|---|
| 1 | "What might demand be, and how sure am I?" | `ucra/models/quantile_lstm.py`, `ucra/core/uncertainty.py` |
| 2 | "Given that uncertainty and risk, how much do I reserve?" | `ucra/core/transform.py` (`phi_transform`) |
| 3 | "Who gets what share of the reservation?" | `ucra/core/allocate.py` |
| 4 | "How do I move the reservation smoothly without oscillating?" | `ucra/core/transform.py` (`update_reservation`) |
| 5 | "The world changed -- how do I adapt without forgetting?" | `ucra/core/evolve.py` |

## 4. One slot in the life of UCRA (concrete numbers)

Follow one decision. Time is 17:00 on a real October evening; the slot is
15 minutes long. Demand is measured in data-volume units per slot.

1. **Input (Stage 1).** The last 96 slots (24 hours) of network demand are
   z-scored with training statistics and fed to the quantile LSTM. The model
   emits 7 quantiles (q05 ... q99) for each of the next 4 slots. For the next
   slot suppose it says: median q50 = 79,400 and q99 = 92,300.
2. **Uncertainty triple (Stage 1 output).** `u_hat = 79,400` (point forecast),
   `spread = 92,300 - 79,400 = 12,900` (uncertainty width U_t), and the risk
   indicator `rho_t = 0.0` (no recent violations, demand level within 2
   standard deviations of training mean -- nothing alarming).
3. **Reservation (Stage 2, Phi).** The 90% quantile is approximated from the
   median and spread: `q90 = 79,400 + (0.9-0.5)/0.49 * 12,900 = 89,930`.
   The risk buffer: `kappa * spread * (1 + rho_w * rho_t)
   = 0.5 * 12,900 * 1.0 = 6,450`. Target: `R_target = 96,380`. Guardrails:
   floor `1.05 * 79,400 = 83,370` (never below 5% over the point forecast),
   ceiling `0.95 * 143,954 = 136,757` (never near physical capacity). Neither
   binds, so `R_target = 96,380`.
4. **Smoothing (Stage 4).** The previous applied reservation was 93,100.
   EWMA with alpha = 0.3: `R_t = 0.7 * 93,100 + 0.3 * 96,380 = 94,084`.
   The reservation moves a third of the way toward the target -- no jumps.
5. **Splitting (Stage 3).** 10% of R_t (9,408) is kept as a best-effort pool;
   the other 84,676 is divided across the 75 sectors proportionally to their
   current demand (raised to pressure exponent 1.0). Sectors asking for less
   than their share have the surplus recycled to hungrier sectors.
6. **Reality check (feedback).** The slot ends; realized demand is 91,200.
   `91,200 < 94,084` -> no violation. Utilization this slot:
   `min(91,200/94,084, 1) = 0.969`. The monitors record (no violation, 91,200).
7. **Monitor (Stage 5, asleep).** Rolling violation rate over the last 96
   slots is ~0%, demand z-score ~1.1 -> no trigger. The model is not touched.
   (If the rolling rate had crossed 15% or z had crossed 3.0, the engine would
   fine-tune the LSTM on a replay buffer of recent windows -- see doc 06.)

Multiply this loop by 252 test slots and you have exactly what
`scripts/run_ucra.py` executed on your machine.

## 5. Design principles worth remembering

1. **No future peeking.** Reservations at slot t use only information up to t
   (model inputs, past violations, past demand). The only "cheating" object in
   the repo is the `oracle_quantile` baseline, which exists precisely to show
   what an upper bound looks like -- and UCRA still beats it on the combined
   objective.
2. **Guardrails over cleverness.** Phi's floor (`>= 1.05 * point forecast`)
   and ceiling (`<= 95% of capacity`) mean even a badly wrong forecast cannot
   produce an absurd reservation.
3. **Smoothness is a feature.** EWMA + release hysteresis (Stage 4) exist
   because oscillating reservations cause real operational churn.
4. **Adaptation is triggered, not continuous.** The model is only fine-tuned
   when drift/violation evidence crosses a threshold -- cheap when healthy,
   responsive when sick.
5. **Everything is a knob.** All thresholds live in `configs/default.yaml`
   (doc 11); nothing is hard-coded in the modules.

## 6. What UCRA is NOT

- Not a **packet scheduler**: it decides reservation sizes, not per-packet
  queues.
- Not an **admission controller**: it does not accept/reject slice requests.
- Not a **hardware capacity planner**: the "capacity" C = 1.3 x observed peak
  is a stand-in for the physical limit (the PM dataset does not publish one) --
  see doc 07 for the honest details.
- The acronym collides with a 2020 IEEE paper ("User-Centric Context-Aware
  Resource Allocation"); doc 14 has the exact related-work sentence to use.

## 7. Where to go next

- Run order and file map: doc 02.
- The math of each stage: docs 03, 04, 05, 06.
- What your printed results mean: doc 09.
- Every term: doc 15 (glossary).
'''

FILES['02_pipeline_walkthrough.md'] = r'''# 02 -- Pipeline Walkthrough: Commands, Files, and Data Flow

This doc follows the exact path your data took, with the real numbers from
your machine. Keep `configs/default.yaml` open beside it (doc 11 explains
every line of it).

## 1. Repository map

@@FENCE@@
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
@@FENCE@@

## 2. The data flow, with your actual numbers

@@FENCE@@
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
@@FENCE@@

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
'''


def write_all() -> None:
    dest = Path("docs")
    dest.mkdir(parents=True, exist_ok=True)
    print("=" * 66)
    print(f" UCRA docs generator -- part {1}/4")
    print("=" * 66)
    for name in FILES:
        text = FILES[name].replace(_TOK, _BT)
        data = text.encode("utf-8")          # raw bytes: LF kept, no CRLF surprises
        (dest / name).write_bytes(data)
        print(f"  [ok] docs/{name:<34} {len(data):>7,} bytes  md5 {hashlib.md5(data).hexdigest()[:8]}")
    print(f"  part {1}/4 done: {len(FILES)} file(s) written")


if __name__ == "__main__":
    write_all()
    print("  next: save & run create_ucra_docs_2of4.py")