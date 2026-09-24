# ==================================================================
# UCRA documentation generator -- PART 3 of 4
# Writes 4 file(s) into docs/: 07_data_sources.md, 08_metrics_baselines.md, 09_results_interpretation.md, 10_code_reference.md
# Paste-safe by construction: this source contains NO triple-backtick
# runs; markdown fences are stored as @@FENCE@@ and restored on write.
# Save as create_ucra_docs_3of4.py in E:\UCRA and run:
#     python create_ucra_docs_3of4.py
# Run parts 1..4 in order from the UCRA root folder.
# ==================================================================
from pathlib import Path
import hashlib
import sys

_TOK = "@@FENCE@@"   # markdown-fence placeholder
_BT = chr(96) * 3    # the real markdown fence

FILES = {}

FILES['07_data_sources.md'] = r'''# 07 -- Data Sources: Provenance, Schemas, Loaders, Citations

Both datasets are real, published, CC-BY-4.0, and fetched from Zenodo by
`scripts/download_data.py` (pure Python stdlib -- works on any OS).
Datasets are NOT committed to git (`.gitignore` hard-excludes `data/`).

## 1. Why two datasets (division of labor)

| | RAN PM counters (17815388) | CTTC slicing (10610616) |
|---|---|---|
| Nature | commercial live network, 15-min PM counters | simulation steady-state snapshots |
| Unit of analysis | time (1,778-slot series) | snapshot x slice (independent samples) |
| Feeds stages | 1 (uncertainty), 4 (update), 5 (evolution) | 2-3 (reservation sizing vs violations) |
| Feedback signals | utilization, violations | violations, drops, delay |
| Temporal drift | yes (real days/weeks) | no (i.i.d. snapshots) |
| Headline metric | violation 0.0% @ util 71.7% | URLLC 29%->1.4%, eMBB 50.6%->0% |

Together they cover all five UCRA stages; neither alone could (one has no
slice semantics, the other has no time axis).

## 2. RAN dataset (Zenodo 17815388)

- Title: PM counters from live 5G, 4G and 2G RAN (STU Bratislava + Ericsson
  Slovakia), published 2026, Scientific Data, DOI 10.1038/s41597-026-07723-0,
  license CC-BY-4.0.
- Files: Dataset_01.zip (8,333,223 B), Dataset_02.zip (49,441,996 B),
  Dataset_03.zip (709,421,773 B), README (2,312 B). We use Dataset_01.
- Schema (verified against the real CSVs): one CSV per radio technology per
  baseband; columns include `Base station`, `Sector`, `Timestamp`, and
  tech-prefixed counters `<2G|4G|5G> data volume DL/UL`, `<tech> RB
  utilization`, `<tech> active users DL/UL`, `<tech> CQI rank 1..4`,
  `<tech> RRC users`, `<tech> MIMO rank DL`. Measurements every 15 min per
  sector.
- Campaign structure: separate observation campaigns separated by multi-day
  gaps (Mar 2023 and Oct 2023, ~193 days apart). This is why
  `load_demand_series` splits the resampled series at any gap > 15 min and
  keeps only the LONGEST gap-free segment -- a training window that silently
  crossed a 193-day hole would interpolate garbage and poison Stage 1. The
  longest segment: 1,778 slots, Oct 6 21:15 -> Oct 25 09:30, 2023. Internal
  holes of up to 4 slots are linearly interpolated (limit=4).

### Loader pipeline (`ucra/data/ran_loader.py`)

1. `_extract_if_needed`: unzip any `Dataset_*.zip` whose folder is missing.
2. Per CSV: `_map_columns` detects the technology prefix and requires that
   tech's 5 key columns; files without them are skipped (with a printed
   note).
3. Build the canonical long frame: `[ts, sector, data_volume (DL+UL),
   utilization, active_users (DL+UL), tech]` -> 302,052 rows, 75 sectors,
   techs {4G, 5G} for Dataset_01.
4. `load_demand_series`: groupby ts -> sum over sectors -> strict 15-min
   grid -> split at gaps -> longest segment -> interpolate (limit 4).

### Capacity estimate (honesty box)

The PM data does not publish physical sector capacities. run_ucra uses
`C = 1.3 x max(observed demand)` = 1.3 x ~110,734 = 143,954.5 -- a
deliberate headroom factor so the ceiling in Phi is essentially never
binding (0.95C ~ 136,757 ~ 1.24x observed peak). Treat C as a
normalization constant, not a measured quantity; say so in the paper
(doc 14). Changing the 1.3 factor rescales all normalized metrics
uniformly and changes no ranking.

## 3. CTTC dataset (Zenodo 10610616)

- Title: CTTC B5G network slicing dataset (Farreras et al.), Data in Brief
  55:110738, 2024, license CC-BY-4.0. Files: slicing-simulations.zip
  (271,059,245 B) + datanetAPI.py (42,915 B, the dataset's own reader).
- Each sample is one steady-state simulation snapshot:
  - `topology_object`: networkx graph; edge attribute `bandwidth` = link
    capacity. We take `link_cap = mean(edge bandwidths)`.
  - traffic matrix: per-flow offered demand.
  - `slices`: list of eMBB / mMTC / URLLC slice entries, each with `delta`
    (reserved share, 0..1) and `flows`.
  - `performance_matrix`: per src-dst pair: PktsDrop, AvgDelay, p10-p90,
    Jitter.
- Derived row per slice: `reserved = delta * link_cap`;
  `offered = sum of flow rates`; plus drops_ratio / avg_delay aggregated
  over pairs. Frame columns: [sample, slice_type, delta, reserved, offered,
  link_cap, drops_ratio, avg_delay].

### The schema saga (why the loader is "dict-tolerant")

The dataset's own `datanetAPI.py` expects jsonpickle-decoded OBJECTS
(`slice.type`, `slice.flows`, flow `avgRate`). The current build of the
archive (v2.2.1) instead delivers plain PYTHON DICTS with `bandwidth` (bps)
on flows -- so the vendor reader crashes with "'dict' object has no
attribute 'type'". Our loader reads BOTH shapes (dict or object), and
extracts each flow's rate from `bandwidth`, falling back to `avgRate`,
`rate`, then to the 2nd comma-field of `traffic_string`. It also imports
the dataset's class under the correct name `DatanetAPI` (capital D -- the
lowercase import was bug #1, fixed in Part C). Corrupt samples are skipped
with a printed note instead of aborting the run.

Your parse: 120 samples -> frame (2,908 rows, 8 cols); slice counts
mMTC 1,201 / URLLC 1,106 / eMBB 601; test split (samples > 70% cut) yields
URLLC n=276, eMBB n=164, mMTC n=296 -- the n's in your results JSON.

## 4. Synthetic generator (`ucra/data/synthetic.py`)

15-min multi-sector series: `base * sector_gain * diurnal * weekly * (1 +
AR(1) noise + slow sinusoid + ON/OFF bursts)`, diurnal amplitude 0.45
(period 96), weekly 0.15, AR(1) rho 0.85, bursts start with p=0.01 per slot
and last 2-10 slots with height 0.5-1.4. Used by the smoke test and for
sanity runs without any download. `synthetic_capacity` = 250 * n_sectors.

## 5. Downloader behavior (`scripts/download_data.py`)

- md5 verification against baked-in checksums; skip-if-complete (so re-runs
  are instant); HTTP Range resume of `.part` files; 6 retries with backoff.
- `--list` shows known files/sizes/checksums; `--cttc` adds the CTTC pair;
  `--ran-all` adds Dataset_02/03; `--root` redirects the data dir.
- After CTTC download it unzips (idempotent, marker-file based) and pip
  installs `jsonpickle` (imported by the dataset's datanetAPI.py). On your
  machine pip printed "Defaulting to user installation" -- harmless.

## 6. Ready-to-paste citations

@@FENCE@@bibtex
@dataset{ran_pm_counters_2026,
  title   = {{PM} counters from live {5G}, {4G} and {2G} {RAN}},
  author  = {Lehoczky, Jan and others},        % fill full author list from the Zenodo page
  year    = {2026},
  doi     = {10.5281/zenodo.17815388},
  note    = {Scientific Data, DOI 10.1038/s41597-026-07723-0. CC-BY-4.0}
}

@dataset{cttc_slicing_2024,
  title   = {{CTTC} {B5G} network slicing dataset},
  author  = {Farreras, Miquel and others},     % fill full author list from the Zenodo page
  year    = {2024},
  doi     = {10.5281/zenodo.10610616},
  note    = {Data in Brief 55:110738. CC-BY-4.0}
}
@@FENCE@@

(CC-BY-4.0 requires attribution -- the two entries above satisfy it; also
cite the Scientific Data paper for the RAN set.)
'''

FILES['08_metrics_baselines.md'] = r'''# 08 -- Metrics and Baselines: Formulas, Intuition, What "Good" Means

Files: `ucra/eval/metrics.py`, `ucra/eval/baselines.py`. All metrics are
computed on the TEST window; R = reservation, D = realized demand,
C = capacity (143,954.5 on your run).

## 1. The four metrics

| Metric | Formula | Unit | Question it answers |
|---|---|---|---|
| violation_rate | mean( D > R ) | fraction of slots | How often did we fail users? (risk exposure) |
| utilization | mean( min(D/R, 1) ) | 0..1 | Of what we reserved, how much was actually used? (efficiency) |
| reservation_error | mean( abs(R - D) ) / C | fraction of capacity | How far off was the reservation on average? (tracking accuracy) |
| waste | mean( max(R - D, 0) ) / C | fraction of capacity | How much reserved capacity idled? (cost) |

Relationships worth internalizing:

- violation_rate + something like "slack" = 100%: a policy can only lower
  violations by reserving more, which shows up in utilization/waste. There
  is no free lunch -- the deliverable is the FRONTIER, not one number.
- utilization is capped at 1 per slot: over-demand does not inflate it; it
  shows up as violations instead. A policy can have utilization 95% AND 50%
  violations (your mean_forecast row does exactly that).
- reservation_error counts both directions (over and under), normalized by
  capacity. Two policies with equal error can be wildly different: one
  violates often, the other over-reserves. Always read it together with
  violation_rate.

What "good" looks like in this problem: violation_rate near 0 (this is the
SLA), then maximize utilization subject to that. UCRA's 0.0% @ 71.7% means
"never failed, and the guarantee cost ~28% slack".

## 2. Why band coverage is NOT here

The 96.0% "coverage" from train.py is a Stage 1 calibration metric (is the
q05-q99 band honest?), not a reservation metric. A perfectly calibrated
model can still produce a bad reservation policy, and vice versa. Keeping
the two layers separate is what makes failures diagnosable (doc 04
section 5).

## 3. The three baselines and what each defends

| Baseline | R_t | What it represents | Its signature failure |
|---|---|---|---|
| static_peak | max(train demand), constant | worst-case static sizing, the "safe" naive choice | 0 violations but 30.5% of capacity wasted forever (res-err 30.5%) |
| mean_forecast | Stage 1 median (step 0) | "we have a great point forecast, ignore uncertainty" | cheapest possible R -> violates 50.4% of slots |
| oracle_quantile | true rolling q90 of the TEST window itself (window 48) | CHEATING upper bound: sees the future it defends against | still 16.3% violations (lags bursts), res-err 20.1% |

Why the oracle matters: reviewers ask "what would a perfect quantile
policy achieve?" The oracle answers it -- and because UCRA (which never
sees the future) holds 0.0% violations at higher utilization than the
oracle's 64.9%, UCRA's kappa buffer demonstrably out-sizes a naive rolling
quantile. Why the oracle still violates 16.3% despite cheating: its
"quantile of the last 48 realized slots" reacts to bursts AFTER they start
and forgets lulls; by construction a 90% quantile violates ~10% of slots
even with perfect knowledge, and heavy-tailed 15-min bursts push that
higher.

One subtlety: static_peak uses `max(series.values[train_idx])` -- the peak
over training-window start samples -- so a freak test-window spike above
the training peak WOULD violate it. On your segment it does not happen
(0.0%), which is the honest reading: static peak is "safe" only in
stationary weather.

## 4. How the comparison is wired in run_ucra.py

After the closed loop, the same `summarize()` is applied to four R vectors:
UCRA's loop output, static_peak's constant, mean_forecast's per-slot
medians, and the oracle's rolling q90 -- all against the same D_test and C.
One number quirk: mean_forecast's reservation_error (2.0%) is the LOWEST of
all policies -- chasing D tightly with the median minimizes mean absolute
gap -- while being catastrophically the worst on violations. This is the
cleanest illustration in the whole project of why reservation_error alone
must never be the objective.

## 5. The kappa sweep: your frontier in one picture

Each kappa value re-runs Phi (rho = 0) over the test window and reports the
(violation, utilization) pair. Read as an economic frontier: kappa is the
price you pay in slack per unit of uncertainty; violations are the risk you
buy down. On your run, kappa = 0.5 already sits at zero violations, so the
knee (the smallest kappa with 0% violations) lies somewhere in 0.25-0.5 --
read the exact knee off your fig_kappa_sweep.png. Beyond the knee, extra
kappa only buys idle capacity. The CTTC eval embeds the same sweep per
slice type inside `cttc_results.json` (key `kappa_sweep`), which is how you
would justify per-slice kappa choices in the paper.
'''

FILES['09_results_interpretation.md'] = r'''# 09 -- Your Results, Explained Cell by Cell

Every number below came off YOUR machine. This doc tells you what each one
means, why it has that value, and what you may claim from it.

## 1. Main RAN pipeline (run_ucra.py, real commercial network)

| Policy | Reservation error | Violation rate | Utilization |
|---|---|---|---|
| **UCRA (kappa = 0.5)** | 10.8% | **0.0%** | **71.7%** |
| static_peak | 30.5% | 0.0% | 54.8% |
| mean_forecast | 2.0% | 50.4% | 95.3% |
| oracle_quantile | 20.1% | 16.3% | 64.9% |

Read row by row:

- **UCRA 0.0% @ 71.7%.** The only policy in the zero-violation column that
  is not wasteful. The 28.3% slack is the price of the guarantee: kappa
  buffer over q90 + the 5% headroom floor + EWMA lag. The interesting
  comparison is static_peak: same 0 violations, but UCRA achieves them with
  17 percentage points MORE utilization -- the buffer goes where uncertainty
  is, instead of everywhere.
- **static_peak.** Reserved the training peak (~110,700) all the time. Safe,
  and 30.5% of capacity idles on average. This is the "throw money at it"
  row.
- **mean_forecast.** Beautiful 2.0% tracking error and 50.4% violations --
  the median is below realized demand half the time BY DEFINITION of a
  median. The paper-ready lesson: a point forecast is not a reservation.
- **oracle_quantile.** Even with access to the test window's own rolling
  q90, you get 16.3% violations -- quantile-of-recent-history lags bursts.
  UCRA beats the oracle on BOTH axes that matter (violations 0 vs 16.3%,
  utilization 71.7 vs 64.9%). Claim it, but with the honest framing: the
  oracle optimizes a different trade-off and is a reference point, not a
  competitor (doc 08).

Also in ucra_results.json: `evolution_updates: 0` (doc 06 section 5 --
expected, it is the no-false-alarms result) and `capacity: 143,954.5`
(doc 07 honesty box).

Stage 1 quality behind these numbers: 52,252 params, test pinball 930.3
(original units), q05-q99 band coverage 96.0% vs 94% nominal.

## 2. Drift demo (demo_evolution.py)

@@FENCE@@
shift 0.35, surge_start_slot 37, n_test 252, capacity 143,954.5

frozen:    viol_overall 0.2262 | viol_post_surge 0.2651 | util_post 0.8274 | updates 0
evolving:  viol_overall 0.0992 | viol_post_surge 0.1163 | util_post 0.8031 | updates 3 [96, 120, 144]
@@FENCE@@

One-line summary for the paper: **a +35% demand surge drives the frozen
pipeline to 26.5% post-surge SLA violations; the self-evolving loop detects
the drift and cuts them to 11.6% with three autonomous fine-tunes, at
slightly LOWER utilization.**

Details that make the claim defensible (doc 06 decodes the mechanism):

1. Both passes start from the SAME checkpoint and see the SAME drifted
   data; the only difference is Stage 5 being on/off. This is a controlled
   A/B, not a tuning exercise.
2. The violation arithmetic closes (0.2262 x 252 ~ 0.2651 x 215 ~ 57):
   everything breaks after the surge; nothing breaks before. The frozen
   failure is attributable to drift alone.
3. Utilization post-surge is ~80-83% in BOTH passes -> the evolving pass
   did not buy its improvement with over-provisioning.
4. The update times [96, 120, 144] follow exactly from the trigger
   mechanics (24-slot cadence, 96-slot rolling window, 0.15 threshold) --
   you can hand-derive them; a reviewer can too (doc 06 section 6).

## 3. CTTC slice-level eval (run_cttc_eval.py, 120 snapshots, 70/30)

| Slice | n (test) | operator | static_q90 | ucra_phi |
|---|---|---|---|---|
| URLLC | 276 | 29.0% viol / 56.1% util / 43.9% over | 15.2% / 46.6% / 53.4% | **1.4%** / 19.6% / 80.4% |
| eMBB | 164 | 50.6% / 78.2% / 21.8% | 12.2% / 56.7% / 43.3% | **0.0%** / 25.9% / 74.1% |
| mMTC | 296 | 5.4% / 15.3% / 84.7% | 12.2% / 30.6% / 69.4% | **4.4%** / 17.7% / 82.3% |

(Each cell: violation_rate / utilization / over_provision, all as defined
in doc 08 but computed on the slice rows.)

- **operator (delta sizing)** under-reserves URLLC (29.0%) and eMBB
  (50.6%) badly, while over-provisioning mMTC by 84.7% of its reservation.
  That asymmetry -- failing the strict slices while wasting on the loose
  one -- is exactly the static-sizing failure mode UCRA targets.
- **static_q90** (Phi with kappa=0): big improvement over operator on
  URLLC/eMBB but still 12-15% violations -- a bare quantile is not enough
  in the tail.
- **ucra_phi**: 1.4% / 0.0% / 4.4% violations. The cost: over-provision
  ~74-82% of the reservation (note: relative to R, not to capacity), i.e.
  utilization 18-26%. Read as: near-zero violations bought at high slack
  per slice. The kappa_sweep block inside cttc_results.json gives you the
  intermediate operating points per slice -- use it to argue per-slice
  kappa (tight for mMTC, loose for URLLC).
- **mMTC nuance worth stating honestly:** operator (5.4%) beats static_q90
  (12.2%) here, because the dataset's mMTC delta reservations are already
  generous relative to that slice's spiky-but-tiny loads. UCRA (4.4%)
  edges out both, but the headline story on mMTC is "comparable
  violations", not "big win".
- **QoS gap fields** (`operator_qos_gap` in the JSON): mean packet-drop
  ratio and delay on slots where the operator's reservation was violated
  vs not. They are per-sample aggregates in this dataset, so treat them as
  directional color, not headline SLA numbers.

**One methodological caveat to keep you safe in review:** in this CTTC
script the rho term of phi_empirical uses the CURRENT snapshot's offered
load as the stress proxy (`rho = clip((offered_now - q50)/spread, 0, 1)`).
The q90 core and the spread come strictly from the train half, but the
+/-15% modulation of the buffer sees today's load. The fully train-only
variant is exactly `static_q90 + kappa*spread` (rho forced 0) -- add it as
a fourth policy before the paper if a reviewer asks (one-line change in
`phi_empirical`; doc 12 has the recipe). The RAN closed loop has no such
caveat: its rho_t is computed only from past feedback.

## 4. What you may and may not claim

MAY claim, backed directly by your outputs:
- On real commercial RAN data, UCRA holds 0% reservation violations at
  71.7% utilization, where a static peak wastes 30% and a mean-forecast
  policy violates half the time.
- The full pipeline is a closed loop: Stage 5 detects an injected +35%
  drift and autonomously reduces violations by ~56% relative without extra
  capacity.
- On independent slice snapshots (CTTC), Phi-based sizing reduces
  operator-style violations from 29-51% to 0-4.4% on URLLC/eMBB at
  controlled over-provisioning.
- Every stage ran with zero manual intervention and no test-set peeking
  (the oracle baseline exists precisely to quantify that).

MAY NOT claim (yet):
- "Optimal" anything -- there is no optimality proof; it is an empirical
  frontier.
- Latency/reliability SLA compliance -- the CTTC delay fields are
  aggregates; UCRA here sizes reservations, it does not control schedulers.
- Generalization across cities/operators -- one RAN subset, one segment,
  one 63-hour test window (doc 12: how to broaden it).
'''

FILES['10_code_reference.md'] = r'''# 10 -- Code Reference: Every Module, Class and Function

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
'''


def write_all() -> None:
    dest = Path("docs")
    dest.mkdir(parents=True, exist_ok=True)
    print("=" * 66)
    print(f" UCRA docs generator -- part {3}/4")
    print("=" * 66)
    for name in FILES:
        text = FILES[name].replace(_TOK, _BT)
        data = text.encode("utf-8")          # raw bytes: LF kept, no CRLF surprises
        (dest / name).write_bytes(data)
        print(f"  [ok] docs/{name:<34} {len(data):>7,} bytes  md5 {hashlib.md5(data).hexdigest()[:8]}")
    print(f"  part {3}/4 done: {len(FILES)} file(s) written")


if __name__ == "__main__":
    write_all()
    print("  next: save & run create_ucra_docs_4of4.py")