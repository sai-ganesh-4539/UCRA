# ==================================================================
# UCRA documentation generator -- PART 4 of 4
# Writes 5 file(s) into docs/: 11_config_guide.md, 12_experiments_guide.md, 13_faq_troubleshooting.md, 14_paper_map.md, 15_glossary.md
# Paste-safe by construction: this source contains NO triple-backtick
# runs; markdown fences are stored as @@FENCE@@ and restored on write.
# Save as create_ucra_docs_4of4.py in E:\UCRA and run:
#     python create_ucra_docs_4of4.py
# Run parts 1..4 in order from the UCRA root folder.
# ==================================================================
from pathlib import Path
import hashlib
import sys

_TOK = "@@FENCE@@"   # markdown-fence placeholder
_BT = chr(96) * 3    # the real markdown fence

FILES = {}

FILES['11_config_guide.md'] = r'''# 11 -- Configuration Guide: Every Knob in default.yaml

`configs/default.yaml` is the single source of truth. Nothing is hard-coded
in the modules. Below: every key, its default, what it does, and which way
to move it.

## seed: 42
Feeds torch/numpy RNG. Change it to quantify training variance (doc 12);
expect +-1-3 points of coverage across seeds/platforms.

## paths
| Key | Default | Meaning |
|---|---|---|
| raw_data | data | where downloads/unzips live |
| outputs | outputs | every artifact (doc 02 section 5) |

## data
| Key | Default | Notes |
|---|---|---|
| dataset | ran | "ran" (main results), "cttc" (per-sample series), "synthetic" (no download) |
| ran.subset | dataset01 | dataset01/02/03 -- 01 is the 8 MB one with 75 sectors |
| ran.target | data_volume | the demand series (DL+UL summed) |
| ran.agg | sum | sum over sectors ("mean" also supported) |
| ran.resample | null | series is already 15-min; a pandas freq string would regrid |
| cttc.samples_dir | data/cttc/slicing-simulations | unzipped archive location |
| cttc.slices | [eMBB, mMTC, URLLC] | informational; the loader keeps whatever the samples contain |
| synthetic.n_steps / n_sectors | 2016 / 8 | 3 weeks x 15-min x 8 sectors |

## model (Stage 1)
| Key | Default | Effect of changing it |
|---|---|---|
| seq_len | 96 | input history (24 h). Longer -> slower, more context; 96 matches the diurnal cycle |
| horizon | 4 | predicted steps. Only step 0 drives the loop today (doc 03 section 4) |
| hidden_size | 64 | capacity; 32 trains faster, 128 may overfit 1.6k windows |
| num_layers | 2 | keep 2 for dropout to exist (torch disables dropout on single-layer LSTM) |
| dropout | 0.1 | regularization |
| quantiles | [0.05 ... 0.99] | Phi uses 0.5 and 0.99; train.py coverage uses 0.05/0.99. Keep 2-decimal values (doc 10 gotcha) |
| batch_size | 64 | smaller -> noisier gradients, sometimes better quantiles |
| lr | 0.001 | Adam; Stage 5 fine-tune uses 0.3x this |
| max_epochs | 50 | early stopping usually fires first |
| patience | 6 | epochs without val improvement before stopping |
| train_val_test | [0.7, 0.15, 0.15] | chronological split fractions |

## phi (Stage 2) -- the identity of the method
| Key | Default | Effect |
|---|---|---|
| base_tau | 0.9 | which quantile the reservation sits on. 0.95 -> stricter, more slack; 0.75 -> leaner, more violations |
| kappa | 0.5 | THE risk dial (doc 04 section 3). 0 = bare quantile + floor |
| rho_weight | 0.3 | feedback authority. 0 disables risk adaptation in Phi (Stage 5 still fine-tunes the model) |
| min_headroom | 0.05 | floor = (1+this) * u_hat. Protects against overconfident forecasts |
| max_reserve_ratio | 0.95 | ceiling = this * capacity. Keep < 1.0 |

## allocation (Stage 3)
| Key | Default | Effect |
|---|---|---|
| pressure_exponent | 1.0 | >1 favors big consumers, <1 flattens (doc 05 section 2) |
| risk_exponent | 0.5 | shape of the risk premium; 0 disables |
| best_effort_share | 0.1 | capacity kept out of guarantees per slot |

## update (Stage 4)
| Key | Default | Effect |
|---|---|---|
| ewma_alpha | 0.3 | higher = snappier, noisier; lower = smoother, laggier. If violations cluster where R_target was already high, lower this |
| release_hysteresis | 0.1 | release requires >10% of CAPACITY idle AND a shrinking target. Lower = releases sooner |
| drift_window | 96 | rolling window for rho_t, plots, and trigger stats |

## evolution (Stage 5)
| Key | Default | Effect |
|---|---|---|
| violation_trigger | 0.15 | fine-tune when the 96-slot violation rate exceeds this. Lower = more sensitive (and more false positives) |
| drift_zscore | 3.0 | or when demand leaves 3 training sigmas |
| replay_size | 512 | buffer for continual learning (doc 06 section 3) |
| finetune_epochs | 3 | keep tiny on purpose; the demo --ft-epochs flag overrides |
| eval_every | 96 | trigger check cadence (24 h). The demo defaults to 24 slots instead |

## eval
| Key | Default | Meaning |
|---|---|---|
| metrics | [reservation_error, violation_rate, utilization, waste] | doc 08 |
| baselines | [static_peak, mean_forecast, oracle_quantile] | doc 08 |
| sweep_kappa | [0.0 ... 2.0] | the frontier grid for --sweep and CTTC |

## Tuning recipes

- **More conservative (carrier SLA mode):** base_tau 0.95, kappa 1.0,
  rho_weight 0.5. Expect utilization drop of ~5-10 points, still ~0
  violations.
- **Leaner (cloud cost mode):** kappa 0.25, min_headroom 0.02. Watch
  violations reappear in the sweep first.
- **Faster adaptation:** update.ewma_alpha 0.5, evolution.eval_every 24.
  Trade smoothness for responsiveness.
- **Sanity mode (no downloads):** data.dataset synthetic; the smoke test
  config in tests/ is an even smaller variant.
- After ANY config change, re-run audit -> (delete outputs/model.pt if the
  model config changed) -> train -> run_ucra --sweep. model.pt is only
  invalidated automatically when run_ucra itself trains; if you edited model
  hyperparameters but a checkpoint exists, DELETE it or you will silently
  evaluate the old architecture.
'''

FILES['12_experiments_guide.md'] = r'''# 12 -- Experiments Guide: Reproduce, Ablate, Extend

## 1. Reproduction checklist (paper numbers)

From a clean clone:

@@FENCE@@
pip install -r requirements.txt
python scripts/download_data.py                     # RAN, 8 MB
python scripts/audit_data.py   --config configs/default.yaml
python scripts/train.py        --config configs/default.yaml
python scripts/run_ucra.py     --config configs/default.yaml --sweep
python scripts/demo_evolution.py --config configs/default.yaml
python scripts/download_data.py --cttc              # 271 MB, one-time
python scripts/run_cttc_eval.py --config configs/default.yaml --max-samples 120
python -m pytest tests/test_smoke.py -q
@@FENCE@@

Expected (your reference machine): 302,052 rows / 75 sectors; segment 1,778
slots; capacity 143,954.5; 52,252 params; coverage ~93-96%; UCRA 0.0% viol /
71.7% util / 10.8% res-err; demo updates [96, 120, 144]; CTTC URLLC 1.4% /
eMBB 0.0% / mMTC 4.4%. Tolerances: coverage +-3 points and small metric
drift across torch versions/platforms (doc 03 section 7). Everything else
(Stages 2-4, baselines math) is deterministic given pred_q.

## 2. Grid: drift severity (one table for the paper)

@@FENCE@@
for s in 0.15 0.25 0.35 0.50:
    python scripts/demo_evolution.py --config configs/default.yaml --shift {s}
@@FENCE@@
(each run overwrites outputs/drift_demo.json -- rename it after each run,
e.g. drift_demo_s015.json). Record: frozen vs evolving viol_post_surge,
updates count, util_post. Expected shape: gap widens with shift until even
Stage 5 cannot keep up; that turning point is itself a finding.

## 3. Grid: drift timing and adaptation speed

- `--at 0.05 / 0.15 / 0.30`: surge early/mid/late in the window. Late
  surges leave less time to adapt -> check whether 3 updates still suffice.
- `--eval-every 12 / 24 / 48`: trigger cadence vs adaptation lag. With 12,
  expect the first update earlier (the window fills past 15% sooner).
- `--ft-epochs 3 / 10 --lr-scale 0.3 / 1 / 3`: we already verified
  insensitivity at the default; document one row to preempt the
  "did you tune the fine-tune?" question.

## 4. kappa frontier, both datasets

- RAN: `run_ucra.py --sweep` prints/renders the sweep; for a TABLE, read
  `fig_kappa_sweep` inputs or re-run the snippet inside run_ucra.py (the
  sweep block) dumping per-kappa `summarize()` to JSON. For a per-kappa JSON
  on the RAN side, copy that block into a small script -- 15 lines, no code
  changes needed.
- CTTC: the per-type `kappa_sweep` dict is ALREADY inside
  `outputs/cttc_results.json` -- table-ready.

## 5. Ablations reviewers ask for (all config-only)

| Question | Config change | Expected reading |
|---|---|---|
| Does the risk feedback (rho) matter? | phi.rho_weight = 0 | On clean RAN almost nothing changes (rho ~ 0 anyway); on the drift demo the frozen gap widens |
| Does the kappa buffer matter vs bare quantile? | phi.kappa = 0 | violations jump to static_q90-ish levels |
| Does hysteresis matter? | update.release_hysteresis = 0 | churn (R oscillation) increases; violations ~same |
| Does EWMA lag hurt? | update.ewma_alpha 0.1 vs 0.5 | smoothness vs lag trade-off visible in ucra_log.csv |
| Is Stage 5 needed if Phi is already risk-aware? | demo: evolving vs frozen IS the ablation | doc 06 numbers |
| Does the headroom floor bind? | phi.min_headroom 0 | small violation uptick during very confident forecasts |

## 6. Wiring per-slice risk into Stage 3 (the honest upgrade)

`risk_adaptive_allocate` already accepts `risks`. For a slice-criticality
experiment on CTTC data, build risks from the frame, e.g.
{"URLLC": 0.8, "eMBB": 0.3, "mMTC": 0.1} mapped per row, and allocate the
per-sample aggregate reservation across the three slice demands. One
afternoon of work; makes Stage 3 non-trivial in the paper.

## 7. Train-only Phi variant for CTTC (removes the rho caveat)

In `scripts/run_cttc_eval.py`, either set `phi.rho_weight = 0` for a fourth
policy run, or add:

@@FENCE@@python
def phi_empirical_clean(hist, phi_cfg):
    q50  = float(np.quantile(hist, 0.5)); q_tau = float(np.quantile(hist, phi_cfg["base_tau"]))
    q99  = float(np.quantile(hist, 0.99)); spread = max(q99 - q50, 1e-9)
    return q_tau + phi_cfg["kappa"] * spread          # rho = 0, train-only
@@FENCE@@

Report it as "Phi (train-only)" next to ucra_phi; the gap between the two
isolates the stress-modulation term.

## 8. Seeds and variance (do this before the paper)

`run_ucra.py` seeds everything from cfg.seed, but torch kernels differ
across builds, so: pick 3 seeds (41, 42, 43), rerun train + run_ucra for
each (rename outputs/ between runs), report mean +- sd for coverage,
violation, utilization. Stages 2-4 need no reseeding. Typical spread on
this data: coverage 93-96%, UCRA violations stay 0.0%, utilization
+-1-2 points.

## 9. Broadening the evidence base (optional, ranked by value/effort)

1. `python scripts/download_data.py --ran-all` + a loop over
   data.ran.subset in {dataset01, dataset02, dataset03} -- three networks
   instead of one (03 is 709 MB).
2. `--max-samples 250` on CTTC (the parser is linear; doubles the test
   rows).
3. Dataset_02/03 have different sector counts and possibly 2G files -- the
   loader handles both automatically (tech-aware mapping).

## 10. Recording results (what to keep per run)

For each experiment: the JSON(s) (ucra_results / drift_demo / cttc_results),
the CSV log, the PNGs, the config diff, torch version (`python -c "import
torch; print(torch.__version__)"`), and the git commit. outputs/ is
git-ignored -- zip the folder per run and store it outside the repo (or
commit selected JSONs under results/ if you prefer them in git).
'''

FILES['13_faq_troubleshooting.md'] = r'''# 13 -- FAQ and Troubleshooting (Every Real Error We Hit)

These are not hypothetical: each item below actually happened during the
build/validation of this repo. Find your symptom, get the cause and the fix.

## Data & downloads

**Q: The download keeps dying / resumes from scratch.**
A: It does not -- `download_data.py` resumes `.part` files via HTTP Range
requests (we tested a 3 MB partial resume) and retries 6 times. Just re-run
the same command. If a file is complete it is skipped via size + md5.

**Q: `pip` says "Defaulting to user installation because normal
site-packages is not writeable".**
A: Harmless Windows notice; the package installs into your user site and
works.

**Q: `[ran_loader] skip (no tech columns): <file>.csv`**
A: Expected. Some CSVs in the archive do not match the tech-prefixed PM
schema; the loader skips them loudly and continues. Only worry if ALL files
are skipped -> then you are pointing at the wrong data root.

**Q: `FileNotFoundError: No dataset files under data/ran...`**
A: Run `python scripts/download_data.py` from the repo root first (or
check `--root`).

**Q: Why is my series only 1,778 slots when the audit says 302,052 rows?**
A: 302,052 = rows across ALL sectors x technologies x campaigns. After
summing sectors and cutting at the 193-day campaign gap, the longest
gap-free 15-min segment is 1,778 slots. The gap cutting is deliberate
(doc 07 section 2) -- do not "fix" it.

## CTTC

**Q: `'dict' object has no attribute 'type'` (crash inside datanetAPI).**
A: The dataset build ships plain dicts where its own reader expects
jsonpickle objects. Fixed: `ucra/data/cttc_loader.py` reads both shapes.
If you see this, your copy of the loader predates Part D -- re-run
`create_ucra_partD.py`.

**Q: `ImportError: datanetAPI.py must sit inside the unzipped dataset
folder`.**
A: Run `python scripts/download_data.py --cttc` (it downloads, unzips, and
pip-installs jsonpickle), then re-run the eval.

**Q: The class is `datanetAPI` or `DatanetAPI`?**
A: `DatanetAPI` (capital D). The lowercase import was a real bug we hit and
fixed in Part C.

**Q: CTTC eval is slow / memory heavy.**
A: Parsing dominates; use `--max-samples 120` (your current setting). 250
is the ceiling we consider "plenty" (doc 12 section 9).

## Training (Stage 1)

**Q: Training stalls -- loss flat around 60-ish, predictions near zero.**
A: You are training on RAW-scale targets. The pinball loss has bounded
gradients; at target scale ~50,000 it cannot climb. The repo trains in
z-space (scY) and inverts afterwards -- keep it that way. This was real
bug #2 of the project.

**Q: `UserWarning: ... variable needed for gradient computation ...
tl += float(l) * len(xb)`.**
A: Historical; fixed via `float(l.detach())` in Part D. If you still see
it, your quantile_lstm.py predates Part D.

**Q: Coverage came out 96.0% but a friend running the same repo gets
93.25%.**
A: Cross-platform torch nondeterminism (2.13 Windows vs 2.14 Linux in our
case), same seed. Both healthy. Report mean +- sd over seeds (doc 12
section 8).

**Q: I changed model config but results look like the old model.**
A: `outputs/model.pt` still exists, so run_ucra loads it. Delete the
checkpoint and retrain (doc 11, last bullet).

## Running the loop

**Q: `evolution_updates: 0` -- is Stage 5 broken?**
A: No. Zero violations + no >3-sigma drift -> the trigger never fires. To
see Stage 5 work, run `scripts/demo_evolution.py` (doc 06).

**Q: Stage 5 never fired in the demo either / fired at odd times.**
A: Trigger checks happen every `eval_every` slots and need the 96-slot
rolling violation window to exceed 0.15. With the default demo cadence
(24) on a +35% surge, updates land at [96, 120, 144]. See the derivation
in doc 06 section 6.

**Q: Violations cluster where `R_target` was already high but `R` lagged.**
A: Stage 4 smoothing too slow -> lower `update.ewma_alpha` (diagnose via
ucra_log.csv columns R vs R_target, doc 04 section 5).

## Git / environment (Windows)

**Q: `git push` fails with a network/DNS error but PyPI works.**
A: Transient DNS. Retry later (it worked for you on the next push:
cf334f1 and cdde350 are on origin/main).

**Q: PowerShell vs bash commands in the README.**
A: Everything repo-provided is pure Python on purpose: use
`python scripts/download_data.py` (not the .sh). The .sh exists only for
Unix convenience.

## Reproducibility sanity checks

- Same config + same checkpoint -> identical loop results (Stages 2-4 are
  deterministic numpy).
- Fresh train on another machine -> coverage within a few points; loop
  metrics nearly identical because Phi reacts to quantile SHAPES, which
  are stable.
- `python -m pytest tests/test_smoke.py -q` must always pass -- it
  exercises every stage in seconds on synthetic data.
'''

FILES['14_paper_map.md'] = r'''# 14 -- Paper Map: From Repo Artifacts to a Submission

## 1. Section plan with repo sources

| Paper section | Content | Source in repo |
|---|---|---|
| Abstract | 5-stage closed loop; real RAN: 0% viol @ 71.7% util; drift: -56% relative violations; CTTC slices: 29-51% -> 0-4.4% | README results; doc 09 |
| 1 Introduction | reservation dilemma; uncertainty as first-class input; contributions list | doc 01 sections 1-2 |
| 2 Related work | static sizing, point-forecast policies, quantile forecasting, continual learning; **naming-collision sentence below** | docs/METHODOLOGY.md; doc 01 section 6 |
| 3 Methodology | the five stages + feedback; equations 1-5 = phi_transform, update_reservation, rho_t, weights, trigger | docs 03-06; diagram in doc 01 section 3 |
| 4 Experimental setup | both datasets (provenance, license), capacity heuristic, splits, compute (CPU, 3 min) | doc 07; doc 02 |
| 5 Results | RAN table; kappa frontier; drift A/B; CTTC per-slice table | doc 09 tables |
| 6 Discussion & limitations | below | doc 09 section 4; doc 07 honesty box |
| Reproducibility statement | config + seeds + commit + Zenodo DOIs | doc 12 section 8, 10 |

**Required related-work sentence (acronym collision):** a 2020 IEEE paper
"User-Centric Context-Aware Resource Allocation for Network Slicing"
(IEEE Access 8, DOI 10.1109/ACCESS.2020.3046198) already uses "UCRA". Use:
*"Unlike the user-centric context-aware resource allocation of [2020],
UCRA here denotes an uncertainty-driven reservation transformation with
self-evolving risk adaptation; the two address complementary problems
(slice admission vs. capacity reservation)."* Or rename before submission
(U2RA / URTA / UnRes are free).

## 2. Figure inventory (PNG, 150 dpi, paper-ready)

| File | Suggested paper role |
|---|---|
| fig_demand_vs_reservation.png | Fig. 4: main qualitative result |
| fig_quantile_band.png | Fig. 3: Stage 1 uncertainty visualization |
| fig_kappa_sweep.png | Fig. 5: risk-utility frontier |
| fig_drift_demo.png | Fig. 6: Stage 5 A/B (two panels) |
| fig_cttc_slices.png | Fig. 7: per-slice policy comparison |
| fig_self_evolution.png | appendix or merge into Fig. 6 |
| history.png / audit_demand.png | appendix: training health / data overview |

## 3. Table inventory

| Table | Build from |
|---|---|
| T1 datasets | doc 07 sections 2-3 |
| T2 main RAN results | ucra_results.json (doc 09 section 1) |
| T3 drift demo | drift_demo.json (doc 09 section 2) |
| T4 CTTC per-slice | cttc_results.json (doc 09 section 3) |
| T5 kappa sweep | --sweep + cttc_results.json kappa_sweep (doc 12 section 4) |
| T6 ablations | doc 12 section 5 (config-only runs) |

## 4. Limitations to state (reviewers always ask)

1. **Capacity is a heuristic** (1.3 x observed peak) -- the PM dataset does
   not publish physical capacity; all normalized metrics share it, rankings
   are unaffected (doc 07).
2. **Single network segment / 63-hour test window** on the RAN side;
   generalization across sites is future work (doc 12 section 9 gives the
   extension path).
3. **CTTC snapshots are i.i.d.**, not a time series -- Stage 2-3 evidence
   there is distributional, not temporal; the rho term in the CTTC eval
   uses current offered load as a stress proxy (doc 09 section 3 caveat +
   the train-only variant recipe in doc 12 section 7).
4. **Quantile crossing is not enforced** structurally (doc 03 section 8).
5. **Stage 5 is trigger-based fine-tuning**, not a formal drift detector
   with guarantees; the demo is a controlled A/B, not a field study.
6. **Single-step reservations** (horizon step 0) -- multi-step lookahead is
   scaffolded (horizon=4) but unused.

## 5. Before you submit (repo polish checklist)

- [ ] Tag the exact commit used for the paper: `git tag v1.0-paper && git push --tags`
- [ ] Pin requirements (torch version you validated on) in requirements.txt
- [ ] Archive the repo on Zenodo (get a DOI; cite it in the paper)
- [ ] Fill the full author lists in the two BibTeX entries (doc 07 section 6)
- [ ] Decide the acronym (keep UCRA + collision sentence, or rename)
- [ ] Run the 3-seed variance pass and put mean +- sd in T2/T3 (doc 12
      section 8)
- [ ] Add the "Phi (train-only)" fourth policy to T4 (doc 12 section 7)
'''

FILES['15_glossary.md'] = r'''# 15 -- Glossary (Plain Language, No Jargon Left Behind)

**URLLC / eMBB / mMTC** -- the three 5G slice families: ultra-reliable
low-latency (factory robots, remote surgery), enhanced mobile broadband
(video, browsing), massive machine-type communication (millions of tiny
sensors). Strictness order: URLLC > eMBB > mMTC.

**RAN** -- radio access network: base stations and their sectors, the part
of the network that talks to phones.

**PM counter** -- performance-management statistic a live network reports
every 15 min: data volumes, RB utilization, active users, CQI, ...

**RB / CQI / RRC / MIMO rank** -- resource block (radio spectrum unit);
channel-quality indicator (signal quality 1-15); radio connection state;
number of spatial streams. All appear in the RAN dataset columns.

**Sector** -- one antenna beam of a base station. Dataset_01: 75 of them.

**DL / UL** -- downlink / uplink traffic.

**Campaign gap** -- the multi-day hole between the dataset's observation
periods (Mar vs Oct 2023). Never model across it (doc 07).

**Slot** -- one 15-min decision step. The unit of everything in UCRA.

**Reservation R_t** -- capacity UCRA commits for slot t before demand is
realized. The central quantity of the whole project.

**Offered vs realized demand** -- CTTC: the traffic the flows ask for;
RAN: the data volume actually observed in the slot.

**delta (CTTC)** -- the slice's reserved share (0..1) in the dataset;
`reserved = delta x link_cap`.

**link_cap (CTTC)** -- mean edge `bandwidth` across the sample's topology
graph; our stand-in for slice capacity.

**Quantile / q_tau** -- the value below which tau of the probability mass
lies. q90 of demand = "level that demand stays under 90% of the time".

**Pinball loss** -- the training loss whose minimizer IS the tau-quantile
(doc 03 section 5). Asymmetric: under-prediction costs tau, over-prediction
costs 1-tau.

**Coverage** -- fraction of realized values inside the predicted
[q05, q99] band. Calibration check for Stage 1 (nominal 94%).

**Quantile crossing** -- when predicted q75 > q99 etc. (heads are
independent). Guarded, not structurally forbidden (doc 03 section 8).

**Z-space / z-score** -- subtract mean, divide by std. Two scalers: one
for inputs, one (scY) for targets. Training in z-space is what un-stalls
pinball learning (doc 03 section 3).

**Leakage** -- any future/test information sneaking into training. The
split is chronological and scalers are fit on train only, so there is
none (except the deliberately-cheating oracle).

**u_hat, U_t (spread), rho_t** -- the Stage 1 triple: point forecast
(median), uncertainty width (q99 - q50), and risk indicator from recent
violations + drift (doc 03 section 10).

**kappa** -- risk aversion: how many units of reservation per unit of
uncertainty spread. The frontier dial (doc 04 section 3).

**tau (base_tau)** -- which quantile Phi anchors on (0.9).

**Phi** -- the transformation (u_hat, U_t, rho_t) -> R_target with floor
and ceiling. The "U-to-R" in UCRA.

**Headroom / floor** -- min_headroom: R never drops below 1.05 x point
forecast.

**Ceiling (max_reserve_ratio)** -- R never exceeds 95% of capacity.

**EWMA** -- exponentially weighted moving average; Stage 4's smoother
(alpha 0.3: each step moves 30% toward the target).

**Hysteresis (release)** -- reservations shrink only when >10% of capacity
stayed idle AND the target is lower; and never below what was actually
used. Prevents oscillation (doc 04 section 4).

**Best-effort pool** -- 10% of R_t kept out of per-entity guarantees.

**Best-effort share / pressure exponent / risk exponent** -- Stage 3
weights: proportional-to-demand with an optional risk premium
(doc 05 section 2).

**Reservoir sampling** -- equal inclusion probability for every past item
in a fixed-size buffer; the anti-forgetting memory of Stage 5 (doc 06
section 3).

**Catastrophic forgetting** -- fine-tuning on only-new data erasing old
knowledge; the replay buffer prevents it.

**Drift / drift z-score** -- the world's statistics changing / how many
training-sigmas the recent demand mean is away from the training mean.

**Trigger** -- Stage 5 fires if rolling violations > 15% OR |z| > 3
(doc 06 section 2).

**Fine-tune** -- 3 gentle epochs (lr x 0.3) on the replay buffer.

**Oracle** -- a baseline that cheats (sees the test window) to define the
achievable envelope (doc 08 section 3).

**Frontier / knee** -- the (violation, utilization) trade-off curve and
its elbow: the cheapest kappa with ~0 violations.

**Capacity C** -- 1.3 x observed peak demand (heuristic stand-in for the
physical limit, doc 07 honesty box).

**Reservation error / utilization / waste / violation rate** -- the four
metrics (doc 08 section 1).

**md5 / HTTP Range** -- checksum for download verification / the header
that enables resuming partial downloads.

**CC-BY-4.0** -- the datasets' license: use freely WITH attribution (doc
07 section 6).

**Zenodo DOI** -- the persistent identifier of each dataset (10.5281/
zenodo.17815388 RAN, 10.5281/zenodo.10610616 CTTC).

**datanetAPI** -- the CTTC dataset's own Python reader; our loader wraps
it dict-tolerantly (doc 07 section 3).

**jsonpickle** -- the package datanetAPI.py imports; auto-installed by
the downloader.

**PPBP-flavoured bursts** -- Poisson Pareto burst process style:
sudden ON/OFF traffic surges with heavy-tailed lengths; what the
synthetic generator mimics.

**Checkpoint (model.pt)** -- saved network weights + architecture config;
loaded by run_ucra/demo/evolution so training happens once.

**Early stopping / patience** -- stop training when validation loss stops
improving for `patience` epochs; restore the best epoch.
'''

README_ANCHOR = '## Repo layout'
README_INSERT = r'''## Documentation

Every module, equation, result and design decision is explained in
[`docs/`](docs/) -- 16 guides with suggested reading orders in
**[docs/README.md](docs/README.md)**.

| Start with | If you want to ... |
|---|---|
| `docs/01_big_picture.md` | understand UCRA in plain English |
| `docs/02_pipeline_walkthrough.md` | know what every command and output file does |
| `docs/09_results_interpretation.md` | understand the printed numbers |
| `docs/13_faq_troubleshooting.md` | fix an error |

'''

def patch_readme() -> None:
    readme = Path("README.md")
    if not readme.exists():
        print("  [skip] README.md not found (no index section added)")
        return
    if "docs/01_big_picture.md" in readme.read_text(encoding="utf-8"):
        print("  [skip] README.md already links the docs index")
        return
    text = readme.read_text(encoding="utf-8").replace(_TOK, _BT)
    ins = README_INSERT.replace(_TOK, _BT)
    if README_ANCHOR in text:
        readme.write_text(text.replace(README_ANCHOR, ins + README_ANCHOR, 1),
                          encoding="utf-8")
        print("  patched README.md (## Documentation section added)")
    else:
        readme.write_text(text.rstrip("\n") + "\n\n" + ins, encoding="utf-8")
        print("  appended README.md (## Documentation section added)")

EXPECTED_MD5 = {
    'README.md': '897a54ba477690649978931db561378e',
    '01_big_picture.md': 'c806d9f6b5f62f6f98f2e52f2acfc59f',
    '02_pipeline_walkthrough.md': 'faae963c3b86dbe6bf4e8cadf778f962',
    '03_stage1_uncertainty.md': '243f1ccf2c296887d3dfa2ffdafe1afc',
    '04_stage2_phi_stage4_update.md': '9207dff63f5f3a15e8d68e06ff38aeb0',
    '05_stage3_allocation.md': 'f85497b779351d0a26fddf5b9b0c7baf',
    '06_stage5_self_evolution.md': 'b218f45edfc1c4552d6aabde66dc29cd',
    '07_data_sources.md': 'dbb5536c6b77c76644d1af5ee1543735',
    '08_metrics_baselines.md': '9897c5bae007c3596692a90b9298227c',
    '09_results_interpretation.md': '98a1b2c8df8cd80f8c6aa8e77cd4e417',
    '10_code_reference.md': '327d3d184b3f25235fba2c87a6855779',
    '11_config_guide.md': '58daa185426683fc447cb68dc5ed8cbc',
    '12_experiments_guide.md': 'f07164716a5c1dda344f2ac4a06968e1',
    '13_faq_troubleshooting.md': '77bfab27af055775fed80168cdd911d4',
    '14_paper_map.md': '56f99ae11f5fabc0080b48b55afe998a',
    '15_glossary.md': 'df93c6f40fbcc83ead9b06483ffe08d0',
}

def verify_all() -> None:
    print("-" * 66)
    print("  verifying all 16 docs against reference checksums ...")
    bad = 0
    for name in sorted(EXPECTED_MD5):
        want = EXPECTED_MD5[name]
        p = Path("docs") / name
        if not p.exists():
            print(f"  [MISSING ] docs/{name}  (run the earlier part(s) first)")
            bad += 1
            continue
        got = hashlib.md5(p.read_bytes()).hexdigest()
        if got != want:
            print(f"  [FAIL    ] docs/{name}  md5 {got[:8]} (want {want[:8]})")
            bad += 1
        else:
            print(f"  [ok      ] docs/{name}")
    if bad:
        print(f"  {bad} problem(s) -- send me the lines marked FAIL/MISSING")
        sys.exit(1)
    print("  ALL 16 DOCS VERIFIED -- byte-identical to the tested reference")


def write_all() -> None:
    dest = Path("docs")
    dest.mkdir(parents=True, exist_ok=True)
    print("=" * 66)
    print(f" UCRA docs generator -- part {4}/4")
    print("=" * 66)
    for name in FILES:
        text = FILES[name].replace(_TOK, _BT)
        data = text.encode("utf-8")          # raw bytes: LF kept, no CRLF surprises
        (dest / name).write_bytes(data)
        print(f"  [ok] docs/{name:<34} {len(data):>7,} bytes  md5 {hashlib.md5(data).hexdigest()[:8]}")
    print(f"  part {4}/4 done: {len(FILES)} file(s) written")


if __name__ == "__main__":
    write_all()
    patch_readme()
    verify_all()
    print("-" * 66)
    print("  DONE. Docs are complete and verified. Final steps:")
    print("    del create_ucra_docs.py")
    print("    git add -A")
    print('    git commit -m "docs: complete 16-guide documentation set"')
    print("    git push")