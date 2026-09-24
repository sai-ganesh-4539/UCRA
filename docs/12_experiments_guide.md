# 12 -- Experiments Guide: Reproduce, Ablate, Extend

## 1. Reproduction checklist (paper numbers)

From a clean clone:

```
pip install -r requirements.txt
python scripts/download_data.py                     # RAN, 8 MB
python scripts/audit_data.py   --config configs/default.yaml
python scripts/train.py        --config configs/default.yaml
python scripts/run_ucra.py     --config configs/default.yaml --sweep
python scripts/demo_evolution.py --config configs/default.yaml
python scripts/download_data.py --cttc              # 271 MB, one-time
python scripts/run_cttc_eval.py --config configs/default.yaml --max-samples 120
python -m pytest tests/test_smoke.py -q
```

Expected (your reference machine): 302,052 rows / 75 sectors; segment 1,778
slots; capacity 143,954.5; 52,252 params; coverage ~93-96%; UCRA 0.0% viol /
71.7% util / 10.8% res-err; demo updates [96, 120, 144]; CTTC URLLC 1.4% /
eMBB 0.0% / mMTC 4.4%. Tolerances: coverage +-3 points and small metric
drift across torch versions/platforms (doc 03 section 7). Everything else
(Stages 2-4, baselines math) is deterministic given pred_q.

## 2. Grid: drift severity (one table for the paper)

```
for s in 0.15 0.25 0.35 0.50:
    python scripts/demo_evolution.py --config configs/default.yaml --shift {s}
```
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

```python
def phi_empirical_clean(hist, phi_cfg):
    q50  = float(np.quantile(hist, 0.5)); q_tau = float(np.quantile(hist, phi_cfg["base_tau"]))
    q99  = float(np.quantile(hist, 0.99)); spread = max(q99 - q50, 1e-9)
    return q_tau + phi_cfg["kappa"] * spread          # rho = 0, train-only
```

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
