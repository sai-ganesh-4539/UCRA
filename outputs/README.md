# outputs/ - experiment artifacts (committed results)

This folder holds everything the UCRA pipeline produces. The small result
artifacts (JSON / CSV / PNG) ARE committed so that every number quoted in the
root README and in docs/ can be inspected without re-running anything.
Binary checkpoints and the raw datasets are NOT committed - both are exactly
reproducible (see the bottom section).

## Artifact map

### RAN pipeline - Zenodo 17815388 (live commercial network, 15-min PM counters)

| file | written by | content |
|---|---|---|
| audit_demand.png | scripts/audit_data.py | demand series + histogram with capacity line |
| history.png | scripts/train.py | training pinball-loss curves |
| train_metrics.json | scripts/train.py | final loss, epochs, parameter count (52,252) |
| ucra_results.json | scripts/run_ucra.py | headline metrics: coverage, violations, utilization, reservation error + all baselines + kappa sweep |
| ucra_log.csv | scripts/run_ucra.py | one row per test slot (252): ts, demand, R_ucra, R_static, violation flags |
| fig_demand_vs_reservation.png | scripts/run_ucra.py | demand vs UCRA reservation vs static reservation |
| fig_quantile_band.png | scripts/run_ucra.py | quantile fan chart (q05-q99) vs realized demand |
| fig_self_evolution.png | scripts/run_ucra.py | rolling violations + Stage-5 update points |
| fig_kappa_sweep.png | scripts/run_ucra.py --sweep | risk-utility frontier (kappa 0 to 2.0) |

### Drift demo + CTTC pipeline - Zenodo 10610616

| file | written by | content |
|---|---|---|
| fig_drift_demo.png | scripts/demo_evolution.py | frozen vs self-evolving policy under +35% demand surge |
| drift_demo.json | scripts/demo_evolution.py | post-surge violations/utilization for both policies + Stage-5 update slots |
| cttc_results.json | scripts/run_cttc_eval.py | per-slice policy comparison (operator / static_q90 / ucra_phi) + kappa sweep + QoS-gap fields |
| fig_cttc_slices.png | scripts/run_cttc_eval.py | violation + utilization bars for all three policies, per slice |

### Not committed (on purpose)

| path | why | how to get it |
|---|---|---|
| model.pt | binary checkpoint | python scripts/train.py (about 3 min on CPU) |
| scaler.npz | binary scaler state | written automatically by train.py / run_ucra.py |
| data/** | 1+ GB of CC-BY datasets | python scripts/download_data.py (md5-verified) |

## Reference numbers (the ones quoted in the root README and docs/)

RAN (real commercial network, 1,778-slot segment, 252 test slots):
- UCRA: 0.0% violations at 71.7% utilization, reservation error 10.8%
- static_peak: 30.5% over-provisioning waste
- mean_forecast: 50.4% of slots in violation
- oracle: 16.3% violations
- quantile coverage 96.0% at the 90% nominal level

Drift demo (+35% surge, Stage-5 self-evolution active):
- post-surge violations: frozen 26.5% -> self-evolving 11.6%
- Stage-5 update slots: [96, 120, 144]

CTTC multi-slice (violations per slice type):
- URLLC: operator 29.0% -> ucra_phi 1.4%
- eMBB:  operator 50.6% -> ucra_phi 0.0%
- mMTC:  operator 5.4%  -> ucra_phi 4.4%

Note: torch results can differ by a fraction of a percent across CPU and
platform combinations (documented in docs/13_faq_troubleshooting.md). The
numbers above are the published reference set.
