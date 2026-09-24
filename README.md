# UCRA - Uncertainty-to-Reservation Transformation Algorithm
### Self-Evolving Risk-Adaptive Resource Allocation for 5G/B5G Network Slicing

> **Naming note:** a 2020 IEEE paper (User-Centric Context-Aware Resource
> Allocation, DOI 10.1109/ACCESS.2020.3046198) already uses the acronym "UCRA"
> in the same domain. Cite and differentiate it in your related-work section
> (see `docs/METHODOLOGY.md`), or rename the algorithm (suggestions: U2RA,
> URTA, UnRes).

UCRA is a closed-loop resource-allocation pipeline that turns **demand
uncertainty** into **capacity reservations**, allocates resources under risk,
and **self-evolves** as the network drifts.

```
Network Data ──► 1. Uncertainty Estimation ──► 2. Φ: Uncertainty→Reservation
(traffic, capacity,             │                        │
 topology, risk)                │                        ▼
        ▲                       │             3. Risk-Adaptive Allocation
        │                       ▼                        │
        │              5. Self-Evolving Learning ◄────────┤
        │                       │                        ▼
        └────────── 4. Reservation Update & Release ◄────┘
                     Feedback: reservation error, violations,
                     utilization, latency / reliability
```

## Datasets

| # | Dataset | Source | License | Role in UCRA |
|---|---------|--------|---------|--------------|
| 1 | **CTTC B5G Network Slicing** (271 MB) | [Zenodo 10610616](https://zenodo.org/records/10610616) - *Data in Brief* 55:110738, 2024 | CC-BY-4.0 | Slice-level reservations (eMBB/mMTC/URLLC), topology, routing, packet drops and delay percentiles → **Stages 2-3 + violation/latency feedback** |
| 2 | **Live 5G/4G/2G RAN PM Counters** (~767 MB, real commercial network) | [Zenodo 17815388](https://zenodo.org/records/17815388) - *Scientific Data*, 2026, DOI 10.1038/s41597-026-07723-0 | CC-BY-4.0 | Real 15-min sector-level demand series (volumes, utilisation, CQI, users, energy) → **Stages 1, 4, 5 + utilization feedback** |
| 3 | *(optional failsafe)* Liverpool 5G High-Density Demand | [opendata.ljmu.ac.uk](https://opendata.ljmu.ac.uk) - *Scientific Data*, 2025 | open | 1-3 s user-level bursts, SINR/PRB/BLER → stress scenarios |

Datasets are **not** committed to this repo - fetch them with
`python scripts/download_data.py` (add `--cttc` for dataset 1, `--ran-all`
for every RAN subset).

## Quickstart

```bash
pip install -r requirements.txt                  # torch CPU build is enough
python scripts/download_data.py                  # real RAN data (8.3 MB)
python scripts/audit_data.py  --config configs/default.yaml
python scripts/train.py       --config configs/default.yaml
python scripts/run_ucra.py    --config configs/default.yaml --sweep
python scripts/make_figures.py --config configs/default.yaml
```

Extra experiments:

```bash
python scripts/demo_evolution.py                 # Stage-5 drift demo (surge + self-healing)
python scripts/download_data.py --cttc           # 271 MB, slice-level dataset
python scripts/run_cttc_eval.py                  # Stage 2-3 on CTTC slices
```

## Results (real network data - reproducible with the commands above)

RAN PM counters (Zenodo 17815388, Dataset_01): 75 sectors, 4G+5G, 15-min
slots; longest gap-free segment 1,778 slots (Oct 6-25, 2023); capacity
estimate ~143,955. Quantile-LSTM Stage 1: 52,252 params, test pinball 930.3,
coverage 96.0% at the 90% nominal level, CPU-trained in ~3 minutes.

| Policy | Reservation error | Violation rate | Utilization |
|---|---|---|---|
| **UCRA (kappa = 0.5)** | 10.8% | **0.0%** | **71.7%** |
| static_peak | 30.5% | 0.0% | 54.8% |
| mean_forecast | 2.0% | 50.4% | 95.3% |
| oracle_quantile | 20.1% | 16.3% | 64.9% |

UCRA is the only policy that simultaneously holds zero violations **and**
high utilization: static over-reservation wastes ~30% of capacity, and naive
mean forecasting violates half of all slots. The kappa sweep
(`outputs/fig_kappa_sweep.png`) traces the full risk-utility frontier.

### Stage 2-3 on CTTC slices (Zenodo 10610616, 120 snapshots, 70/30 split)

Per-slice-type empirical offered-load quantiles feed the same Phi transform;
policies are compared on held-out snapshots (`scripts/run_cttc_eval.py`):

| Slice | Operator (static delta) | Static q90 | UCRA (Phi) |
|---|---|---|---|
| URLLC | 29.0% violations | 15.2% | **1.4%** |
| eMBB | 50.6% violations | 12.2% | **0.0%** |
| mMTC | 5.4% violations | 12.2% | **4.4%** |

Static delta sizing misses URLLC/eMBB demand spikes; the risk-aware
transform holds near-zero violations in exchange for over-provisioning
(see `outputs/fig_cttc_slices.png` and the kappa sweep inside
`outputs/cttc_results.json`).

## Documentation

Every module, equation, result and design decision is explained in
[`docs/`](docs/) -- 16 guides with suggested reading orders in
**[docs/README.md](docs/README.md)**.

| Start with | If you want to ... |
|---|---|
| `docs/01_big_picture.md` | understand UCRA in plain English |
| `docs/02_pipeline_walkthrough.md` | know what every command and output file does |
| `docs/09_results_interpretation.md` | understand the printed numbers |
| `docs/13_faq_troubleshooting.md` | fix an error |

## Repo layout

```
UCRA/
├── configs/default.yaml     # all knobs (quantiles, Φ params, evolution triggers)
├── ucra/
│   ├── data/                # Zenodo loaders + synthetic generator
│   ├── features/            # windowing, splits, scaling
│   ├── models/              # quantile LSTM (pinball loss)
│   ├── core/                # uncertainty, Φ transform, allocation, evolution
│   └── eval/                # metrics, baselines, plots
├── scripts/                 # download / audit / train / run / demo / figures
├── tests/                   # smoke tests (synthetic, CPU-fast)
└── docs/METHODOLOGY.md      # diagram blocks ↔ code modules mapping
```

## Status

- [x] Step 1 - repo scaffold
- [x] Step 2 - data layer (downloader, loaders, audit)
- [x] Step 3 - feature pipeline
- [x] Step 4 - quantile LSTM uncertainty model
- [x] Step 5 - Φ transformation + risk-adaptive allocation
- [x] Step 6 - evaluation + baselines + figures
- [x] Step 7 - self-evolving loop (+ `scripts/demo_evolution.py`)
- [x] Step 8 - full real-data run (results above)
- [x] Stage 2-3 slice-level eval on CTTC (`scripts/run_cttc_eval.py`)

## License

MIT - see [LICENSE](LICENSE). Dataset licenses remain CC-BY-4.0 (cite the
anchor papers linked above when you publish).
