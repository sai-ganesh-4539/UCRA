# UCRA — Uncertainty-to-Reservation Transformation Algorithm
### Self-Evolving Risk-Adaptive Resource Allocation for 5G/B5G Network Slicing

> **Naming note:** a 2020 IEEE paper (User-Centric Context-Aware Resource
> Allocation, DOI 10.1109/ACCESS.2020.3046198) already uses the acronym "UCRA"
> in the same domain. Cite and differentiate it in your related-work section
> (see docs/METHODOLOGY.md), or rename the algorithm (suggestions: U2RA,
> URTA, UnRes).

UCRA is a closed-loop resource-allocation pipeline that turns **demand
uncertainty** into **capacity reservations**, allocates resources under risk,
and **self-evolves** as the network drifts.

```
Network Data --> 1. Uncertainty Estimation --> 2. Phi: Uncertainty->Reservation
(traffic, capacity,             |                        |
 topology, risk)                |                        v
        ^                       |             3. Risk-Adaptive Allocation
        |                       v                        |
        |              5. Self-Evolving Learning <---------+
        |                       |                        v
        +---------- 4. Reservation Update & Release <-----+
                     Feedback: reservation error, violations,
                     utilization, latency / reliability
```

## Datasets

| # | Dataset | Source | License | Role in UCRA |
|---|---------|--------|---------|--------------|
| 1 | **CTTC B5G Network Slicing** (271 MB) | Zenodo 10610616 — Data in Brief 55:110738, 2024 | CC-BY-4.0 | Slice-level reservations (eMBB/mMTC/URLLC), topology, routing, packet drops and delay percentiles -> Stages 2-3 + violation/latency feedback |
| 2 | **Live 5G/4G/2G RAN PM Counters** (~767 MB, real commercial network) | Zenodo 17815388 — Scientific Data, 2026, DOI 10.1038/s41597-026-07723-0 | CC-BY-4.0 | Real 15-min sector-level demand series (volumes, utilisation, CQI, users, energy) -> Stages 1, 4, 5 + utilization feedback |
| 3 | *(optional failsafe)* Liverpool 5G High-Density Demand | opendata.ljmu.ac.uk — Scientific Data, 2025 | open | 1-3 s user-level bursts, SINR/PRB/BLER -> stress scenarios |

Dataset links: https://zenodo.org/records/10610616 and https://zenodo.org/records/17815388
Datasets are **not** committed to this repo — fetch them with
`bash scripts/download_data.sh` (Step 2).

## Quickstart

```bash
pip install -r requirements.txt        # torch: CPU build is enough
bash scripts/download_data.sh          # Step 2
python scripts/audit_data.py           # look at the data before modeling
python scripts/train.py  --config configs/default.yaml    # Step 4
python scripts/run_ucra.py --config configs/default.yaml  # Steps 5-6
python scripts/make_figures.py                            # Step 6
```

## Repo layout

```
UCRA/
+-- configs/default.yaml     # all knobs (quantiles, Phi params, evolution triggers)
+-- ucra/
|   +-- data/                # Zenodo loaders + synthetic generator
|   +-- features/            # windowing, splits, scaling
|   +-- models/              # quantile LSTM (pinball loss)
|   +-- core/                # uncertainty, Phi transform, allocation, evolution
|   +-- eval/                # metrics, baselines, plots
+-- scripts/                 # download / audit / train / run / figures
+-- tests/                   # smoke tests (synthetic, CPU-fast)
+-- docs/METHODOLOGY.md      # diagram blocks <-> code modules mapping
```

## Status

- [x] Step 1 — repo scaffold
- [ ] Step 2 — data layer (download script, loaders, audit)
- [ ] Step 3 — feature pipeline
- [ ] Step 4 — quantile LSTM uncertainty model
- [ ] Step 5 — Phi transformation + risk-adaptive allocation
- [ ] Step 6 — evaluation + baselines + figures
- [ ] Step 7 — self-evolving loop
- [ ] Step 8 — full real-data run

## License

MIT — see LICENSE. Dataset licenses remain CC-BY-4.0 (cite the anchor papers
linked above when you publish).
