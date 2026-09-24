#!/usr/bin/env python3
# ============================================================
# UCRA Step 1 scaffold generator
# Usage:  python create_step1_files.py
# Run this INSIDE an empty folder named UCRA (it uses cwd).
# Creates the full repo skeleton + initial files.
# ============================================================
from pathlib import Path

BT3 = "`" * 3  # triple backtick, injected at write time (keeps this script paste-safe)

FILES = {}

# ---------------- README.md ----------------
FILES["README.md"] = """\
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

{{BT3}}
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
{{BT3}}

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

{{BT3}}bash
pip install -r requirements.txt        # torch: CPU build is enough
bash scripts/download_data.sh          # Step 2
python scripts/audit_data.py           # look at the data before modeling
python scripts/train.py  --config configs/default.yaml    # Step 4
python scripts/run_ucra.py --config configs/default.yaml  # Steps 5-6
python scripts/make_figures.py                            # Step 6
{{BT3}}

## Repo layout

{{BT3}}
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
{{BT3}}

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
"""

# ---------------- requirements.txt ----------------
FILES["requirements.txt"] = """\
# UCRA — Uncertainty-to-Reservation Transformation Algorithm
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
matplotlib>=3.7
PyYAML>=6.0
tqdm>=4.65
networkx>=3.0        # CTTC topology objects
scipy>=1.10
# Deep learning (quantile LSTM). CPU build is sufficient:
#   pip install torch --index-url https://download.pytorch.org/whl/cpu
torch>=2.2
# dev / tests
pytest>=8.0
"""

# ---------------- .gitignore ----------------
FILES[".gitignore"] = """\
# --- data (never commit datasets) ---
data/**
!data/**/
!data/.gitkeep

# --- python ---
__pycache__/
*.py[cod]
*.egg-info/
.eggs/
build/
dist/
.venv/
venv/
.env

# --- experiment outputs ---
outputs/
runs/
*.pt
*.pth
*.ckpt

# --- notebooks ---
.ipynb_checkpoints/

# --- os / ide ---
.DS_Store
Thumbs.db
.idea/
.vscode/

# --- logs ---
*.log
wandb/
"""

# ---------------- LICENSE ----------------
FILES["LICENSE"] = """\
MIT License

Copyright (c) 2026 UCRA Project Authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

# ---------------- configs/default.yaml ----------------
FILES["configs/default.yaml"] = """\
# ============================================================
# UCRA default configuration
# Uncertainty-to-Reservation Transformation Algorithm
# for Self-Evolving Risk-Adaptive Resource Allocation
# ============================================================

seed: 42

paths:
  raw_data: data                 # downloaded Zenodo archives live here
  outputs: outputs               # models, metrics, figures

# ------------------------------------------------------------
# Dataset 1: CTTC B5G Network Slicing (Zenodo 10610616)
#   role: slice-level reservation logic (Stages 2-3 + violations)
# Dataset 2: Live 5G/4G/2G RAN PM counters (Zenodo 17815388)
#   role: real temporal demand uncertainty (Stages 1, 4, 5)
# Dataset 3 (optional failsafe): Liverpool 5G HDD
# ------------------------------------------------------------
data:
  dataset: ran                   # "ran" | "cttc" | "synthetic"
  ran:
    # 15-min sector-level PM counters; we aggregate sectors -> network demand
    subset: dataset01            # dataset01 | dataset02 | dataset03
    target: data_volume          # demand series to forecast (data volume)
    agg: sum                     # sector aggregation
    resample: null               # series is already 15-min; null = keep
  cttc:
    samples_dir: data/cttc/slicing-simulations   # unpacked archive
    slices: [eMBB, mMTC, URLLC]
  synthetic:
    n_steps: 2016                # 3 weeks @ 15-min
    n_sectors: 8

# ------------------------------------------------------------
# Stage 1 - Uncertainty Estimation (quantile LSTM)
# ------------------------------------------------------------
model:
  name: quantile_lstm
  seq_len: 96                    # input window: 24 h @ 15-min
  horizon: 4                     # predict 1 h ahead @ 15-min steps
  hidden_size: 64
  num_layers: 2
  dropout: 0.1
  quantiles: [0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
  batch_size: 64
  lr: 0.001
  max_epochs: 50
  patience: 6                    # early stopping
  train_val_test: [0.7, 0.15, 0.15]

# ------------------------------------------------------------
# Stage 2 - Uncertainty-to-Reservation Transformation  Phi
#   R_t = clip( q_tau_hat + kappa * spread_t * (1 + rho_weight * rho_t),
#               0, capacity )
# ------------------------------------------------------------
phi:
  base_tau: 0.9                  # base quantile the reservation follows
  kappa: 0.5                     # risk scaling of the uncertainty spread
  rho_weight: 0.3                # weight of the risk indicator rho_t
  min_headroom: 0.05             # keep >= 5% headroom above point forecast
  max_reserve_ratio: 0.95        # never reserve more than 95% of capacity

# ------------------------------------------------------------
# Stage 3 - Risk-Adaptive Allocation
# ------------------------------------------------------------
allocation:
  pressure_exponent: 1.0         # demand-pressure weighting
  risk_exponent: 0.5             # risk weighting per slice/sector
  best_effort_share: 0.1         # capacity kept out of reservations

# ------------------------------------------------------------
# Stage 4 - Reservation Update & Release
# ------------------------------------------------------------
update:
  ewma_alpha: 0.3                # smoothing of reservation adjustments
  release_hysteresis: 0.1        # release headroom only if >10% unused
  drift_window: 96               # rolling window for drift detection

# ------------------------------------------------------------
# Stage 5 - Self-Evolving Learning
# ------------------------------------------------------------
evolution:
  violation_trigger: 0.15        # fine-tune if rolling violation rate > 15%
  drift_zscore: 3.0              # or if demand z-score exceeds 3
  replay_size: 512               # replay buffer for continual learning
  finetune_epochs: 3
  eval_every: 96                 # check evolution triggers every 24 h

# ------------------------------------------------------------
# Evaluation (Feedback & Evaluation layer)
# ------------------------------------------------------------
eval:
  metrics: [reservation_error, violation_rate, utilization, waste]
  baselines: [static_peak, mean_forecast, oracle_quantile]
  sweep_kappa: [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]
"""

# ---------------- docs/METHODOLOGY.md ----------------
FILES["docs/METHODOLOGY.md"] = """\
# UCRA Methodology — Diagram Blocks <-> Code Modules

> Filled progressively per implementation step. This file is the mapping you
> cite in the paper when reviewers ask "which module implements which block?"

## Stage mapping

| UCRA diagram block | Inputs | Outputs | Code module | Step |
|---|---|---|---|---|
| Network Data (Self-Evolving Network Environment) | Zenodo 17815388 PM counters; Zenodo 10610616 slice samples | demand series D_t, capacity C, topology, slice metadata | ucra/data/ | 2 |
| 1. Uncertainty Estimation | D_t windows | u_t (point forecast), U_t (quantile band), rho_t (risk indicator) | ucra/models/quantile_lstm.py + ucra/core/uncertainty.py | 4 |
| 2. Uncertainty-to-Reservation Phi(U_t, u_t, rho_t) | u_t, U_t, rho_t | R_t reservation | ucra/core/transform.py | 5 |
| 3. Risk-Adaptive Allocation | R_t, per-slice/sector demand | A_t allocation | ucra/core/allocate.py | 5 |
| 4. Reservation Update & Release | R_t, realized D_t, EWMA, hysteresis | R_t+1 | ucra/core/transform.py | 5 |
| 5. Self-Evolving Learning | violation/drift triggers, replay buffer | theta_t+1 model weights | ucra/core/evolve.py | 7 |
| Feedback & Evaluation | A_t vs D_t | reservation error, violation rate (risk exposure), utilization, latency proxy (CTTC AvgDelay) | ucra/eval/ | 6 |

## Phi — core equation (implemented in ucra/core/transform.py)

{{BT3}}
R_t = clip( q_tau_hat + kappa * spread_t * (1 + rho_weight * rho_t), 0, C * max_reserve_ratio )

where
  q_tau_hat(t) = tau-quantile forecast of demand at time t   (base_tau = 0.9)
  spread_t     = q_0.99(t) - q_0.50(t)                       (uncertainty width U_t)
  rho_t        = recent violation rate / drift signal        (risk indicator)
  kappa        = risk aversion (kappa, swept in evaluation)
  rho_weight   = rho_weight
{{BT3}}

## Baselines (defend why UCRA is better)

1. **static_peak** — reserve the historical peak demand (worst-case static).
2. **mean_forecast** — reserve the point forecast only (no uncertainty).
3. **oracle_quantile** — reserve the *true* tau-quantile computed on the test
   window (upper bound; UCRA should approach it without seeing the future).

## Naming-collision handling (related work)

The acronym UCRA was used in 2020 by "User-Centric Context-Aware Resource
Allocation for Network Slicing" (IEEE Access 8, DOI 10.1109/ACCESS.2020.3046198).
Required sentence for the paper: "Unlike the user-centric context-aware
resource allocation of [2020], UCRA here denotes an uncertainty-driven
reservation transformation with self-evolving risk adaptation; the two
address complementary problems (slice admission vs. capacity reservation)."
Alternatively rename the algorithm (U2RA / URTA / UnRes) before submission.
"""

# ---------------- package inits ----------------
FILES["ucra/__init__.py"] = '''\
"""UCRA - Uncertainty-to-Reservation Transformation Algorithm.

Pipeline stages (see docs/METHODOLOGY.md):
  1. Uncertainty Estimation            -> ucra.models.quantile_lstm
  2. Uncertainty-to-Reservation (Phi)  -> ucra.core.transform
  3. Risk-Adaptive Allocation          -> ucra.core.allocate
  4. Reservation Update & Release      -> ucra.core.transform
  5. Self-Evolving Learning            -> ucra.core.evolve
"""

__version__ = "0.1.0"
'''
for _pkg in ("data", "features", "models", "core", "eval"):
    FILES[f"ucra/{_pkg}/__init__.py"] = f'"""UCRA.{_pkg} package (built in upcoming steps)."""\n'
FILES["tests/__init__.py"] = '"""UCRA test suite."""\n'


def main() -> None:
    root = Path.cwd()
    # directories (incl. empty ones that git tracks via .gitkeep)
    for d in ("configs", "docs", "scripts", "tests", "data/cttc", "data/ran",
              "ucra/data", "ucra/features", "ucra/models", "ucra/core", "ucra/eval"):
        (root / d).mkdir(parents=True, exist_ok=True)
    (root / "data" / ".gitkeep").touch()

    for rel, content in FILES.items():
        text = content.replace("{{BT3}}", BT3)
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        print(f"  wrote {rel}")

    print("\nStep 1 scaffold complete. Next commands:")
    print("  git init")
    print('  git add . && git commit -m "Step 1: repo scaffold"')
    print("  git branch -M main")
    print("  git remote add origin https://github.com/<YOUR-USERNAME>/UCRA.git")
    print("  git push -u origin main")


if __name__ == "__main__":
    main()