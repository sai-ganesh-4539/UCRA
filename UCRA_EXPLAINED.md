# UCRA — Your Project, Explained

> **One line:** UCRA predicts 5G network demand *with uncertainty*, converts that
> uncertainty into safe capacity reservations, and adapts itself when traffic shifts —
> validated on a real commercial network.
>
> **10-second version:** predict range → reserve capacity → 0% violations at 71.7%
> utilization → self-heal under drift.

---

## 1. The problem

Network operators reserve radio capacity for slices (URLLC / eMBB / mMTC).
Reserve too little → demand exceeds reservation → **violations** (SLA breach).
Reserve too much → wasted capacity (money). Existing methods do one or the other.
UCRA does both: near-zero violations **and** low waste, by sizing reservations
from *forecast uncertainty* instead of a flat forecast.

## 2. The 5-stage pipeline

```
 demand history ─► [1] Quantile LSTM ─► uncertainty triple (q50, spread, rho)
                        │
                        ▼
                   [2] Phi transform      R_target = q90 + kappa * spread
                        │
                        ▼
                   [3] Risk allocation    guardrails: floor / ceiling
                        │
                        ▼
                   [4] Update & release   EWMA smoothing, per 15-min slot
                        │
                        ▼                   drift detected? violations rising?
                   [5] Self-evolution  ◄──── fine-tune model, repeat
```

Each stage = one module in `ucra/` — the code mirrors the framework diagram 1:1.

## 3. Datasets (both open, CC-BY-4.0)

| # | Dataset | Source | Role |
|---|---------|--------|------|
| 1 | **Live 5G/4G/2G RAN PM counters** | Zenodo 17815388 (STU Bratislava + Ericsson Slovakia, Nature Sci Data 2026) | **Primary.** Real commercial network, 75 sectors, 15-min counters → Stages 1, 4, 5 |
| 2 | **CTTC B5G Network Slicing** | Zenodo 10610616 (Data in Brief 2024) | Simulated slice-level demands (URLLC/eMBB/mMTC) + reservations → Stages 2–3 |

Data is never committed (1+ GB) — `scripts/download_data.py` re-downloads with
baked md5 checksums, so anyone reproduces the exact same bytes.

## 4. Feature engineering (what happens to raw data)

1. 75 sectors' **data-volume DL** counters → summed into one network demand series (15-min slots).
2. Campaign gap handling: Mar + Oct 2023 recordings have a **193-day gap** → extractor keeps only the **longest gap-free segment** (Oct 6–25, 1,778 slots). Train/test never cross the gap (no leakage).
3. Capacity baseline: **C = 143,954.5** = 1.3 × observed peak demand.
4. Windows: lookback **96 slots (24 h)** → predict **4 slots (1 h) ahead**; 1,679 windows total.
5. z-score scaling on train only (scaler inverted after prediction).

## 5. The model + training

- **Quantile LSTM** — 2 layers × 64 hidden, dropout 0.1, **52,252 params**.
- Outputs **7 quantiles at once**: [0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99].
- Loss: **pinball** (quantile loss) — penalizes under- and over-prediction asymmetrically per quantile.
- Trained in z-space (stabilizes gradients), inverted back to real units after.
- SGD-Adam, lr 0.001, batch 64, early stopping (patience 6, max 50 epochs).
- Split **1,175 / 252 / 252** (train/val/test) — ~3 min on a plain CPU.

## 6. Stages 2–4: how a reservation is actually computed

Worked example from your real test set:

| step | value |
|---|---|
| model median q50 | 79,400 |
| uncertainty spread (q99−q50) | 12,900 |
| q90 ≈ q50 + (0.9−0.5)/0.49 × spread | ≈ 89,930 |
| Phi buffer: κ=0.5 × spread | +6,450 |
| **R_target** | **96,380** |
| EWMA smoothing (α=0.3) | **94,084** (this is what gets reserved) |
| realized demand | 91,200 → utilization 0.969, no violation |
| guardrails | floor 83,370 (headroom ≥ median), ceiling 136,757 (≤ C × max ratio) |

Formula: `R_target = clip( q_tau + κ·spread·(1 + 0.3·rho_t), floor, ceiling )`
— **κ (kappa)** is the risk knob: κ=0 → bare q90 (more violations), κ→1+ → safer, fatter buffer.

## 7. Stage 5: self-evolution

A rolling monitor watches violations. If they cross a trigger threshold
(demand drifted), Stage 5 **fine-tunes** the LSTM on recent data — no full retrain.
Proof it works — your drift demo: **+35% demand surge** injected 15% into the test window:

| policy | post-surge violations | utilization |
|---|---|---|
| frozen model | 26.5% | 82.7% |
| **self-evolving** | **11.6%** | 80.3% |

Stage-5 fired exactly 3 times: slots **[96, 120, 144]**.

## 8. Results (your real runs)

**RAN pipeline (real network, 252 test slots):**

| metric | value |
|---|---|
| quantile coverage | **96.0%** (nominal 90%) |
| violations | **0.0%** |
| utilization | **71.7%** |
| reservation error | **10.8%** |

**Baselines (same test set):**

| policy | violations | verdict |
|---|---|---|
| static_peak (reserve historical max) | 0% but **30.5% waste** | safe, expensive |
| mean_forecast | **50.4%** of slots violate | cheap, broken |
| oracle (knows tomorrow) | 16.3% | lower bound |
| **UCRA** | **0.0% @ 71.7% util** | only policy with both ✓ |

**CTTC multi-slice (120 snapshots, operator vs UCRA):**

| slice | operator | static q90 | **ucra_phi** |
|---|---|---|---|
| URLLC | 29.0% | 15.2% | **1.4%** |
| eMBB | 50.6% | 12.2% | **0.0%** |
| mMTC | 5.4% | 12.2% | **4.4%** |

Kappa sweep (0 → 2.0) traces the full risk-utility frontier in `outputs/fig_kappa_sweep.png`.

## 9. Code map (what file does what)

```
ucra/
  data/       ran_loader.py (real RAN) · cttc_loader.py (CTTC) · synthetic.py
  features/   windowing.py (96-slot windows + scaler)
  models/     quantile_lstm.py (Stage 1)
  core/       transform.py (Stages 2+4: Phi, EWMA, guardrails)
              allocate.py   (Stage 3: risk-adaptive allocation)
              evolve.py     (Stage 5: drift trigger + fine-tune)
  eval/       metrics.py · baselines.py · plots.py
scripts/      download_data.py · audit_data.py · train.py · run_ucra.py
              demo_evolution.py · run_cttc_eval.py · make_figures.py
configs/      default.yaml — every knob in one place
outputs/      committed results: 13 artifacts + README manifest
docs/         16-guide documentation set (see below)
```

## 10. How to run everything

```
pip install -r requirements.txt
python scripts/download_data.py          # Zenodo, md5-verified
python scripts/audit_data.py             # data sanity + histogram
python scripts/train.py                  # ~3 min CPU
python scripts/run_ucra.py --sweep       # main result + kappa frontier
python scripts/demo_evolution.py         # drift demo (Stage 5 proof)
python scripts/run_cttc_eval.py          # multi-slice validation
python scripts/make_figures.py           # refresh figures
```

## 11. Metrics explained (and the confusion-matrix question)

- **Coverage** — % of actual demand inside the predicted [q05, q99] band. 96% vs 90% nominal = honest, slightly conservative.
- **Violations** — % of slots where demand > reservation (the SLA-killer).
- **Utilization** — demand / reservation. High = no wasted capacity.
- **Reservation error** — mean |demand − reservation| / demand.
- **Pinball loss** — the training loss for quantile models.
- **Confusion matrix?** Not applicable — that's for classifiers (yes/no labels).
  UCRA is regression + allocation, so the quality quadruple is
  **coverage / violations / utilization / reservation error**.

## 12. Where everything lives now

- GitHub `sai-ganesh-4539/UCRA` — code + 16 docs + committed results (all pushed).
- `README.md` → results tables + quickstart. `docs/README.md` → index of all guides.
- Deep dives: `docs/03–06` (one per stage) · `docs/07` (datasets) ·
  `docs/09` (how to read every number) · `docs/13` (FAQ/troubleshooting) ·
  `docs/14` (how this maps to a future paper).

**Status: project complete — code, docs, results, reproduction path all published.**
