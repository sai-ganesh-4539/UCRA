# DEEP_PIPELINE.md — the 5-stage loop, function by function

Companion to `README.md`. This file explains **exactly** how UCRA works:
every formula, every tensor shape, every config knob, and a worked example
with the real numbers from your trained run. File references point to the
actual code. Read top-to-bottom once and the whole system is yours.

```
            +--------------------------------------------------------------+
            |                        FEEDBACK LOOP                          |
            |   violations, realized demand, drift stats                    |
            v                                                               |
 [raw slots] -> [Stage 1] -> (u_hat, spread, rho_t) -> [Stage 2] -> R_target
                 quantile                                            |
                 LSTM                                                v
                                  [Stage 4] EWMA + hysteresis <- [Stage 3]
                        |                                          allocate
                        v
                 R_{t+1} applied -> realized demand -> metrics -> back to top
                                  [Stage 5] drift trigger -> fine-tune Stage 1
```

| stage | module | core functions |
|---|---|---|
| 1 | `ucra/models/quantile_lstm.py`, `ucra/core/uncertainty.py` | `QuantileLSTM`, `make_loss`, `train_model`, `predict_quantiles`, `extract_uncertainty` |
| 2 | `ucra/core/transform.py` | `phi_transform` |
| 3 | `ucra/core/allocate.py` | `risk_adaptive_allocate` |
| 4 | `ucra/core/transform.py` | `update_reservation` |
| 5 | `ucra/core/evolve.py` | `DriftMonitor`, `ReplayBuffer`, `EvolutionEngine` |
| loop | `scripts/run_ucra.py` | `main()` — wires stages 1-5 over the test window |

---

## Stage 1 — Uncertainty Estimation

### 1.1 Architecture (exact)

```python
nn.LSTM(input_size=1, hidden_size=64, num_layers=2, dropout=0.1)
nn.Linear(64, horizon * n_quantiles)          # horizon=4, Q=7 -> 28 outputs
```

Data flow for one batch:

```
x (B, 96)             input window: 96 slots = 24 h of 15-min demand
  -> unsqueeze(-1)    (B, 96, 1)      LSTM reads 1 feature per step
  -> LSTM out         (B, 96, 64)     hidden state per timestep
  -> out[:, -1, :]    (B, 64)         last step carries the summary
  -> Linear head      (B, 28)
  -> view             (B, 4, 7)       4 future slots x 7 quantiles
```

The 7 quantile levels are `[0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]`
(`configs/default.yaml -> model.quantiles`). One network predicts the whole
distribution directly — no sampling, no ensembles.

**Parameter count = 52,252**, and you can verify it:

| block | formula | params |
|---|---|---|
| LSTM layer 1 | 4H(I + H + 2) = 4·64·(1+64) + 2·256 | 17,152 |
| LSTM layer 2 | 4H(H + H + 2) = 4·64·128 + 512 | 33,280 |
| quantile head | 64·28 + 28 | 1,820 |
| **total** | | **52,252** |

### 1.2 The pinball loss (why quantiles come out calibrated)

For quantile level `tau` and error `e = y_true - y_pred`:

```
L_tau(e) = max( tau * e ,  (tau - 1) * e )
```

- under-prediction (e > 0) costs `tau * e`
- over-prediction (e < 0) costs `(1 - tau) * |e|`

At `tau = 0.9` missing low is 9x more expensive than missing high, so the
network learns the 90th percentile, not the mean. `make_loss()` broadcasts
this over all 7 heads at once and averages over batch x horizon x quantiles.

**Critical implementation detail (documented in the module docstring):**
training runs in **z-scored space** — targets are scaled too. Pinball
gradients are bounded (~tau in magnitude), so with raw targets in the tens
of thousands the loss surface is nearly flat and training stalls at a
constant prediction (this was a real bug during development: loss stuck at
~66 vs mean 53k). Fix: fit a separate scaler for Y (`scY`), train on
`scY.transform(Y)`, and invert with `scY.inverse(...)` after prediction.

### 1.3 Training loop (`train_model`)

- Adam, lr 0.001, batch 64, max 50 epochs, early stopping patience 6
- best-val state kept in memory (`best_state`) and restored at the end
- history returned as `{"train": [...], "val": [...]}` -> `outputs/history.png`
- seeds fixed (`seed: 42`) for reproducibility; ~3 min on CPU

### 1.4 From (4, 7) quantiles to the uncertainty triple

`extract_uncertainty(pred_q[t], quantiles, state)` collapses the full
prediction into the three numbers everything downstream consumes:

| output | definition | role |
|---|---|---|
| `u_hat` | q50 of the **next slot** (horizon step 0) | point forecast |
| `spread` | **q99 - q50** | uncertainty width U_t (asymmetric: upside risk only) |
| `rho_t` | see below | risk indicator in [0, 1.5] |

`rho_t` comes from `UncertaintyState` — a rolling 96-slot feedback memory:

```
viol_rate = mean(last 96 violation flags)              # 1 if D > R that slot
z         = |mean(recent demand) - train_mu| / train_sd # demand drift
rho_t     = min(1.5, viol_rate + max(0, z - 2) / 4)
```

So rho stays 0 while the network is calm, grows with recent violations, and
adds a drift term only once demand moves more than 2 sigma off the train
baseline. The feedback loop closes here: violations observed at slot t-1
raise rho_t at slot t, which fattens the reservation in Stage 2.

---

## Stage 2 — Phi: Uncertainty-to-Reservation

The whole transformation is one clamped equation (`phi_transform`):

```
q_tau    = q50 + (tau - 0.5) / 0.49 * spread          # tau = base_tau = 0.9
R_target = q_tau + kappa * spread * (1 + rho_weight * rho_t)
R_target = clip( R_target,
                 (1 + min_headroom) * u_hat,           # floor: 1.05 * q50
                 capacity * max_reserve_ratio )        # ceiling: 0.95 * C
```

Interpretation of every term:

| term | config | meaning |
|---|---|---|
| `q_tau` interpolation | `base_tau: 0.9` | linear map of the 0.5->0.99 spread onto the 0.9 level; at tau=0.99 it returns exactly q99 |
| `kappa * spread` | `kappa: 0.5` | the risk knob — half the upside uncertainty added as buffer |
| `(1 + 0.3 * rho_t)` | `rho_weight: 0.3` | the feedback multiplier — rho>0 inflates the buffer |
| floor `1.05 * q50` | `min_headroom: 0.05` | never reserve less than 5% above the point forecast |
| ceiling `0.95 * C` | `max_reserve_ratio: 0.95` | never claim more than 95% of physical capacity |

### Worked example (your real test set, slot mid-run)

| step | computation | value |
|---|---|---|
| q50 (next slot) | model median | 79,400 |
| spread = q99 - q50 | model | 12,900 |
| q_tau = q90 | 79,400 + 0.4/0.49 x 12,900 | 89,930 |
| kappa buffer | 0.5 x 12,900 x (1 + 0.3x0) | +6,450 |
| **R_target** | | **96,380** |
| EWMA (Stage 4, alpha=0.3) | blends previous reservation | **94,084** |
| realized demand | | 91,200 -> utilization 0.969, no violation |
| floor / ceiling check | 1.05x79,400 / 0.95x143,954.5 | 83,370 / 136,757 (not binding here) |

**Kappa semantics:** kappa=0 reserves the bare q90 (aggressive, more
violations); kappa=1 reserves q99; kappa>1 goes beyond. The sweep in
`run_ucra.py --sweep` traces kappa over `[0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]`
with rho forced to 0 (pure model uncertainty, no feedback contamination) and
EWMA disabled — that curve is the honest risk-utility frontier.

---

## Stage 3 — Risk-Adaptive Allocation

`risk_adaptive_allocate(r_t, demands, risks, alloc_cfg)` splits the slot
reservation across competing entities (sectors on RAN data, slices on CTTC):

```
best_effort_pool = 0.10 * r_t                   # config: best_effort_share
reservable       = r_t - best_effort_pool

w_i      = demand_i^1.0 * (1 + risk_i)^0.5      # pressure x risk weighting
share_i  = w_i / sum(w)
alloc_i  = min(demand_i, reservable * share_i)
# leftover from entities whose demand < share is re-split, proportional to
# share, among the still-unmet entities
unmet_i  = max(demand_i - alloc_i, 0)           # <- these are the violations
```

Design points worth knowing:

- The 10% best-effort pool is never promised to anyone — it absorbs bursts.
- `min(demand_i, ...)` means an entity never gets booked above its need;
  surplus flows to congested neighbors in the leftover pass.
- `risk_exponent: 0.5` (sqrt) deliberately dampens risk so a single
  high-risk entity cannot starve everyone else.
- On the RAN run UCRA feeds `risks=None` (allocation is pressure-driven);
  the hook exists for per-slice risk inputs (CTTC evaluation uses
  slice-type policies instead — see `DEEP_RESULTS.md`).

---

## Stage 4 — Reservation Update & Release

`update_reservation(r_prev, r_target, realized, capacity, upd_cfg)`:

```
r_next = (1 - 0.3) * r_prev + 0.3 * r_target                    # EWMA smooth

headroom_ratio = (r_prev - realized) / capacity
if headroom_ratio > 0.1 and r_target < r_prev:                  # release path
    r_next = (1 - 0.3) * r_prev + 0.3 * max(r_target, realized)
```

Why two paths: ramping UP fast is safe (anticipate demand), but releasing
capacity immediately after one quiet slot causes oscillation ("yo-yo"
provisioning). So releases only start when **>10% of capacity** sits unused
(`release_hysteresis: 0.1`) and even then the floor of the blend is the
realized demand — you never release below what was actually consumed.

Loop start: `r_prev` is initialized to the **train-window q90**
(`np.quantile(series[itr], 0.9)`) so slot 0 starts from a statistically
sane reservation, not zero. At t=0 the "realized" argument is the target
itself (no history exists yet).

---

## Stage 5 — Self-Evolving Learning

Three components in `ucra/core/evolve.py`:

### DriftMonitor (the tripwire)

Rolling 96-slot window, checked **every 96 slots** (`eval_every: 24 h`),
fires when either:

```
viol_rate > 0.15            # config: violation_trigger
z > 3.0                     # config: drift_zscore, z = |D_t - mu| / sd
```

This is why your clean RAN run had **0 updates** — the monitor never fired
because UCRA held 0% violations (no false positives, no wasted retraining).

### ReplayBuffer (continual-learning memory)

Reservoir sampling with capacity 512: every test slot, `evo.remember(x, y)`
stores the (window, target) pair; when full, each new pair replaces a random
old one with probability 512/n_seen. The buffer therefore always holds a
uniform random sample of the entire history — old regimes included — which
is what prevents catastrophic forgetting during fine-tuning.

Note: windows/targets are stored in **z-space** (both scalers), matching
what the model trains on (this was fixed in the Part D patch — storing
raw-scale Y would have broken fine-tuning).

### EvolutionEngine.finetune

```
Adam(lr = 0.001 * 0.3)      # 30% of the original lr: gentle touch
3 epochs full-batch on the 512-pair buffer
```

After an update, `pred_q` is recomputed with the evolved weights, so the
very next slot already benefits. Result in your drift demo: updates fired
at slots **[96, 120, 144]** and cut post-surge violations from 26.5% (frozen)
to **11.6%** (self-evolving).

---

## The closed loop in `scripts/run_ucra.py`

Per test slot `t` (of 252):

```
1. unc      = extract_uncertainty(pred_q[t], ...)      # Stage 1 outputs
2. r_target = phi_transform(u_hat, spread, rho_t, C)   # Stage 2
3. r_t      = update_reservation(r_prev, r_target, D[t-1])   # Stage 4
4. alloc    = risk_adaptive_allocate(r_t, sector_demands)    # Stage 3
5. violated = D[t] > r_t  ->  state.observe, drift.observe   # feedback
6. if t % 96 == 0 and t > 0 and drift.triggered:        # Stage 5
       evo.finetune(); pred_q = re-predict with new weights
7. evo.remember(window_t, target_t)                     # memory grows
8. log row: t, ts, demand, R, R_target, u_hat, spread, rho_t,
            violated, viol_rate, z                -> outputs/ucra_log.csv
```

After the loop: `summarize()` computes the four metrics for UCRA and all
three baselines, everything lands in `outputs/ucra_results.json`, and four
figures are rendered (plus the kappa frontier with `--sweep`).

### Runtime & footprint

| item | value |
|---|---|
| test slots | 252 (1,778-slot segment, 0.7/0.15/0.15 split of 1,679 windows) |
| model | 52,252 params, CPU-trained ~3 min |
| per-slot cost | one LSTM forward (batch of 1) + arithmetic — negligible |
| full run incl. sweep | well under a minute once the checkpoint exists |

### Where each config knob lives

`configs/default.yaml` sections map 1:1: `model.` -> Stage 1, `phi.` ->
Stage 2 + guardrails, `allocation.` -> Stage 3, `update.` -> Stage 4,
`evolution.` -> Stage 5, `eval.sweep_kappa` -> the frontier grid. Change
anything there — no code edits needed.
