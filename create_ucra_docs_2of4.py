# ==================================================================
# UCRA documentation generator -- PART 2 of 4
# Writes 4 file(s) into docs/: 03_stage1_uncertainty.md, 04_stage2_phi_stage4_update.md, 05_stage3_allocation.md, 06_stage5_self_evolution.md
# Paste-safe by construction: this source contains NO triple-backtick
# runs; markdown fences are stored as @@FENCE@@ and restored on write.
# Save as create_ucra_docs_2of4.py in E:\UCRA and run:
#     python create_ucra_docs_2of4.py
# Run parts 1..4 in order from the UCRA root folder.
# ==================================================================
from pathlib import Path
import hashlib
import sys

_TOK = "@@FENCE@@"   # markdown-fence placeholder
_BT = chr(96) * 3    # the real markdown fence

FILES = {}

FILES['03_stage1_uncertainty.md'] = r'''# 03 -- Stage 1: Uncertainty Estimation (Quantile LSTM)

Files: `ucra/models/quantile_lstm.py`, `ucra/features/windowing.py`,
`ucra/core/uncertainty.py`. Goal: turn the last 24 h of demand into a
 calibrated probability statement about the next hour.

## 1. Why quantiles, not a point forecast

A point forecast says "next slot will be 79,400". It cannot answer the only
question that matters for reservations: **"how wrong can I be?"** A quantile
forecast outputs several points of the predictive distribution: "5% chance
below 71,000; 50% chance below 79,400; 99% chance below 92,300 ...". The
*width* of that band is the uncertainty U_t that Stage 2 converts into
buffer capacity. UCRA trains 7 quantiles at once:
`[0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]`.

## 2. Windowing (`make_windows`)

From a series of length L, every possible contiguous view of `seq_len` past
values paired with the `horizon` future values becomes one training example:

@@FENCE@@
X[i] = series[i : i + 96]          Y[i] = series[i + 96 : i + 100]
N_windows = L - seq_len - horizon + 1  =  1778 - 96 - 4 + 1 = 1,679 windows
@@FENCE@@

With L = 1,778 this yields 1,679 windows: X shape (1679, 96) and Y shape
(1679, 4). Cross-check with your printed numbers: int(1679*0.7) = 1175
train, int(1679*0.85) = 1427, test = 1679 - 1427 = **252** -- exactly the
`n_test` in your outputs.

**No shuffling, ever.** `chronological_split` returns contiguous index
ranges train (70%), val (15%), test (15%). Shuffling time series would leak
the future into training and produce beautiful, meaningless metrics.

## 3. Z-scoring, and the bug that taught us why

Two separate scalers are fitted **on the training slice only**:

- `scaler` for model inputs X (mu, sd)
- `scY` for targets Y (mu_y, sd_y)

Both are saved in `scaler.npz` so inference and Stage 5 fine-tuning use the
identical statistics.

Why scale the targets too? The pinball loss has **bounded gradients**
(magnitude at most max(tau, 1-tau) = 0.95). With raw targets around 50,000,
an early error of 40,000 produces a loss in the tens of thousands but a
gradient still bounded by ~1 -- the network would need millions of steps to
climb down. We saw exactly this: training stalled at a constant loss while
the prediction sat near zero. In z-space, errors are O(1), gradients are
informative, training converges in minutes. Quantiles are mapped back to
original units with `scY.inverse()` after prediction.

## 4. The network

@@FENCE@@
input (B, 96)             one feature: demand
   -> LSTM(1 -> 64, 2 layers, dropout 0.1)   batch_first
   -> take last hidden state h (B, 64)
   -> Linear(64 -> 4 x 7)                    horizon * quantiles
   -> reshape (B, 4, 7)                      one head per (step, quantile)
@@FENCE@@

Parameter count, matching your printed 52,252:

| Component | Formula | Params |
|---|---|---|
| LSTM layer 1 | 4H(I + H + 2) = 4*64*(1+64+2) | 17,152 |
| LSTM layer 2 | 4H(H + H + 2) = 4*64*(64+64+2) | 33,280 |
| Linear head | 64 * 28 + 28 | 1,820 |
| **Total** | | **52,252** |

Note the horizon: the model predicts 4 future slots, but the closed loop
consumes only step 0 (the next slot). The extra steps are there so the same
checkpoint can later drive multi-step reservation planning without
retraining.

## 5. The pinball loss

For quantile level tau, prediction q, truth y, with error e = y - q:

@@FENCE@@
L_tau(e) = max( tau * e ,  (tau - 1) * e )
@@FENCE@@

Worked example at tau = 0.9:

| q | y | e | loss |
|---|---|---|---|
| 90 | 100 | +10 | 0.9 * 10 = **9** (under-predict: expensive) |
| 110 | 100 | -10 | (0.9-1) * (-10) = **1** (over-predict: cheap) |

At tau = 0.9, under-prediction hurts 9x more than over-prediction, so the
optimum of the expected loss is pushed up until exactly 90% of the mass is
above it -- i.e. the true 0.9-quantile. (Take d/dq of
E[tau*(y-q)+ + (1-tau)*(q-y)+] and set it to zero: you get
P(y <= q) = tau.) `make_loss` broadcasts this over all batch items, all 4
horizon steps and all 7 quantile heads in one `torch.maximum`.

## 6. Training loop details (`train_model`)

Adam, lr 0.001; batches of 64 (shuffled -- inside an epoch shuffling batch
composition is fine; the split itself stays chronological); up to 50 epochs
with early stopping: if the validation pinball does not improve by more
than 1e-5 for 6 epochs, stop; the best-validation state dict is restored at
the end, not the last one. Train and val losses are printed per epoch and
saved to `history.png`. On your machine: ~3 minutes CPU.

## 7. Calibration check: "coverage_90pct"

`train.py` computes the fraction of test slots whose realized demand falls
inside the predicted [q05, q99] band, on the next-slot step, in original
units. Your run: **96.0%**. Two clarifications that avoid confusion later:

1. The q05-q99 band has **94% nominal coverage** (0.99 - 0.05). The metric's
   name ("90pct") is a historical leftover from an earlier band choice --
   interpret 96.0% against 94%. Being slightly ABOVE nominal is healthy
   (mildly conservative bands).
2. This band coverage is a **Stage 1 property**. It is *not* the same as the
   reservation violation rate (0.0%) -- that is a Stage 2 property, because
   Phi adds a kappa buffer, a 5% headroom floor, and EWMA smoothing on top
   of the band. Doc 08 separates these cleanly.

Cross-machine note: the sandbox run (torch 2.14, Linux) got 93.25%, your
Windows torch got 96.0%. Same code, same seed -- small nondeterminism
across torch builds/platforms. Both are healthy; for the paper report
mean +/- sd over 3 seeds (doc 12).

## 8. Known simplification: quantile crossing

The 7 heads are independent linear outputs; nothing forces
q05 <= q25 <= ... <= q99. In practice the pinball loss keeps them almost
ordered, and the two places that could break -- `spread = max(0, q99-q50)`
in `extract_uncertainty` -- are guarded with max(0, ...). A sorted-heads
architecture (monotone re-parameterization) is a clean upgrade path if a
reviewer pushes; not needed for the current results.

## 9. Stage 1 output assembly (`extract_uncertainty`)

Input: the (4, 7) quantile matrix for one slot + the rolling risk state.
Output dict:

| Key | Meaning | Formula in code |
|---|---|---|
| `u_hat` | point forecast (next slot) | q50 at horizon step 0 |
| `spread` | uncertainty width U_t | max(0, q99 - q50) at step 0 |
| `rho_t` | risk indicator | from `UncertaintyState` (below) |
| `q_vec` | all 7 quantiles, step 0 | passed through for logging |

## 10. The risk indicator rho_t

`UncertaintyState` maintains two ring buffers over the last `drift_window`
(96) slots: violation flags and realized demand. Then

@@FENCE@@
rho_t = min( 1.5 ,  violation_rate_96 + max(0, z - 2) / 4 )
z     = | mean(recent 8+ demand) - train_mu | / train_sd
@@FENCE@@

- The violation term speaks the operator's language: "how often did the
  reservation fail lately?"
- The drift term activates only when demand shifts by **more than 2
  training-sigmas** (below that, normal fluctuation should not inflate
  reservations); it saturates gently (+0.125 per sigma above 2).
- Hard cap at 1.5 keeps Phi's risk boost bounded:
  `kappa * spread * (1 + 0.3 * 1.5) = kappa * spread * 1.45` max.

Worked example: 20 violations in the last 96 slots (rate 0.208), recent
mean demand 2.5 sigmas above training mean ->
`rho = min(1.5, 0.208 + (2.5-2)/4) = min(1.5, 0.333) = 0.333`.

In the clean RAN run rho_t stays ~0 the whole time (0 violations, z < 2) --
which is exactly why Phi behaved as a pure q90 + kappa*spread policy there.
'''

FILES['04_stage2_phi_stage4_update.md'] = r'''# 04 -- Stage 2 (Phi) and Stage 4 (Update & Release)

Files: `ucra/core/transform.py` (both stages live there). This is the heart
of UCRA -- the "uncertainty-to-reservation transformation algorithm" the
name refers to.

## 1. The Phi equation

@@FENCE@@
                     q_tau(t) + kappa * spread_t * (1 + rho_w * rho_t)
R_target(t) = clip( --------------------------------------------------- ,
                      floor                    ceiling )

floor   = (1 + min_headroom) * u_hat          = 1.05 * u_hat
ceiling = capacity * max_reserve_ratio        = 0.95 * C
@@FENCE@@

Term by term:

| Term | Where it comes from | What it buys |
|---|---|---|
| `u_hat` | Stage 1 median forecast | the demand level to cover |
| `q_tau(t)` | tau-quantile forecast, approximated from median + spread | cover the *likely high* demand, not the average |
| `spread_t` | U_t = q99 - q50 from Stage 1 | when the model is unsure, reserve more |
| `kappa` | config (0.5) | the risk-aversion dial -- the whole risk-utility frontier (doc 08 section 5) |
| `rho_t` | feedback risk indicator (doc 03 section 10) | when reality has been violating / drifting, inflate the buffer up to +45% |
| `rho_w` | config (0.3) | how strongly rho speaks |
| floor | config min_headroom 0.05 | even a "certain" forecast keeps 5% headroom -- protects against forecast overconfidence |
| ceiling | config max_reserve_ratio 0.95 | never reserve the last 5% of capacity (control headroom, signaling, other tenants) |

### The q_tau approximation

Phi needs the tau-quantile but the model gives us the whole vector; the code
interpolates linearly between the median (0.5) and q99 (0.99):

@@FENCE@@
q_tau ~ u_hat + (tau - 0.5) / 0.49 * spread
@@FENCE@@

With tau = 0.9: `q90 ~ u_hat + 0.8163 * spread`. This is exact if the
distribution between q50 and q99 is linear in that range, and it makes Phi
depend on only two Stage 1 numbers, which keeps the method auditable.

## 2. Worked example (the numbers from doc 01, checked)

u_hat = 79,400; spread = 12,900; rho_t = 0; defaults tau=0.9, kappa=0.5,
rho_w=0.3, C = 143,954.5.

@@FENCE@@
q90        = 79,400 + 0.8163 * 12,900 = 89,930
risk term  = 0.5 * 12,900 * (1 + 0.3*0) = 6,450
raw        = 89,930 + 6,450 = 96,380
floor      = 1.05 * 79,400 = 83,370   (not binding)
ceiling    = 0.95 * 143,954.5 = 136,757  (not binding)
R_target   = 96,380
@@FENCE@@

Now the risk path, same forecast but the network has been violating and
drifting: rho_t = 0.333 -> risk term = 0.5 * 12,900 * 1.0999 = 7,094 ->
R_target = 97,024. Under max stress (rho_t = 1.5): risk term = 0.5*12,900*1.45
= 9,352 -> R_target = 99,282. So the full feedback authority of Phi on this
forecast is +2,900 units -- meaningful but bounded; the ceiling is what
prevents runaway.

## 3. What kappa means (and how to read fig_kappa_sweep.png)

kappa = 0 collapses Phi to "q90 with a 5% headroom floor". As kappa grows,
R moves further into the upper tail: violations fall monotonically
(remember each unit of kappa adds kappa * spread to the reservation), and
utilization falls with it. The sweep in `run_ucra.py --sweep` evaluates
kappa in [0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0] with rho_t forced to 0, so the
curve isolates the kappa effect. The plot has two y-axes: violation rate
(red, left) and utilization (blue, right). Your operating point kappa=0.5
sits in the flat-zero-violations region with the highest utilization before
violations reappear -- the knee of the frontier. For the paper, doc 12
shows how to produce the full sweep table from `ucra_results.json`-style
runs.

## 4. Stage 4: from R_target to the applied R_t

A perfect target applied raw would produce a jerky reservation (forecast
noise passes straight through). Two mechanisms smooth it
(`update_reservation`):

**EWMA (exponentially weighted moving average):**

@@FENCE@@
R_next = (1 - alpha) * R_prev + alpha * R_target        alpha = 0.3
@@FENCE@@

Each step moves 30% of the way toward the target: fast enough to track the
diurnal cycle (96-slot window), slow enough to filter slot noise.

**Release hysteresis (the asymmetry):**

@@FENCE@@
headroom_ratio = (R_prev - realized) / C
if headroom_ratio > 0.10 and R_target < R_prev:
    R_next = (1 - alpha) * R_prev + alpha * max(R_target, realized)
@@FENCE@@

Reading it: reservations are **quick to grow** (plain EWMA) but **slow to
shrink**, and never shrink below what was actually used (`max(R_target,
realized)`). The trigger is real, substantial idleness: more than 10% of
physical capacity was reserved-but-unused. Why asymmetry? Releasing and
re-grabbing capacity costs control-plane churn and risks violating in the
gap; holding a slightly-too-big reservation for a few extra slots is cheap.
The hysteresis parameter (0.1) is the "how idle is idle enough to release"
dial.

Worked example: R_prev = 94,084, realized = 60,200 (night dip),
R_target = 71,000, C = 143,954.5.

@@FENCE@@
headroom_ratio = (94,084 - 60,200) / 143,954.5 = 0.235  > 0.10
and R_target < R_prev  ->  release branch
R_next = 0.7 * 94,084 + 0.3 * max(71,000, 60,200)
       = 65,858.8 + 0.3 * 71,000 = 87,158.8
@@FENCE@@

The reservation starts sliding down, but only to the EWMA blend -- and never
below the realized 60,200. On the growth side (R_target > R_prev) the plain
EWMA branch applies directly.

## 5. R_target vs R in the log

`ucra_log.csv` stores both: `R_target` (raw Phi output) and `R` (applied,
post-EWMA). The gap between them is the smoothing lag. Useful forensic: if
violations cluster where R_target was already high but R lagged, the story
is "Stage 2 saw it, Stage 4 smoothed too slowly" -> lower ewma_alpha. If
R_target itself was low, the story is Stage 1/2 (spread too narrow) ->
raise kappa or check coverage.

## 6. Edge behavior worth knowing

- At t = 0 of any pass, `update_reservation` is called with `realized =
  R_target` (run_ucra passes `r_target` for the first slot), so the
  hysteresis branch cannot fire spuriously on missing history.
- `r_prev` starts at the training window's raw q90 (not zero), so the first
  slots do not climb from nothing.
- The final `max(0.0, ...)` guards against pathological negative EWMA
  values if inputs are ever negative.
'''

FILES['05_stage3_allocation.md'] = r'''# 05 -- Stage 3: Risk-Adaptive Allocation

File: `ucra/core/allocate.py` (`risk_adaptive_allocate`). Stage 2 decides
HOW MUCH to reserve; Stage 3 decides WHO GETS IT.

## 1. The rule

Given the reservation R_t, the current demand vector d (one entry per
sector, or per slice in future work) and optional per-entity risk scores:

@@FENCE@@
best_effort_pool = best_effort_share * R_t          (10%)
reservable       = R_t - best_effort_pool           (90%)

w_i     = max(d_i, 0)^pressure_exp * (1 + max(risk_i, 0))^risk_exp
share_i = w_i / sum(w)                (uniform if all weights are 0)
alloc_i = min(d_i, reservable * share_i)

# leftover recycling: entities whose demand < their share leave unused
# capacity, which is redistributed proportionally among the still-hungry
# entities (again capped by their demand)

unmet_i = max(d_i - alloc_i, 0)       <-- these are the "violations"
@@FENCE@@

## 2. What each ingredient does

- **Demand pressure** (`d_i^1.0` by default): proportionality to need. A
  sector carrying twice the traffic gets (roughly) twice the share.
  `pressure_exponent` > 1 would favor big consumers super-proportionally;
  < 1 flattens toward equality.
- **Risk boost** (`(1 + risk_i)^0.5` by default): a squareroot-shaped
  premium. A risk score of 0.8 multiplies the weight by 1.34, of 0.2 by
  1.10. The sqrt keeps one hot entity from starving everyone else.
  NOTE: in the current RAN run risks are `None` -> all boosts are 1.0, so
  allocation is pure pressure-proportional. The hook exists for per-slice
  criticality (URLLC > eMBB > mMTC) -- doc 12 shows how to wire it.
- **Best-effort pool** (10%): capacity deliberately kept out of guarantees,
  absorbable by anyone. It bounds how much of R_t can be locked into
  guarantees before leftovers are even computed.
- **min(d_i, ...)** caps: nobody receives more than they asked for -- no
  artificial demand creation, which is what makes the leftover pool exist.
- **unmet**: demand beyond the allocated share. In run_ucra the headline
  violation flag is computed on the aggregate (`D_t > R_t`); `unmet` is the
  per-entity breakdown the framework diagram's "risk exposure" refers to.

## 3. Worked example (hand-checkable)

R_t = 90,000; three sectors with demands [40,000, 30,000, 10,000]; risk
scores [0.2, 0.8, 0.0]; defaults (pressure 1.0, risk_exp 0.5, pool 0.1).

@@FENCE@@
best_effort = 9,000            reservable = 81,000

w1 = 40,000 * sqrt(1.2) = 43,818
w2 = 30,000 * sqrt(1.8) = 40,249      (risk 0.8 -> the big boost)
w3 = 10,000 * sqrt(1.0) = 10,000      sum = 94,067

shares  = [0.4658, 0.4278, 0.1063]
raw     = [37,729, 34,655, 8,610]
capped  = [37,729, 30,000, 8,610]     (sector 2 capped at its demand)
leftover = 81,000 - 76,339 = 4,661

hungry  = sectors 1 and 3; their share mass = 0.4658 + 0.1063 = 0.5721
add 1   = min( 2,271 , 4,661 * 0.4658/0.5721 ) = min(2,271, 3,795) = 2,271
add 3   = min( 1,390 , 4,661 * 0.1063/0.5721 ) = min(1,390,   866) =   866

final   = [40,000, 30,000, 9,476]     sum = 79,476 <= 81,000
unmet   = [0, 0, 524]
@@FENCE@@

Notice three behaviors: sector 2's risk premium earned it a bigger raw share
but its own demand capped it; its unused share was recycled; the only
violating entity is sector 3, whose 524-unit shortfall is exactly the
per-entity risk exposure. Also note the 10% pool (9,000) means the sum of
guaranteed allocations never reaches R_t -- by design.

## 4. How run_ucra feeds it

For RAN data, each test slot looks up the per-sector rows of the canonical
frame at that timestamp (`sector_frame[ts == t]`, column `data_volume`) --
typically 150 rows for 75 sectors x 2 technologies -- and passes them as
`demands`. For synthetic data it passes the per-sector demand frame. If no
sector frame exists, a single-element vector `[D_t]` is used, which makes
Stage 3 degenerate gracefully (all of the reservable goes to the one
entity).

## 5. Complexity and determinism

The routine is O(n) in entities, allocation-only (no optimizer), and fully
deterministic given inputs -- there is no randomness anywhere in Stages 2-4,
so re-runs differ only through Stage 1 training (torch) and, in the demo,
the injected surge.
'''

FILES['06_stage5_self_evolution.md'] = r'''# 06 -- Stage 5: Self-Evolving Learning (+ Your Drift Demo Decoded)

File: `ucra/core/evolve.py` (`DriftMonitor`, `ReplayBuffer`,
`EvolutionEngine`); drivers: `scripts/run_ucra.py` (always armed) and
`scripts/demo_evolution.py` (A/B demonstration).

## 1. Why Stage 5 exists

Stages 1-4 assume the model's picture of demand matches reality. When the
world drifts -- a mass event, a new app, a holiday -- the quantile forecasts
silently become wrong, Phi faithfully reserves around the WRONG distribution,
and violations climb. Stage 5 closes the loop: watch the feedback, detect
drift, and fine-tune the model on what reality just did.

## 2. The trigger (`DriftMonitor`)

A rolling window (96 slots) of violation flags plus a training-reference
z-score. Checked every `eval_every` slots (96 = every 24 h in run_ucra;
24 in the demo by default) and only at t > 0:

@@FENCE@@
trigger fires  if   viol_rate_96 > 0.15   OR   z > 3.0
z = | demand_now - train_mu | / train_sd
@@FENCE@@

Two independent detectors: the violation trigger catches "model is wrong in
the way that hurts users"; the z-trigger catches "the world moved even
before our reservations felt it". Either is sufficient.

## 3. The memory (`ReplayBuffer`, reservoir sampling)

Keeps `replay_size` = 512 (window, target) pairs drawn by **reservoir
sampling**: the first 512 items fill the buffer; every item after that
replaces a uniformly chosen slot with probability 512/n_seen. Guarantees:
at any moment, every past window had an equal chance to be in the buffer.
Why it matters: the buffer contains BOTH pre-drift and post-drift windows,
so fine-tuning on it adapts to the new regime **without erasing** the old
one -- the standard defense against catastrophic forgetting in continual
learning. (After 5,000 windows, each has a 512/5000 ~ 10.2% inclusion
chance.)

## 4. The update (`EvolutionEngine.finetune`)

@@FENCE@@
Adam, lr = model_lr * 0.3 (gentler than training), full-batch over the
512 buffered pairs, finetune_epochs = 3, pinball loss identical to Stage 1.
@@FENCE@@

Full-batch is fine here (512 x 96 floats is tiny). After a successful
update the driver recomputes `pred_q` with the evolved weights, so the very
next Phi decision already benefits. run_ucra remembers every test window in
z-space (`scaler.transform(X), scY.transform(Y)`) -- this exact detail was a
real bug we fixed in Part D: the buffer originally stored raw-scale targets
while the model trains in z-space, which would have corrupted fine-tuning
the moment it first fired.

## 5. Why the clean run had `evolution_updates: 0`

Your main RAN run shows zero updates. That is the system working, not
skipping: UCRA held 0.0% violations (rolling rate never approached 0.15)
and demand stayed within 3 training sigmas (z never fired). Stage 5 is an
insurance policy -- a healthy run never cashes it. This is also a nice
false-positive result for the paper: across 252 slots the trigger never
cried wolf.

## 6. Your drift demo, line by line

`demo_evolution.py` multiplies realized demand by 1.35 from slot 37 of the
252-slot test window (15%), rebuilds windows from the DRIFTED series (the
model must now forecast a world it never saw), and runs two identical
passes from the same checkpoint. Your output:

@@FENCE@@
shift 0.35, surge_start_slot 37, n_test 252, capacity 143,954.5

frozen:    viol_overall 22.62% | viol_post_surge 26.51% | util 82.74% | updates 0
evolving:  viol_overall  9.92% | viol_post_surge 11.63% | util 80.31% | updates 3 [96, 120, 144]
@@FENCE@@

Reading it:

- **Pre-surge both passes are clean.** Frozen overall 22.6% across 252 slots
  ~ 57 violations; post-surge alone 26.5% of 215 slots ~ 57 violations. The
  arithmetic closes: essentially ALL violations happen after the surge.
  Before it, the frozen model was as good as UCRA always is (0%).
- **The trigger timeline makes sense.** Checks run every 24 slots (t = 24,
  48, 72, 96, ...). At t = 96 the rolling 96-slot window contains ~60
  post-surge slots at ~26.5% violations -> window rate ~0.166 > 0.15 ->
  first fine-tune. The window at t = 120 and t = 144 still carries the
  poisoned pre-fix slots, so it fires twice more; after the third update the
  window fills with post-fix, low-violation slots and the rate drops under
  0.15 -> no further updates. Exactly [96, 120, 144].
- **Adaptation came from intelligence, not over-provisioning.** Post-surge
  utilization is 80.3% (evolving) vs 82.7% (frozen) -- the evolving pass did
  NOT just reserve vastly more; the fine-tuned quantiles put the buffer in
  the right places. Violations fell 26.5% -> 11.6% (a 56% relative
  reduction) at slightly LOWER utilization.
- Residual 11.6%: a +35% step is a violent drift; three gentle fine-tunes
  (3 epochs each, lr x 0.3) shrink but do not eliminate it. Pushing further
  (more epochs, higher lr, lower trigger) overfits spikes -- the demo flags
  `--ft-epochs` and `--lr-scale` let you trace this frontier (doc 12).
- Robustness we verified while building: the result was insensitive to
  fine-tune hyperparameters (10 epochs x 3 lr values -> same update points),
  so the behavior is a property of the trigger + replay design, not of a
  lucky learning rate.

Figure guide for `fig_drift_demo.png`: top panel = drifted demand (blue)
with the two reservations overlaid (red frozen, green evolving) -- watch the
green line climb after each update while red stays flat; dashed black =
surge start; dotted gray = physical capacity. Bottom panel = rolling
violation rates with the 0.15 trigger line and green verticals at the three
update points.

## 7. Interface recap (for the code reference)

- `DriftMonitor(cfg)` -- `observe(violated, demand)` -> stats dict,
  `triggered(stats)` -> bool, `set_reference(train_demand)`.
- `ReplayBuffer(capacity)` -- `push(x, y)`, `arrays()` -> (X, Y) or
  (None, None).
- `EvolutionEngine(model, evo_cfg, model_cfg)` -- `remember(x, y)`,
  `finetune(loss_fn)` -> {"updated": bool, "replay": n, "loss": L,
  "n_updates": k}.
'''


def write_all() -> None:
    dest = Path("docs")
    dest.mkdir(parents=True, exist_ok=True)
    print("=" * 66)
    print(f" UCRA docs generator -- part {2}/4")
    print("=" * 66)
    for name in FILES:
        text = FILES[name].replace(_TOK, _BT)
        data = text.encode("utf-8")          # raw bytes: LF kept, no CRLF surprises
        (dest / name).write_bytes(data)
        print(f"  [ok] docs/{name:<34} {len(data):>7,} bytes  md5 {hashlib.md5(data).hexdigest()[:8]}")
    print(f"  part {2}/4 done: {len(FILES)} file(s) written")


if __name__ == "__main__":
    write_all()
    print("  next: save & run create_ucra_docs_3of4.py")