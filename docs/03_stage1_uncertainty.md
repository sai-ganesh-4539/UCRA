# 03 -- Stage 1: Uncertainty Estimation (Quantile LSTM)

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

```
X[i] = series[i : i + 96]          Y[i] = series[i + 96 : i + 100]
N_windows = L - seq_len - horizon + 1  =  1778 - 96 - 4 + 1 = 1,679 windows
```

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

```
input (B, 96)             one feature: demand
   -> LSTM(1 -> 64, 2 layers, dropout 0.1)   batch_first
   -> take last hidden state h (B, 64)
   -> Linear(64 -> 4 x 7)                    horizon * quantiles
   -> reshape (B, 4, 7)                      one head per (step, quantile)
```

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

```
L_tau(e) = max( tau * e ,  (tau - 1) * e )
```

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

```
rho_t = min( 1.5 ,  violation_rate_96 + max(0, z - 2) / 4 )
z     = | mean(recent 8+ demand) - train_mu | / train_sd
```

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
