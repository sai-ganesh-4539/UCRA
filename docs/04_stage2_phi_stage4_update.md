# 04 -- Stage 2 (Phi) and Stage 4 (Update & Release)

Files: `ucra/core/transform.py` (both stages live there). This is the heart
of UCRA -- the "uncertainty-to-reservation transformation algorithm" the
name refers to.

## 1. The Phi equation

```
                     q_tau(t) + kappa * spread_t * (1 + rho_w * rho_t)
R_target(t) = clip( --------------------------------------------------- ,
                      floor                    ceiling )

floor   = (1 + min_headroom) * u_hat          = 1.05 * u_hat
ceiling = capacity * max_reserve_ratio        = 0.95 * C
```

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

```
q_tau ~ u_hat + (tau - 0.5) / 0.49 * spread
```

With tau = 0.9: `q90 ~ u_hat + 0.8163 * spread`. This is exact if the
distribution between q50 and q99 is linear in that range, and it makes Phi
depend on only two Stage 1 numbers, which keeps the method auditable.

## 2. Worked example (the numbers from doc 01, checked)

u_hat = 79,400; spread = 12,900; rho_t = 0; defaults tau=0.9, kappa=0.5,
rho_w=0.3, C = 143,954.5.

```
q90        = 79,400 + 0.8163 * 12,900 = 89,930
risk term  = 0.5 * 12,900 * (1 + 0.3*0) = 6,450
raw        = 89,930 + 6,450 = 96,380
floor      = 1.05 * 79,400 = 83,370   (not binding)
ceiling    = 0.95 * 143,954.5 = 136,757  (not binding)
R_target   = 96,380
```

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

```
R_next = (1 - alpha) * R_prev + alpha * R_target        alpha = 0.3
```

Each step moves 30% of the way toward the target: fast enough to track the
diurnal cycle (96-slot window), slow enough to filter slot noise.

**Release hysteresis (the asymmetry):**

```
headroom_ratio = (R_prev - realized) / C
if headroom_ratio > 0.10 and R_target < R_prev:
    R_next = (1 - alpha) * R_prev + alpha * max(R_target, realized)
```

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

```
headroom_ratio = (94,084 - 60,200) / 143,954.5 = 0.235  > 0.10
and R_target < R_prev  ->  release branch
R_next = 0.7 * 94,084 + 0.3 * max(71,000, 60,200)
       = 65,858.8 + 0.3 * 71,000 = 87,158.8
```

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
