# 08 -- Metrics and Baselines: Formulas, Intuition, What "Good" Means

Files: `ucra/eval/metrics.py`, `ucra/eval/baselines.py`. All metrics are
computed on the TEST window; R = reservation, D = realized demand,
C = capacity (143,954.5 on your run).

## 1. The four metrics

| Metric | Formula | Unit | Question it answers |
|---|---|---|---|
| violation_rate | mean( D > R ) | fraction of slots | How often did we fail users? (risk exposure) |
| utilization | mean( min(D/R, 1) ) | 0..1 | Of what we reserved, how much was actually used? (efficiency) |
| reservation_error | mean( abs(R - D) ) / C | fraction of capacity | How far off was the reservation on average? (tracking accuracy) |
| waste | mean( max(R - D, 0) ) / C | fraction of capacity | How much reserved capacity idled? (cost) |

Relationships worth internalizing:

- violation_rate + something like "slack" = 100%: a policy can only lower
  violations by reserving more, which shows up in utilization/waste. There
  is no free lunch -- the deliverable is the FRONTIER, not one number.
- utilization is capped at 1 per slot: over-demand does not inflate it; it
  shows up as violations instead. A policy can have utilization 95% AND 50%
  violations (your mean_forecast row does exactly that).
- reservation_error counts both directions (over and under), normalized by
  capacity. Two policies with equal error can be wildly different: one
  violates often, the other over-reserves. Always read it together with
  violation_rate.

What "good" looks like in this problem: violation_rate near 0 (this is the
SLA), then maximize utilization subject to that. UCRA's 0.0% @ 71.7% means
"never failed, and the guarantee cost ~28% slack".

## 2. Why band coverage is NOT here

The 96.0% "coverage" from train.py is a Stage 1 calibration metric (is the
q05-q99 band honest?), not a reservation metric. A perfectly calibrated
model can still produce a bad reservation policy, and vice versa. Keeping
the two layers separate is what makes failures diagnosable (doc 04
section 5).

## 3. The three baselines and what each defends

| Baseline | R_t | What it represents | Its signature failure |
|---|---|---|---|
| static_peak | max(train demand), constant | worst-case static sizing, the "safe" naive choice | 0 violations but 30.5% of capacity wasted forever (res-err 30.5%) |
| mean_forecast | Stage 1 median (step 0) | "we have a great point forecast, ignore uncertainty" | cheapest possible R -> violates 50.4% of slots |
| oracle_quantile | true rolling q90 of the TEST window itself (window 48) | CHEATING upper bound: sees the future it defends against | still 16.3% violations (lags bursts), res-err 20.1% |

Why the oracle matters: reviewers ask "what would a perfect quantile
policy achieve?" The oracle answers it -- and because UCRA (which never
sees the future) holds 0.0% violations at higher utilization than the
oracle's 64.9%, UCRA's kappa buffer demonstrably out-sizes a naive rolling
quantile. Why the oracle still violates 16.3% despite cheating: its
"quantile of the last 48 realized slots" reacts to bursts AFTER they start
and forgets lulls; by construction a 90% quantile violates ~10% of slots
even with perfect knowledge, and heavy-tailed 15-min bursts push that
higher.

One subtlety: static_peak uses `max(series.values[train_idx])` -- the peak
over training-window start samples -- so a freak test-window spike above
the training peak WOULD violate it. On your segment it does not happen
(0.0%), which is the honest reading: static peak is "safe" only in
stationary weather.

## 4. How the comparison is wired in run_ucra.py

After the closed loop, the same `summarize()` is applied to four R vectors:
UCRA's loop output, static_peak's constant, mean_forecast's per-slot
medians, and the oracle's rolling q90 -- all against the same D_test and C.
One number quirk: mean_forecast's reservation_error (2.0%) is the LOWEST of
all policies -- chasing D tightly with the median minimizes mean absolute
gap -- while being catastrophically the worst on violations. This is the
cleanest illustration in the whole project of why reservation_error alone
must never be the objective.

## 5. The kappa sweep: your frontier in one picture

Each kappa value re-runs Phi (rho = 0) over the test window and reports the
(violation, utilization) pair. Read as an economic frontier: kappa is the
price you pay in slack per unit of uncertainty; violations are the risk you
buy down. On your run, kappa = 0.5 already sits at zero violations, so the
knee (the smallest kappa with 0% violations) lies somewhere in 0.25-0.5 --
read the exact knee off your fig_kappa_sweep.png. Beyond the knee, extra
kappa only buys idle capacity. The CTTC eval embeds the same sweep per
slice type inside `cttc_results.json` (key `kappa_sweep`), which is how you
would justify per-slice kappa choices in the paper.
