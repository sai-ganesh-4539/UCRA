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

```
R_t = clip( q_tau_hat + kappa * spread_t * (1 + rho_weight * rho_t), 0, C * max_reserve_ratio )

where
  q_tau_hat(t) = tau-quantile forecast of demand at time t   (base_tau = 0.9)
  spread_t     = q_0.99(t) - q_0.50(t)                       (uncertainty width U_t)
  rho_t        = recent violation rate / drift signal        (risk indicator)
  kappa        = risk aversion (kappa, swept in evaluation)
  rho_weight   = rho_weight
```

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
