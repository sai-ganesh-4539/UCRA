# 01 -- The Big Picture: What UCRA Is and Why It Exists

## 1. The problem in one paragraph

A 5G operator sells "network slices" -- guaranteed levels of service (URLLC
for factory robots, eMBB for video, mMTC for millions of tiny sensors). Every
15 minutes the network must decide **how much capacity to reserve**. This is a
bet placed *before* demand is known:

- Reserve **too little** -> demand exceeds the reservation -> **violations**:
  dropped packets, missed latency deadlines, angry customers, contractual
  penalties.
- Reserve **too much** -> the extra capacity sits idle -> **waste**: you pay
  for spectrum/energy/servers that carry no traffic.

Classic approaches pick one static number (historical peak, or a forecast
mean) and hope. Both fail: the peak wastes ~30% of capacity all the time, the
mean violates roughly half the time. The core difficulty is that demand is
*uncertain* (you never know the next value exactly) and *non-stationary*
(behavior drifts -- new events, new apps, new days). UCRA is a closed-loop
answer to exactly these two difficulties.

## 2. The idea in one sentence

> Forecast the **whole probability distribution** of future demand (not just
> one number), transform that **uncertainty** into a single reservation size
> through an explicit risk formula (Phi), split that reservation across
> competing consumers under risk weighting, and **keep re-learning** whenever
> reality drifts away from what the model expects.

## 3. The five stages (your framework diagram, annotated)

```
        Network Data  (traffic, capacity, topology, risk)
             |
             v
  +---------------------+
  | 1. UNCERTAINTY      |  quantile LSTM: one input window ->
  |    ESTIMATION       |  7 quantiles of future demand
  +---------------------+  outputs: u_hat (point), U_t (spread), rho_t (risk)
             |
             v
  +---------------------+
  | 2. Phi TRANSFORM    |  R_t = q_tau + kappa * spread * (1 + rho_w * rho_t)
  |    uncertainty ->   |  (floored and ceilinged by guardrails)
  |    reservation      |
  +---------------------+
             |
             v
  +---------------------+          +--------------------------------+
  | 3. RISK-ADAPTIVE    |          | Feedback: reservation error,   |
  |    ALLOCATION       |--------->| violations, utilization,       |
  +---------------------+          | drops / delay (CTTC)           |
             |                     +--------------------------------+
             v                              |            |
  +---------------------+                   |            |
  | 4. RESERVATION      |<------------------+            |
  |    UPDATE & RELEASE |   (EWMA smoothing + release hysteresis)
  +---------------------+                                |
             |                                           |
             v                                           v
  +-----------------------------------------------------------+
  | 5. SELF-EVOLVING LEARNING                                  |
  |  drift monitor -> trigger -> fine-tune on replay buffer    |
  +-----------------------------------------------------------+
```

Stage roles in one line each:

| Stage | Question it answers | Code |
|---|---|---|
| 1 | "What might demand be, and how sure am I?" | `ucra/models/quantile_lstm.py`, `ucra/core/uncertainty.py` |
| 2 | "Given that uncertainty and risk, how much do I reserve?" | `ucra/core/transform.py` (`phi_transform`) |
| 3 | "Who gets what share of the reservation?" | `ucra/core/allocate.py` |
| 4 | "How do I move the reservation smoothly without oscillating?" | `ucra/core/transform.py` (`update_reservation`) |
| 5 | "The world changed -- how do I adapt without forgetting?" | `ucra/core/evolve.py` |

## 4. One slot in the life of UCRA (concrete numbers)

Follow one decision. Time is 17:00 on a real October evening; the slot is
15 minutes long. Demand is measured in data-volume units per slot.

1. **Input (Stage 1).** The last 96 slots (24 hours) of network demand are
   z-scored with training statistics and fed to the quantile LSTM. The model
   emits 7 quantiles (q05 ... q99) for each of the next 4 slots. For the next
   slot suppose it says: median q50 = 79,400 and q99 = 92,300.
2. **Uncertainty triple (Stage 1 output).** `u_hat = 79,400` (point forecast),
   `spread = 92,300 - 79,400 = 12,900` (uncertainty width U_t), and the risk
   indicator `rho_t = 0.0` (no recent violations, demand level within 2
   standard deviations of training mean -- nothing alarming).
3. **Reservation (Stage 2, Phi).** The 90% quantile is approximated from the
   median and spread: `q90 = 79,400 + (0.9-0.5)/0.49 * 12,900 = 89,930`.
   The risk buffer: `kappa * spread * (1 + rho_w * rho_t)
   = 0.5 * 12,900 * 1.0 = 6,450`. Target: `R_target = 96,380`. Guardrails:
   floor `1.05 * 79,400 = 83,370` (never below 5% over the point forecast),
   ceiling `0.95 * 143,954 = 136,757` (never near physical capacity). Neither
   binds, so `R_target = 96,380`.
4. **Smoothing (Stage 4).** The previous applied reservation was 93,100.
   EWMA with alpha = 0.3: `R_t = 0.7 * 93,100 + 0.3 * 96,380 = 94,084`.
   The reservation moves a third of the way toward the target -- no jumps.
5. **Splitting (Stage 3).** 10% of R_t (9,408) is kept as a best-effort pool;
   the other 84,676 is divided across the 75 sectors proportionally to their
   current demand (raised to pressure exponent 1.0). Sectors asking for less
   than their share have the surplus recycled to hungrier sectors.
6. **Reality check (feedback).** The slot ends; realized demand is 91,200.
   `91,200 < 94,084` -> no violation. Utilization this slot:
   `min(91,200/94,084, 1) = 0.969`. The monitors record (no violation, 91,200).
7. **Monitor (Stage 5, asleep).** Rolling violation rate over the last 96
   slots is ~0%, demand z-score ~1.1 -> no trigger. The model is not touched.
   (If the rolling rate had crossed 15% or z had crossed 3.0, the engine would
   fine-tune the LSTM on a replay buffer of recent windows -- see doc 06.)

Multiply this loop by 252 test slots and you have exactly what
`scripts/run_ucra.py` executed on your machine.

## 5. Design principles worth remembering

1. **No future peeking.** Reservations at slot t use only information up to t
   (model inputs, past violations, past demand). The only "cheating" object in
   the repo is the `oracle_quantile` baseline, which exists precisely to show
   what an upper bound looks like -- and UCRA still beats it on the combined
   objective.
2. **Guardrails over cleverness.** Phi's floor (`>= 1.05 * point forecast`)
   and ceiling (`<= 95% of capacity`) mean even a badly wrong forecast cannot
   produce an absurd reservation.
3. **Smoothness is a feature.** EWMA + release hysteresis (Stage 4) exist
   because oscillating reservations cause real operational churn.
4. **Adaptation is triggered, not continuous.** The model is only fine-tuned
   when drift/violation evidence crosses a threshold -- cheap when healthy,
   responsive when sick.
5. **Everything is a knob.** All thresholds live in `configs/default.yaml`
   (doc 11); nothing is hard-coded in the modules.

## 6. What UCRA is NOT

- Not a **packet scheduler**: it decides reservation sizes, not per-packet
  queues.
- Not an **admission controller**: it does not accept/reject slice requests.
- Not a **hardware capacity planner**: the "capacity" C = 1.3 x observed peak
  is a stand-in for the physical limit (the PM dataset does not publish one) --
  see doc 07 for the honest details.
- The acronym collides with a 2020 IEEE paper ("User-Centric Context-Aware
  Resource Allocation"); doc 14 has the exact related-work sentence to use.

## 7. Where to go next

- Run order and file map: doc 02.
- The math of each stage: docs 03, 04, 05, 06.
- What your printed results mean: doc 09.
- Every term: doc 15 (glossary).
