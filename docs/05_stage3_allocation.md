# 05 -- Stage 3: Risk-Adaptive Allocation

File: `ucra/core/allocate.py` (`risk_adaptive_allocate`). Stage 2 decides
HOW MUCH to reserve; Stage 3 decides WHO GETS IT.

## 1. The rule

Given the reservation R_t, the current demand vector d (one entry per
sector, or per slice in future work) and optional per-entity risk scores:

```
best_effort_pool = best_effort_share * R_t          (10%)
reservable       = R_t - best_effort_pool           (90%)

w_i     = max(d_i, 0)^pressure_exp * (1 + max(risk_i, 0))^risk_exp
share_i = w_i / sum(w)                (uniform if all weights are 0)
alloc_i = min(d_i, reservable * share_i)

# leftover recycling: entities whose demand < their share leave unused
# capacity, which is redistributed proportionally among the still-hungry
# entities (again capped by their demand)

unmet_i = max(d_i - alloc_i, 0)       <-- these are the "violations"
```

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

```
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
```

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
