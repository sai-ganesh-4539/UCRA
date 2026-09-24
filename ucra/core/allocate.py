"""Stage 3 - Risk-Adaptive Allocation.

Distributes the reservation R_t across competing entities (sectors or
slices) proportionally to demand pressure and per-entity risk, keeping a
best-effort pool aside. Violation = demand that exceeds the allocated share.
"""
from __future__ import annotations

import numpy as np


def risk_adaptive_allocate(r_t: float, demands: np.ndarray,
                           risks: np.ndarray | None = None,
                           alloc_cfg: dict | None = None) -> dict:
    """Split R_t across entities given their current demand and risk.

    weights_i = demand_i^pressure_exp * (1 + risk_i)^risk_exp
    A_i = min(demand_i, R_t * w_i / sum(w)) + best-effort leftovers

    Returns dict with allocations A, unmet (violations) and the pool split.
    """
    cfg = alloc_cfg or {}
    p_exp = cfg.get("pressure_exponent", 1.0)
    r_exp = cfg.get("risk_exponent", 0.5)
    be_share = cfg.get("best_effort_share", 0.1)

    demands = np.asarray(demands, dtype=float)
    risks = np.zeros_like(demands) if risks is None \
        else np.asarray(risks, dtype=float)
    n = len(demands)
    if n == 0 or r_t <= 0:
        return {"alloc": np.zeros(n), "unmet": demands.copy(),
                "best_effort": 0.0}

    best_effort_pool = be_share * r_t
    reservable = r_t - best_effort_pool

    w = (np.maximum(demands, 0.0) ** p_exp) * ((1.0 + np.maximum(risks, 0.0)) ** r_exp)
    tot = float(w.sum())
    shares = (w / tot) if tot > 0 else np.full(n, 1.0 / n)

    alloc = np.minimum(demands, reservable * shares)
    # leftover from entities whose demand < share -> best effort
    leftover = reservable - float(alloc.sum())
    if leftover > 0:
        unmet_idx = np.where(demands > alloc)[0]
        if len(unmet_idx):
            add = np.minimum(demands[unmet_idx] - alloc[unmet_idx],
                             leftover * shares[unmet_idx] / max(shares[unmet_idx].sum(), 1e-9))
            alloc[unmet_idx] += add
    unmet = np.maximum(demands - alloc, 0.0)
    return {"alloc": alloc, "unmet": unmet,
            "best_effort": best_effort_pool}
