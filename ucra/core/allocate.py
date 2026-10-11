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


def allocation_totals(alloc_out: dict, demands: np.ndarray,
                      r_t: float) -> dict:
    """Per-slot Stage-3 accounting totals from a risk_adaptive_allocate result.

    Pure bookkeeping so the runner can LOG what Stage 3 actually handed
    out instead of discarding the allocator's output. The headline
    violation metric (aggregate D_t > R_t) is computed independently of
    this and is unchanged.
    """
    demands = np.asarray(demands, dtype=float)
    alloc = np.asarray(alloc_out["alloc"], dtype=float)
    unmet = np.asarray(alloc_out["unmet"], dtype=float)
    return {
        "n_entities": int(len(demands)),
        "alloc_total": float(alloc.sum()),
        "shortfall_total": float(unmet.sum()),
        "best_effort": float(alloc_out.get("best_effort", 0.0)),
        "demand_total": float(demands.sum()),
        "r_t": float(r_t),
    }


def allocation_summary(totals_rows: list) -> dict:
    """Aggregate Stage-3 allocation accounting over the test window.

    Reported ALONGSIDE (not instead of) the reservation-level metrics:
      - guaranteed share  : fraction of R_t committed to entities
                            (never reaches 1.0 - the best-effort pool
                            beta * R_t is withheld from guarantees by design)
      - shortfall share   : per-entity unmet demand relative to total demand
    Shortfall can be positive on slots where the aggregate SLA holds,
    because guarantees cover only (1 - beta) * R_t.
    """
    if not totals_rows:
        return {"n_slots": 0}
    alloc = np.array([row["alloc_total"] for row in totals_rows])
    shortfall = np.array([row["shortfall_total"] for row in totals_rows])
    demand = np.array([row["demand_total"] for row in totals_rows])
    r_t = np.array([row["r_t"] for row in totals_rows])
    n_entities = np.array([row["n_entities"] for row in totals_rows])
    has_demand = demand > 0
    return {
        "n_slots": int(len(totals_rows)),
        "mean_entities_per_slot": float(n_entities.mean()),
        "mean_guaranteed_share": float(
            np.mean(alloc / np.maximum(r_t, 1e-9))),
        "shortfall_slot_share": float(np.mean(shortfall > 1e-9)),
        "mean_shortfall_share": float(
            np.mean(shortfall[has_demand] / demand[has_demand]))
            if has_demand.any() else 0.0,
        "total_shortfall": float(shortfall.sum()),
        "definition": (
            "Per-entity allocation accounting from Stage 3 "
            "(sum of A_i and sum of unmet_i = max(d_i - A_i, 0)). "
            "The headline violation_rate remains the aggregate D_t > R_t "
            "test applied identically to every method; shortfall here is "
            "demand beyond an entity's guaranteed share, which can be "
            "positive while the slot-level SLA holds because the "
            "best-effort pool (best_effort_share of R_t) is withheld "
            "from guarantees by design."),
    }
