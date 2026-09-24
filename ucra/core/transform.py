"""Stage 2 (Phi: Uncertainty-to-Reservation) and Stage 4 (Update & Release).

Phi (see docs/METHODOLOGY.md):
    R_target = clip( q_tau + kappa * spread * (1 + rho_weight * rho_t),
                     min_headroom * u_hat, capacity * max_reserve_ratio )

Stage 4 smooths the target into R_{t+1} with EWMA + release hysteresis:
reservations drop only when headroom stays large (avoids oscillation).
"""
from __future__ import annotations

import numpy as np


def phi_transform(u_hat: float, spread: float, rho_t: float,
                  capacity: float, phi_cfg: dict) -> float:
    """Uncertainty triple (u_hat, U_t, rho_t) -> reservation R_t."""
    tau = phi_cfg["base_tau"]
    # q_tau approximated from the median and the spread ratio:
    # q_tau ~ q50 + (tau-0.5)/(0.99-0.5) * spread
    q_tau = u_hat + (tau - 0.5) / 0.49 * spread
    r = q_tau + phi_cfg["kappa"] * spread * (1.0 + phi_cfg["rho_weight"] * rho_t)
    floor = (1.0 + phi_cfg["min_headroom"]) * u_hat
    ceil = capacity * phi_cfg["max_reserve_ratio"]
    return float(np.clip(r, floor, ceil))


def update_reservation(r_prev: float, r_target: float, realized: float,
                       capacity: float, upd_cfg: dict) -> float:
    """Stage 4: EWMA-smoothed reservation with release hysteresis."""
    a = upd_cfg["ewma_alpha"]
    r_next = (1 - a) * r_prev + a * r_target
    # release only if unused headroom exceeds hysteresis threshold
    headroom_ratio = (r_prev - realized) / max(capacity, 1e-9)
    if headroom_ratio > upd_cfg["release_hysteresis"] and r_target < r_prev:
        r_next = (1 - a) * r_prev + a * max(r_target, realized)
    return float(max(0.0, r_next))
