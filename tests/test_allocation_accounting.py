"""Additive validation: Stage-3 allocation accounting invariants.

Property tests for risk_adaptive_allocate:
  A1 allocations are non-negative and never exceed demand
  A2 allocated + best-effort pool never exceeds the reservation R_t
  A3 unmet is exactly max(demand - alloc, 0)
  A4 edge cases: zero demand, zero reservation, single entity
  A5 zero-demand entities receive zero allocation

Demand inputs are non-negative by system contract (traffic demand
counts/bytes from the dataset loaders); the tests document that contract.
Measurement-only: no core module is modified.
"""
import numpy as np

from ucra.core.allocate import risk_adaptive_allocate


def test_alloc_random_properties_hold():
    rng = np.random.default_rng(0)
    for _ in range(200):
        n = int(rng.integers(1, 8))
        d = rng.uniform(0, 100, n)
        risks = rng.uniform(0, 2, n)
        r_t = float(rng.uniform(0, 500))
        out = risk_adaptive_allocate(r_t, d, risks)
        assert np.all(out["alloc"] >= 0), "A1: allocations must be >= 0"
        assert np.all(out["alloc"] <= d + 1e-9), "A1: alloc <= demand"
        assert out["alloc"].sum() + out["best_effort"] <= r_t + 1e-9, \
            "A2: total handed out + pool must fit inside R_t"
        assert np.allclose(out["unmet"], np.maximum(d - out["alloc"], 0)), \
            "A3: unmet must equal max(demand - alloc, 0)"


def test_alloc_zero_demand_gets_nothing():
    out = risk_adaptive_allocate(100.0, np.zeros(3))
    assert np.allclose(out["alloc"], 0.0)
    assert np.allclose(out["unmet"], 0.0)


def test_alloc_zero_reservation_all_unmet():
    out = risk_adaptive_allocate(0.0, np.array([10.0, 20.0]))
    assert np.allclose(out["alloc"], 0.0)
    assert np.allclose(out["unmet"], [10.0, 20.0])


def test_alloc_single_entity_capped_at_demand():
    out = risk_adaptive_allocate(1000.0, np.array([50.0]))
    assert out["alloc"][0] == 50.0
    assert out["unmet"][0] == 0.0
    assert out["best_effort"] > 0.0


def test_alloc_best_effort_pool_accounted_separately():
    """beta = 0.1 of R_t is withheld by design (paper Eq. alloc, beta=0.1)."""
    d = np.array([80.0, 80.0])
    out = risk_adaptive_allocate(200.0, d, np.array([0.0, 0.0]))
    assert abs(out["best_effort"] - 20.0) < 1e-9
    assert out["alloc"].sum() <= 180.0 + 1e-9


def test_alloc_empty_entity_list():
    out = risk_adaptive_allocate(100.0, np.zeros(0))
    assert len(out["alloc"]) == 0
    assert out["best_effort"] == 0.0