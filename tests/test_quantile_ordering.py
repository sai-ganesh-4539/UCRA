"""Additive validation: quantile ordering & the Stage-2 spread guard.

Measures (does not change) quantile-crossing behaviour:
  1. extract_uncertainty's spread = max(0, q99 - q50) must clamp crossings
     so Stage 2 can never receive a negative uncertainty width.
  2. a reusable counter for crossing rate over (N, H, Q) prediction tensors.

These tests are measurement-only: they do not modify the model, the
loss, or any committed experiment result.
"""
import numpy as np

from ucra.core.uncertainty import UncertaintyState, extract_uncertainty

QUANTILES = [0.05, 0.25, 0.5, 0.75, 0.9, 0.99]


def count_quantile_crossings(pred: np.ndarray) -> int:
    """Count adjacent-quantile crossings in a (N, H, Q) prediction tensor."""
    pred = np.asarray(pred)
    if pred.shape[-1] < 2:
        return 0
    return int(np.sum(np.diff(pred, axis=-1) < 0))


def test_spread_guard_clamps_crossing_to_nonnegative():
    """Even if q99 crosses below q50, spread must never go negative."""
    # (horizon=4, Q=6); at h=0 the q99 entry is deliberately below q50
    pred = np.array([
        [10.0, 11.0, 12.0, 13.0, 14.0, 9.0],   # crossing at h=0
        [20.0, 21.0, 22.0, 23.0, 24.0, 30.0],  # ordered
        [30.0, 31.0, 32.0, 33.0, 34.0, 40.0],  # ordered
        [40.0, 41.0, 42.0, 43.0, 44.0, 50.0],  # ordered
    ])
    state = UncertaintyState()
    state.set_reference(np.full(32, 10.0))
    u = extract_uncertainty(pred, QUANTILES, state)
    assert u["u_hat"] == 12.0          # median passed through unchanged
    assert u["spread"] == 0.0          # negative width clamped to zero
    from ucra.core.transform import phi_transform
    r = phi_transform(u["u_hat"], u["spread"], 0.0, 1000.0,
                      {"base_tau": 0.9, "kappa": 0.5,
                       "rho_weight": 0.3, "min_headroom": 0.05,
                       "max_reserve_ratio": 0.95})
    assert r >= 12.0                   # floor (1+h)*u_hat keeps it sane


def test_crossing_counter_counts_and_clears():
    """The counter detects a planted crossing and accepts ordered bands."""
    pred = np.zeros((4, 3, len(QUANTILES)))
    pred[0, 0, 0] = 5.0                # q05 above q25 at exactly one point
    assert count_quantile_crossings(pred) == 1
    ordered = np.cumsum(np.ones((2, 2, len(QUANTILES))), axis=-1)
    assert count_quantile_crossings(ordered) == 0
    assert count_quantile_crossings(np.zeros((1, 1, 1))) == 0


def test_well_calibrated_bands_have_few_crossings():
    """Constructed-ordered bands give 0 crossings; planted noise gives >0."""
    rng = np.random.default_rng(7)
    base = rng.uniform(50, 100, size=(200, 4, 1))
    widths = np.abs(rng.normal(1.0, 0.2, size=(200, 4, len(QUANTILES) - 1)))
    good = np.concatenate([base, base + widths.cumsum(-1)], axis=-1)
    assert count_quantile_crossings(good) == 0
    bad = good.copy()
    bad[:, ::2, :] = bad[:, ::2, ::-1]     # reverse quantile order on half
    assert count_quantile_crossings(bad) > 0