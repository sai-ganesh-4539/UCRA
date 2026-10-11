"""Service-performance feedback tests (qos_gap helper).

The architecture diagram's Feedback & Evaluation block lists "Service
Performance (latency / reliability)". In the CTTC evaluation this maps to
ucra.eval.metrics.qos_gap: observed packet-drop ratio and average delay,
split by each policy's reservation-violation state.

Covers:
  Q1 mixed groups      -> correct means and group sizes
  Q2 all violated      -> None (no honored group to compare against)
  Q3 none violated     -> None (zero-violation policy, e.g. UCRA)
  Q4 JSON-safe types   -> plain int/float only
  Q5 input frames not mutated
"""
import numpy as np
import pandas as pd

from ucra.eval.metrics import qos_gap


def _frame(offered, drops, delay):
    return pd.DataFrame({"offered": offered, "drops_ratio": drops,
                         "avg_delay": delay})


def test_qos_gap_mixed_groups():
    # reservation R=100: offered 150, 120 violate; 80, 90 are honored
    sub = _frame([150.0, 120.0, 80.0, 90.0],
                 [0.10, 0.20, 0.02, 0.04],
                 [5.0, 7.0, 1.0, 2.0])
    g = qos_gap(sub, np.array([100.0, 100.0, 100.0, 100.0]))
    assert g is not None
    assert g["n_violated"] == 2 and g["n_ok"] == 2
    assert abs(g["drops_violated"] - 0.15) < 1e-9
    assert abs(g["drops_ok"] - 0.03) < 1e-9
    assert abs(g["delay_violated"] - 6.0) < 1e-9
    assert abs(g["delay_ok"] - 1.5) < 1e-9


def test_qos_gap_all_violated_returns_none():
    sub = _frame([150.0, 120.0], [0.1, 0.2], [5.0, 7.0])
    assert qos_gap(sub, np.array([100.0, 100.0])) is None


def test_qos_gap_no_violations_returns_none():
    # zero-violation policy (e.g. UCRA) -> nothing to compare against
    sub = _frame([80.0, 90.0], [0.02, 0.04], [1.0, 2.0])
    assert qos_gap(sub, np.array([100.0, 100.0])) is None


def test_qos_gap_json_safe_types():
    sub = _frame([150.0, 80.0], [0.1, 0.02], [5.0, 1.0])
    g = qos_gap(sub, np.array([100.0, 100.0]))
    assert isinstance(g["n_violated"], int)
    assert isinstance(g["n_ok"], int)
    for k in ("drops_violated", "drops_ok", "delay_violated", "delay_ok"):
        assert isinstance(g[k], float)


def test_qos_gap_does_not_mutate_inputs():
    sub = _frame([150.0, 80.0], [0.1, 0.02], [5.0, 1.0])
    before = sub.copy()
    qos_gap(sub, np.array([100.0, 100.0]))
    assert sub.equals(before)
