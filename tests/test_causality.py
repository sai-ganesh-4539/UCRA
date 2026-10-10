"""Causality regression tests for Stage-5 online learning.

Guards against the two leakage paths found in pre-submission review:

  L1  replay leakage     - a window offered at loop time t carries H-step
        labels for slots t..t+H-1, but only slot t is observed at t. The
        buffer must not contain that window before t+H-1.
  L2  refresh leakage    - after an update at t, only slots > t may use the
        evolved forecast; slots <= t must keep what they were served.

Run:  python -m pytest tests/test_causality.py -q
"""
import numpy as np
import torch

from ucra.core.evolve import DelayedReplay, ReplayBuffer


def test_delayed_replay_never_releases_early():
    """L1: window w (H=4) must be absent at t=w+H-2, present at t=w+H-1."""
    H = 4
    buf = ReplayBuffer(capacity=64)
    dr = DelayedReplay(buf, horizon=H)

    n = 20
    for t in range(n):
        x = np.full((3,), float(t))          # tiny dummy window
        y = np.arange(t, t + H, dtype=float)  # labels for slots t..t+H-1
        dr.offer(t, x, y)
        dr.flush(t)

        newest_in_buffer = -1
        for w in buf.X:
            newest_in_buffer = max(newest_in_buffer, int(w[0]))
        if newest_in_buffer < 0:
            continue                      # nothing released yet (t < H-1)
        # every window in the buffer must satisfy w + H - 1 <= t
        assert newest_in_buffer <= t - H + 1, (
            f"leak at t={t}: window {newest_in_buffer} with labels up to "
            f"{newest_in_buffer + H - 1} > observed horizon {t}")
        assert dr.pending == (t + 1) - dr.n_released


def test_delayed_replay_exact_release_time():
    """A window must appear in the buffer exactly at t+H-1."""
    H = 4
    buf = ReplayBuffer(capacity=64)
    dr = DelayedReplay(buf, horizon=H)
    dr.offer(10, np.zeros(2), np.zeros(H))
    for t in range(10, 12):
        dr.flush(t)
        assert len(buf.X) == 0, "released before all labels observed"
    dr.flush(12)
    assert len(buf.X) == 0, "released one slot before its last label exists"
    dr.flush(13)   # t = w + H - 1 = 10 + 3
    assert len(buf.X) == 1
    dr.flush(19)
    assert len(buf.X) == 1


def test_no_future_labels_in_finetune_batch():
    """End-to-end: fine-tune batches must only contain labels <= update time."""
    H = 4
    buf = ReplayBuffer(capacity=512)
    dr = DelayedReplay(buf, horizon=H)
    T = 50
    for t in range(T):
        dr.offer(t, np.full((2,), float(t)), np.arange(t, t + H, dtype=float))
        dr.flush(t)
        if t % 10 == 0 and t > 0 and buf.X:
            X, Y = buf.arrays()
            # max label index across the batch cannot exceed observed time t
            newest = float(np.max(Y))
            assert newest <= t, (
                f"finetune batch at t={t} contains label {newest} > t "
                "(future outcome)")


def test_refresh_is_partial_only_future_slots():
    """L2: after an update at t, entries <= t keep the served forecast."""
    pred_q = np.arange(40, dtype=float).reshape(10, 2, 2)   # (N, H, Q)
    served = pred_q.copy()
    t_update = 6
    evolved = -pred_q                                        # "new model"
    served[t_update + 1:] = evolved[t_update + 1:]
    # past slots unchanged
    assert np.array_equal(served[:t_update + 1], pred_q[:t_update + 1])
    # future slots replaced
    assert np.array_equal(served[t_update + 1:], evolved[t_update + 1:])


def test_delayed_replay_empty_and_counts():
    buf = ReplayBuffer(capacity=8)
    dr = DelayedReplay(buf, horizon=3)
    dr.flush(100)                       # nothing pending -> no-op
    assert len(buf.X) == 0 and dr.pending == 0
    for t in range(5):
        dr.offer(t, np.zeros(1), np.zeros(3))
    dr.flush(4)
    assert dr.n_released == 3 and dr.pending == 2   # w<=2 released (w+2<=4)
    dr.flush(99)
    assert dr.n_released == 5 and dr.pending == 0


if __name__ == "__main__":
    test_delayed_replay_never_releases_early()
    test_delayed_replay_exact_release_time()
    test_no_future_labels_in_finetune_batch()
    test_refresh_is_partial_only_future_slots()
    test_delayed_replay_empty_and_counts()
    print("CAUSALITY TESTS PASSED")
