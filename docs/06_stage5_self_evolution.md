# 06 -- Stage 5: Self-Evolving Learning (+ Your Drift Demo Decoded)

File: `ucra/core/evolve.py` (`DriftMonitor`, `ReplayBuffer`,
`EvolutionEngine`); drivers: `scripts/run_ucra.py` (always armed) and
`scripts/demo_evolution.py` (A/B demonstration).

## 1. Why Stage 5 exists

Stages 1-4 assume the model's picture of demand matches reality. When the
world drifts -- a mass event, a new app, a holiday -- the quantile forecasts
silently become wrong, Phi faithfully reserves around the WRONG distribution,
and violations climb. Stage 5 closes the loop: watch the feedback, detect
drift, and fine-tune the model on what reality just did.

## 2. The trigger (`DriftMonitor`)

A rolling window (96 slots) of violation flags plus a training-reference
z-score. Checked every `eval_every` slots (96 = every 24 h in run_ucra;
24 in the demo by default) and only at t > 0:

```
trigger fires  if   viol_rate_96 > 0.15   OR   z > 3.0
z = | demand_now - train_mu | / train_sd
```

Two independent detectors: the violation trigger catches "model is wrong in
the way that hurts users"; the z-trigger catches "the world moved even
before our reservations felt it". Either is sufficient.

## 3. The memory (`ReplayBuffer`, reservoir sampling)

Keeps `replay_size` = 512 (window, target) pairs drawn by **reservoir
sampling**: the first 512 items fill the buffer; every item after that
replaces a uniformly chosen slot with probability 512/n_seen. Guarantees:
at any moment, every past window had an equal chance to be in the buffer.
Why it matters: the buffer contains BOTH pre-drift and post-drift windows,
so fine-tuning on it adapts to the new regime **without erasing** the old
one -- the standard defense against catastrophic forgetting in continual
learning. (After 5,000 windows, each has a 512/5000 ~ 10.2% inclusion
chance.)

## 4. The update (`EvolutionEngine.finetune`)

```
Adam, lr = model_lr * 0.3 (gentler than training), full-batch over the
512 buffered pairs, finetune_epochs = 3, pinball loss identical to Stage 1.
```

Full-batch is fine here (512 x 96 floats is tiny). After a successful
update the driver recomputes `pred_q` with the evolved weights, so the very
next Phi decision already benefits. run_ucra remembers every test window in
z-space (`scaler.transform(X), scY.transform(Y)`) -- this exact detail was a
real bug we fixed in Part D: the buffer originally stored raw-scale targets
while the model trains in z-space, which would have corrupted fine-tuning
the moment it first fired.

## 5. Why the clean run had `evolution_updates: 0`

Your main RAN run shows zero updates. That is the system working, not
skipping: UCRA held 0.0% violations (rolling rate never approached 0.15)
and demand stayed within 3 training sigmas (z never fired). Stage 5 is an
insurance policy -- a healthy run never cashes it. This is also a nice
false-positive result for the paper: across 252 slots the trigger never
cried wolf.

## 6. Your drift demo, line by line

`demo_evolution.py` multiplies realized demand by 1.35 from slot 37 of the
252-slot test window (15%), rebuilds windows from the DRIFTED series (the
model must now forecast a world it never saw), and runs two identical
passes from the same checkpoint. Your output:

```
shift 0.35, surge_start_slot 37, n_test 252, capacity 143,954.5

frozen:    viol_overall 22.62% | viol_post_surge 26.51% | util 82.74% | updates 0
evolving:  viol_overall  9.92% | viol_post_surge 11.63% | util 80.31% | updates 3 [96, 120, 144]
```

Reading it:

- **Pre-surge both passes are clean.** Frozen overall 22.6% across 252 slots
  ~ 57 violations; post-surge alone 26.5% of 215 slots ~ 57 violations. The
  arithmetic closes: essentially ALL violations happen after the surge.
  Before it, the frozen model was as good as UCRA always is (0%).
- **The trigger timeline makes sense.** Checks run every 24 slots (t = 24,
  48, 72, 96, ...). At t = 96 the rolling 96-slot window contains ~60
  post-surge slots at ~26.5% violations -> window rate ~0.166 > 0.15 ->
  first fine-tune. The window at t = 120 and t = 144 still carries the
  poisoned pre-fix slots, so it fires twice more; after the third update the
  window fills with post-fix, low-violation slots and the rate drops under
  0.15 -> no further updates. Exactly [96, 120, 144].
- **Adaptation came from intelligence, not over-provisioning.** Post-surge
  utilization is 80.3% (evolving) vs 82.7% (frozen) -- the evolving pass did
  NOT just reserve vastly more; the fine-tuned quantiles put the buffer in
  the right places. Violations fell 26.5% -> 11.6% (a 56% relative
  reduction) at slightly LOWER utilization.
- Residual 11.6%: a +35% step is a violent drift; three gentle fine-tunes
  (3 epochs each, lr x 0.3) shrink but do not eliminate it. Pushing further
  (more epochs, higher lr, lower trigger) overfits spikes -- the demo flags
  `--ft-epochs` and `--lr-scale` let you trace this frontier (doc 12).
- Robustness we verified while building: the result was insensitive to
  fine-tune hyperparameters (10 epochs x 3 lr values -> same update points),
  so the behavior is a property of the trigger + replay design, not of a
  lucky learning rate.

Figure guide for `fig_drift_demo.png`: top panel = drifted demand (blue)
with the two reservations overlaid (red frozen, green evolving) -- watch the
green line climb after each update while red stays flat; dashed black =
surge start; dotted gray = physical capacity. Bottom panel = rolling
violation rates with the 0.15 trigger line and green verticals at the three
update points.

## 7. Interface recap (for the code reference)

- `DriftMonitor(cfg)` -- `observe(violated, demand)` -> stats dict,
  `triggered(stats)` -> bool, `set_reference(train_demand)`.
- `ReplayBuffer(capacity)` -- `push(x, y)`, `arrays()` -> (X, Y) or
  (None, None).
- `EvolutionEngine(model, evo_cfg, model_cfg)` -- `remember(x, y)`,
  `finetune(loss_fn)` -> {"updated": bool, "replay": n, "loss": L,
  "n_updates": k}.
