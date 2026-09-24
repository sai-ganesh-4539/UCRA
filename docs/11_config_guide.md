# 11 -- Configuration Guide: Every Knob in default.yaml

`configs/default.yaml` is the single source of truth. Nothing is hard-coded
in the modules. Below: every key, its default, what it does, and which way
to move it.

## seed: 42
Feeds torch/numpy RNG. Change it to quantify training variance (doc 12);
expect +-1-3 points of coverage across seeds/platforms.

## paths
| Key | Default | Meaning |
|---|---|---|
| raw_data | data | where downloads/unzips live |
| outputs | outputs | every artifact (doc 02 section 5) |

## data
| Key | Default | Notes |
|---|---|---|
| dataset | ran | "ran" (main results), "cttc" (per-sample series), "synthetic" (no download) |
| ran.subset | dataset01 | dataset01/02/03 -- 01 is the 8 MB one with 75 sectors |
| ran.target | data_volume | the demand series (DL+UL summed) |
| ran.agg | sum | sum over sectors ("mean" also supported) |
| ran.resample | null | series is already 15-min; a pandas freq string would regrid |
| cttc.samples_dir | data/cttc/slicing-simulations | unzipped archive location |
| cttc.slices | [eMBB, mMTC, URLLC] | informational; the loader keeps whatever the samples contain |
| synthetic.n_steps / n_sectors | 2016 / 8 | 3 weeks x 15-min x 8 sectors |

## model (Stage 1)
| Key | Default | Effect of changing it |
|---|---|---|
| seq_len | 96 | input history (24 h). Longer -> slower, more context; 96 matches the diurnal cycle |
| horizon | 4 | predicted steps. Only step 0 drives the loop today (doc 03 section 4) |
| hidden_size | 64 | capacity; 32 trains faster, 128 may overfit 1.6k windows |
| num_layers | 2 | keep 2 for dropout to exist (torch disables dropout on single-layer LSTM) |
| dropout | 0.1 | regularization |
| quantiles | [0.05 ... 0.99] | Phi uses 0.5 and 0.99; train.py coverage uses 0.05/0.99. Keep 2-decimal values (doc 10 gotcha) |
| batch_size | 64 | smaller -> noisier gradients, sometimes better quantiles |
| lr | 0.001 | Adam; Stage 5 fine-tune uses 0.3x this |
| max_epochs | 50 | early stopping usually fires first |
| patience | 6 | epochs without val improvement before stopping |
| train_val_test | [0.7, 0.15, 0.15] | chronological split fractions |

## phi (Stage 2) -- the identity of the method
| Key | Default | Effect |
|---|---|---|
| base_tau | 0.9 | which quantile the reservation sits on. 0.95 -> stricter, more slack; 0.75 -> leaner, more violations |
| kappa | 0.5 | THE risk dial (doc 04 section 3). 0 = bare quantile + floor |
| rho_weight | 0.3 | feedback authority. 0 disables risk adaptation in Phi (Stage 5 still fine-tunes the model) |
| min_headroom | 0.05 | floor = (1+this) * u_hat. Protects against overconfident forecasts |
| max_reserve_ratio | 0.95 | ceiling = this * capacity. Keep < 1.0 |

## allocation (Stage 3)
| Key | Default | Effect |
|---|---|---|
| pressure_exponent | 1.0 | >1 favors big consumers, <1 flattens (doc 05 section 2) |
| risk_exponent | 0.5 | shape of the risk premium; 0 disables |
| best_effort_share | 0.1 | capacity kept out of guarantees per slot |

## update (Stage 4)
| Key | Default | Effect |
|---|---|---|
| ewma_alpha | 0.3 | higher = snappier, noisier; lower = smoother, laggier. If violations cluster where R_target was already high, lower this |
| release_hysteresis | 0.1 | release requires >10% of CAPACITY idle AND a shrinking target. Lower = releases sooner |
| drift_window | 96 | rolling window for rho_t, plots, and trigger stats |

## evolution (Stage 5)
| Key | Default | Effect |
|---|---|---|
| violation_trigger | 0.15 | fine-tune when the 96-slot violation rate exceeds this. Lower = more sensitive (and more false positives) |
| drift_zscore | 3.0 | or when demand leaves 3 training sigmas |
| replay_size | 512 | buffer for continual learning (doc 06 section 3) |
| finetune_epochs | 3 | keep tiny on purpose; the demo --ft-epochs flag overrides |
| eval_every | 96 | trigger check cadence (24 h). The demo defaults to 24 slots instead |

## eval
| Key | Default | Meaning |
|---|---|---|
| metrics | [reservation_error, violation_rate, utilization, waste] | doc 08 |
| baselines | [static_peak, mean_forecast, oracle_quantile] | doc 08 |
| sweep_kappa | [0.0 ... 2.0] | the frontier grid for --sweep and CTTC |

## Tuning recipes

- **More conservative (carrier SLA mode):** base_tau 0.95, kappa 1.0,
  rho_weight 0.5. Expect utilization drop of ~5-10 points, still ~0
  violations.
- **Leaner (cloud cost mode):** kappa 0.25, min_headroom 0.02. Watch
  violations reappear in the sweep first.
- **Faster adaptation:** update.ewma_alpha 0.5, evolution.eval_every 24.
  Trade smoothness for responsiveness.
- **Sanity mode (no downloads):** data.dataset synthetic; the smoke test
  config in tests/ is an even smaller variant.
- After ANY config change, re-run audit -> (delete outputs/model.pt if the
  model config changed) -> train -> run_ucra --sweep. model.pt is only
  invalidated automatically when run_ucra itself trains; if you edited model
  hyperparameters but a checkpoint exists, DELETE it or you will silently
  evaluate the old architecture.
