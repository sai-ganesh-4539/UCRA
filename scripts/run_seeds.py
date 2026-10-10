#!/usr/bin/env python3
"""Three-seed verification study (repo doc 12, section 8).

For each seed in eval.seeds (default 41/42/43):
  1. train a fresh Stage-1 quantile LSTM into outputs/seed_<s>/
  2. run the causal closed loop (run_ucra.run_closed_loop, with kappa sweep)
  3. run the causal drift demo (demo_evolution.run_drift_demo)

Then aggregate mean +- (sample) sd across seeds for every metric into
outputs/seed_aggregate.json, and copy the median-seed run's figures nowhere
(the default seed-42 run in outputs/ remains the paper's reference run).

Every driver in the loop uses the CAUSAL replay intake (DelayedReplay), so
no seed ever fine-tunes on labels that were not yet observed.

Usage:  python scripts/run_seeds.py --config configs/default.yaml
Saves:  outputs/seed_<s>/{ucra_results,kappa_sweep,drift_demo,
        train_metrics}.json + outputs/seed_aggregate.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import copy
import yaml

from scripts.run_ucra import run_closed_loop
from scripts.demo_evolution import run_drift_demo
from scripts.train import train_and_save


def _flatten(obj, prefix=""):
    """Flatten nested dicts/lists of numbers into {path: value}."""
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list) and obj and all(
            isinstance(x, (int, float)) for x in obj):
        for i, v in enumerate(obj):
            out[f"{prefix}[{i}]"] = v
    elif isinstance(obj, (int, float)):
        out[prefix] = obj
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--seeds", type=int, nargs="*", default=None,
                    help="override the seed list (default: eval.seeds)")
    args = ap.parse_args()
    base_cfg = yaml.safe_load(open(args.config))
    seeds = args.seeds or base_cfg.get("eval", {}).get(
        "seeds", [41, 42, 43])
    print(f"[seeds] study seeds: {seeds}")

    per_seed = {}
    for s in seeds:
        cfg = copy.deepcopy(base_cfg)
        cfg["seed"] = s
        cfg["paths"]["outputs"] = str(
            Path(base_cfg["paths"]["outputs"]) / f"seed_{s}")
        print(f"\n===== seed {s} -> {cfg['paths']['outputs']} =====")

        tm = train_and_save(cfg, verbose=True)
        # the checkpoint now exists in this seed's outputs dir; the closed
        # loop will load it. Sweep always on -> kappa_sweep.json per seed.
        cfg["_do_sweep"] = True
        res, _log = run_closed_loop(cfg, make_figures=False, verbose=False)
        sweep = json.loads(
            (Path(cfg["paths"]["outputs"]) / "kappa_sweep.json").read_text())
        demo = run_drift_demo(cfg, shift=0.35, at=0.15, eval_every=24,
                              make_figures=False, verbose=False)

        per_seed[str(s)] = {
            "train": tm,
            "closed_loop": res,
            "kappa_sweep": sweep,
            "drift_demo": demo,
        }
        print(f"[seeds] seed {s}: ucra viol {res['ucra']['violation_rate']:.4f}"
              f" | util {res['ucra']['utilization']:.4f}"
              f" | demo updates {demo['evolving']['updates']}")

    # ---------- aggregate ----------
    flat = {s: _flatten(d) for s, d in per_seed.items()}
    keys = sorted(set().union(*[set(f) for f in flat.values()]))
    agg = {}
    for k in keys:
        vals = [flat[s][k] for s in flat if k in flat[s]]
        if len(vals) == len(seeds):
            agg[k] = {"mean": statistics.fmean(vals),
                      "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0,
                      "values": vals}
    out = {"seeds": list(seeds), "n_seeds": len(seeds),
           "causal_replay": True, "aggregate": agg, "per_seed": per_seed}
    dest = Path(base_cfg["paths"]["outputs"]) / "seed_aggregate.json"
    dest.write_text(json.dumps(out, indent=2))

    # ---------- readable summary ----------
    print("\n[seeds] aggregated (mean +- sd over seeds):")
    interesting = [
        ("ucra.violation_rate", "UCRA violation"),
        ("ucra.utilization", "UCRA utilization"),
        ("ucra.waste", "UCRA waste"),
        ("train.coverage_90pct", "q05-q99 coverage"),
        ("closed_loop.evolution_updates", "evolution updates (clean)"),
        ("drift_demo.frozen.viol_post_surge", "demo frozen viol(post-surge)"),
        ("drift_demo.evolving.viol_post_surge", "demo evolving viol(post)"),
        ("drift_demo.evolving.updates", "demo updates"),
    ]
    for path, label in interesting:
        a = agg.get(path)
        if a:
            print(f"  {label:34s} {a['mean']:.4f} +- {a['sd']:.4f}   {a['values']}")
    print(f"[seeds] saved {dest}")


if __name__ == "__main__":
    main()
