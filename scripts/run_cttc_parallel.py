#!/usr/bin/env python3
"""Parallel CTTC evaluation: parse the tar archives with a worker pool
(the sequential loader needs ~17 min for 250 samples; 2-3 workers bring it
under the runtime budget), merge the per-archive frames, then run the exact
same causal policy evaluation as run_cttc_eval.evaluate_policies.

Usage:  python scripts/run_cttc_parallel.py --config configs/default.yaml
Saves:  data/cttc/frame_part_<k>.csv + outputs/cttc_results.json
        outputs/fig_cttc_slices.png
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import yaml

from ucra.data.cttc_loader import load_cttc

SAMPLES_DIR = "data/cttc/slicing-simulations"
MAX_TOTAL = 250   # same scope as the reference run


def _parse_tar(task):
    """Worker: parse one archive into a frame, cache it as CSV immediately."""
    idx, tar_name, max_samples = task
    cache = Path(f"data/cttc/frame_part_{idx}.csv")
    if cache.exists():
        return idx, tar_name, pd.read_csv(cache)
    df = load_cttc(SAMPLES_DIR, max_samples=max_samples,
                   verbose=False, tars=[tar_name])
    df.to_csv(cache, index=False)
    return idx, tar_name, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--max-samples", type=int, default=MAX_TOTAL)
    ap.add_argument("--workers", type=int, default=2)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out = Path(cfg["paths"]["outputs"])
    out.mkdir(exist_ok=True)

    # which archives does DatanetAPI read, in its order?
    sys.path.insert(0, SAMPLES_DIR)
    from datanetAPI import DatanetAPI
    reader = DatanetAPI(SAMPLES_DIR)
    tar_order = [f for _r, f in reader.get_available_files()]
    print(f"[cttc-par] {len(tar_order)} archives; taking first "
          f"{args.max_samples} samples")

    # distribute: walk archives in order, each gets the remaining quota
    quota = args.max_samples
    tasks = []
    for t in tar_order:
        if quota <= 0:
            break
        take = min(100, quota)     # archives hold ~100 samples each
        tasks.append((len(tasks), t, take))
        quota -= take
    cached = [t for i, t, _n in tasks
              if Path(f"data/cttc/frame_part_{i}.csv").exists()]
    todo = [t for t in tasks
            if not Path(f"data/cttc/frame_part_{t[0]}.csv").exists()]
    print(f"[cttc-par] tasks: {tasks} | cached: {len(cached)} | todo: {len(todo)}")

    if todo:
        with mp.Pool(min(args.workers, len(todo))) as pool:
            done = pool.map(_parse_tar, todo)
        print(f"[cttc-par] parsed {len(done)} archives this run")

    results = [(i, t, pd.read_csv(f"data/cttc/frame_part_{i}.csv"))
               for i, t, _n in tasks]

    frames = []
    offset = 0
    for idx, tar_name, df in results:
        take = df["sample"].max() + 1 if len(df) else 0
        print(f"[cttc-par] {tar_name}: {df.shape}")
        df = df.copy()
        df["sample"] = df["sample"] + offset   # sequential numbering across tars
        offset += int(take)
        frames.append(df)
    full = pd.concat(frames, ignore_index=True)
    print(f"[cttc-par] merged frame: {full.shape}, "
          f"{full['sample'].nunique()} samples")

    from scripts.run_cttc_eval import evaluate_policies
    evaluate_policies(full, cfg, out, train_frac=0.7)


if __name__ == "__main__":
    main()
