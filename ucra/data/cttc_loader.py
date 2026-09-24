"""Loader for Zenodo 10610616 - CTTC B5G Network Slicing Dataset
(Farreras et al., Data in Brief 55:110738, 2024, CC-BY-4.0).

Each sample is a steady-state snapshot of a slicing simulation:
  - traffic_matrix : per-flow offered demand
  - slices         : eMBB / mMTC / URLLC with `delta` = reserved share
  - performance_matrix : PktsDrop, AvgDelay, p10-p90, Jitter per src-dst
  - topology_object: networkx graph (link bandwidth = capacity)

The loader converts samples into a slice-level frame:

    [sample, slice_type, reserved, offered, drops_ratio, avg_delay]

where `reserved = delta * link_capacity`. This is what UCRA's Stages 2-3
evaluation uses (reservation sizing vs observed violations/delays).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


def _ensure_datanetapi(samples_dir: Path) -> None:
    """Make the dataset's own datanetAPI.py importable (downloads nothing)."""
    if str(samples_dir) not in sys.path:
        sys.path.insert(0, str(samples_dir))


def load_cttc(samples_dir: str | Path = "data/cttc/slicing-simulations",
              max_samples: int | None = None, verbose: bool = True) -> pd.DataFrame:
    """Iterate datanetAPI samples and build the slice-level frame."""
    samples_dir = Path(samples_dir)
    if not samples_dir.exists():
        raise FileNotFoundError(
            f"{samples_dir} not found. Run: python scripts/download_data.py --cttc")
    _ensure_datanetapi(samples_dir)

    try:
        from datanetAPI import DatanetAPI  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "datanetAPI.py must sit inside the unzipped dataset folder "
            "(scripts/download_data.py --cttc places it there)") from e

    reader = DatanetAPI(str(samples_dir))
    rows = []
    for i, sample in enumerate(reader):
        if max_samples is not None and i >= max_samples:
            break
        try:
            topo = sample.get_topology_object()
            caps = [d.get("bandwidth", 0) for _, _, d in topo.edges(data=True)]
            link_cap = float(np.mean(caps)) if caps else 0.0

            perf = sample.get_performance_matrix()
            n = sample.get_network_size()

            for sid in range(len(sample.get_slices())):
                stype = str(sample.get_slice_type(sid))
                delta = float(sample.get_slice_delta(sid))
                flows = sample.get_slice_flows(sid)
                offered = 0.0
                for fl in (flows if flows is not None else []):
                    offered += float(getattr(fl, "avgRate", 0) or 0) \
                        + float(getattr(fl, "rate", 0) or 0)
                # aggregate drop/delay over all src-dst pairs weighted equally
                drops, delays, cells = [], [], 0
                for s in range(n):
                    for d in range(n):
                        try:
                            info = perf[s, d]
                            agg = info["AggInfo"] if isinstance(info, dict) \
                                else getattr(info, "AggInfo", None)
                            if agg is None:
                                continue
                            drops.append(float(agg.get("PktsDrop", 0) if isinstance(agg, dict)
                                               else getattr(agg, "PktsDrop", 0)))
                            delays.append(float(agg.get("AvgDelay", 0) if isinstance(agg, dict)
                                                else getattr(agg, "AvgDelay", 0)))
                            cells += 1
                        except Exception:
                            continue
                rows.append({
                    "sample": i, "slice_type": stype, "delta": delta,
                    "reserved": delta * link_cap, "offered": offered,
                    "link_cap": link_cap,
                    "drops_ratio": (float(np.mean(drops)) if drops else np.nan),
                    "avg_delay": (float(np.mean(delays)) if delays else np.nan),
                })
        except Exception as exc:  # skip corrupt samples, keep going
            if verbose:
                print(f"[cttc_loader] sample {i} skipped: {exc}")
            continue
        if verbose and (i + 1) % 25 == 0:
            print(f"[cttc_loader] processed {i + 1} samples ...")

    df = pd.DataFrame(rows)
    if verbose and len(df):
        print(f"[cttc_loader] frame: {df.shape}, slices per type: "
              f"{df['slice_type'].value_counts().to_dict()}")
    return df
