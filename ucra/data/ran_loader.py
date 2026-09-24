"""Loader for Zenodo 17815388 - Performance Management Counters from Live
5G/4G/2G RAN (Lehoczky et al., Scientific Data 2026, CC-BY-4.0).

Verified real schema (Dataset_01/Baseband_02/):
    Base station, Sector, Timestamp,
    <2G|4G|5G> max active users DL/UL, <tech> data volume DL/UL,
    <tech> max RRC users, <tech> RB utilization, <tech> CQI rank 1..4,
    <tech> RRC users, <tech> active users UL/DL, <tech> MIMO rank DL

One CSV per radio technology per baseband; sectors measured every 15 min.
The loader returns a canonical long frame:

    [ts, sector, data_volume, utilization, active_users, tech]
"""
from __future__ import annotations

import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd


def _extract_if_needed(root: Path) -> None:
    for z in sorted(root.glob("Dataset_*.zip")):
        out = root / z.stem
        if not out.exists():
            print(f"[ran_loader] extracting {z.name} ...")
            with zipfile.ZipFile(z) as f:
                f.extractall(root)


def _map_columns(cols: list[str]) -> dict | None:
    """Build the canonical mapping for one file's header (tech-prefix aware)."""
    tech = None
    for c in cols:
        m = re.match(r"^(2G|4G|5G)\s", str(c))
        if m:
            tech = m.group(1)
            break
    if tech is None:
        return None
    need = {
        "vol_dl": f"{tech} data volume DL",
        "vol_ul": f"{tech} data volume UL",
        "util": f"{tech} RB utilization",
        "users_dl": f"{tech} active users DL",
        "users_ul": f"{tech} active users UL",
    }
    missing = [k for k, v in need.items() if v not in cols]
    if missing:
        return None
    return {"tech": tech, **need}


def load_ran(data_dir: str | Path = "data/ran", subset: str = "dataset01",
             verbose: bool = True) -> pd.DataFrame:
    """Load one RAN subset into the canonical long-format frame.

    Returns DataFrame [ts, sector, data_volume, utilization, active_users, tech].
    """
    root = Path(data_dir)
    if not root.exists() or not any(root.iterdir()):
        raise FileNotFoundError(
            f"No dataset files under {root}. Run: bash scripts/download_data.sh")
    _extract_if_needed(root)

    if subset[-1].isdigit():
        scope = root / f"Dataset_0{subset[-1]}"
        csvs = sorted(scope.rglob("*.csv")) if scope.exists() else sorted(root.rglob("*.csv"))
    else:
        csvs = sorted(root.rglob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files under {root}")
    if verbose:
        print(f"[ran_loader] {len(csvs)} CSV file(s)")

    frames = []
    for csv in csvs:
        df = pd.read_csv(csv, engine="c", low_memory=False)
        mapping = _map_columns(list(df.columns))
        if mapping is None:
            if verbose:
                print(f"[ran_loader] skip (no tech columns): {csv.name}")
            continue
        tech = mapping["tech"]
        keep = pd.DataFrame({
            "ts": pd.to_datetime(df["Timestamp"], errors="coerce"),
            "sector": df["Base station"].astype(str) + "_S" + df["Sector"].astype(str),
            "data_volume": pd.to_numeric(df[mapping["vol_dl"]], errors="coerce").fillna(0)
                         + pd.to_numeric(df[mapping["vol_ul"]], errors="coerce").fillna(0),
            "utilization": pd.to_numeric(df[mapping["util"]], errors="coerce"),
            "active_users": pd.to_numeric(df[mapping["users_dl"]], errors="coerce").fillna(0)
                          + pd.to_numeric(df[mapping["users_ul"]], errors="coerce").fillna(0),
            "tech": tech,
        }).dropna(subset=["ts"])
        frames.append(keep)
        if verbose:
            print(f"[ran_loader] {csv.name}: {len(keep):,} rows [{tech}]")

    if not frames:
        raise RuntimeError("No usable CSV matched the expected PM-counter schema.")
    out = pd.concat(frames, ignore_index=True)
    if verbose:
        print(f"[ran_loader] canonical frame: {out.shape} | "
              f"sectors={out['sector'].nunique()} | techs={sorted(out['tech'].unique())} | "
              f"span {out['ts'].min()} -> {out['ts'].max()}")
    return out


def load_demand_series(df: pd.DataFrame, agg: str = "sum",
                       freq: str | None = None,
                       keep: str = "longest") -> pd.Series:
    """Aggregate sectors (and techs) into a network demand series.

    The live-PM data comes in separate observation campaigns separated by
    multi-day gaps. Windows must never cross those gaps, so after resampling
    to a strict 15-min grid the series is split into contiguous segments and
    (keep='longest') only the longest gap-free segment is returned.
    """
    g = df.groupby("ts")["data_volume"]
    s = g.sum() if agg == "sum" else g.mean()
    s = s.sort_index().rename("demand").astype(float)
    if freq:
        s = s.asfreq(freq)

    # split at gaps > one slot; keep longest contiguous segment
    idx = s.index.to_series()
    breaks = idx.diff() > pd.Timedelta("15min")
    seg_id = breaks.cumsum()
    if keep == "longest":
        best = s.groupby(seg_id).count().idxmax()
        s = s[seg_id == best]
    # small internal holes (up to 4 slots) get interpolated
    s = s.asfreq("15min").interpolate(limit=4, limit_direction="both")
    return s
