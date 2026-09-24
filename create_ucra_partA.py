#!/usr/bin/env python3
# ============================================================
# UCRA — Part A/2 — the ucra package (loaders, model, core, eval)
# Usage: run INSIDE your existing UCRA folder:  python create_ucra_partA.py
# ============================================================
from pathlib import Path

FILES = {}

FILES["ucra/__init__.py"] = r'''"""UCRA - Uncertainty-to-Reservation Transformation Algorithm.

A self-evolving, risk-adaptive resource-allocation framework for 5G/B5G
network slicing, built on:

  Dataset 1  CTTC B5G Network Slicing (Zenodo 10610616, CC-BY-4.0)
  Dataset 2  Performance Management Counters from Live 5G/4G/2G RAN
             (Zenodo 17815388, CC-BY-4.0; Scientific Data, 2026)
  Optional   Liverpool 5G High-Density Demand Dataset (failsafe / stress)

Pipeline stages (see docs/METHODOLOGY.md):
  1. Uncertainty Estimation            -> ucra.models.quantile_lstm
  2. Uncertainty-to-Reservation (Phi)  -> ucra.core.transform
  3. Risk-Adaptive Allocation          -> ucra.core.allocate
  4. Reservation Update & Release      -> ucra.core.transform / evolve
  5. Self-Evolving Learning            -> ucra.core.evolve
"""

__version__ = "0.1.0"
'''

FILES["ucra/data/__init__.py"] = r'''"""Dataset dispatch: config -> canonical demand frame."""
from __future__ import annotations

from .synthetic import generate_synthetic, synthetic_capacity


def load_dataset(cfg: dict, verbose: bool = True):
    """Return (demand_series, extras) for cfg['data']['dataset'].

    demand_series is a float pd.Series indexed by timestamp with the network
    demand per 15-min slot. extras may carry per-slice frames (CTTC) or the
    sector-level frame (RAN) for downstream analysis.
    """
    name = cfg["data"]["dataset"]

    if name == "synthetic":
        scfg = cfg["data"]["synthetic"]
        df = generate_synthetic(scfg["n_steps"], scfg["n_sectors"], cfg["seed"])
        series = df.groupby("ts")["demand"].sum().sort_index()
        series.rename("demand", inplace=True)
        cap = synthetic_capacity(scfg["n_sectors"])
        return series, {"sector_frame": df, "capacity": cap}

    if name == "ran":
        from .ran_loader import load_ran, load_demand_series
        rcfg = cfg["data"]["ran"]
        df = load_ran(cfg["paths"]["raw_data"] + "/ran", rcfg["subset"], verbose)
        series = load_demand_series(df, rcfg.get("agg", "sum"), rcfg.get("resample"))
        cap = float(series.max() * 1.3)  # headroom factor over observed peak
        return series, {"sector_frame": df, "capacity": cap}

    if name == "cttc":
        from .cttc_loader import load_cttc
        ccfg = cfg["data"]["cttc"]
        sl = load_cttc(ccfg["samples_dir"], verbose=verbose)
        series = sl.groupby("sample")["offered"].sum().sort_index()
        series.index = pd.RangeIndex(len(series))
        series.rename("demand", inplace=True)
        cap = float(sl["link_cap"].max()) if len(sl) else 0.0
        return series, {"slice_frame": sl, "capacity": cap}

    raise ValueError(f"unknown dataset '{name}' (use ran|cttc|synthetic)")
'''

FILES["ucra/data/synthetic.py"] = r'''"""Synthetic bursty demand generator (UCRA smoke tests & demos).

Produces a 15-min-resolution multi-sector demand series with:
- diurnal + weekly seasonality
- bursty ON/OFF surges (PPBP-flavoured)
- AR(1) noise
so the full UCRA pipeline can run without the Zenodo downloads.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic(n_steps: int = 2016, n_sectors: int = 8,
                       seed: int = 42, base: float = 100.0) -> pd.DataFrame:
    """Return long-format DataFrame [ts, sector, demand] at 15-min resolution."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2026-01-01", periods=n_steps, freq="15min")
    t = np.arange(n_steps)

    diurnal = 1.0 + 0.45 * np.sin(2 * np.pi * t / 96 - np.pi / 2)      # daily
    weekly = 1.0 + 0.15 * np.sin(2 * np.pi * t / (96 * 7))             # weekly
    noise = np.zeros(n_steps)
    for i in range(1, n_steps):                                        # AR(1)
        noise[i] = 0.85 * noise[i - 1] + rng.normal(0, 0.05)

    frames = []
    for s in range(n_sectors):
        phase = rng.uniform(0, 2 * np.pi)
        sector_gain = rng.uniform(0.7, 1.3)
        bursts = np.zeros(n_steps)
        on = 0
        for i in range(n_steps):                                       # ON/OFF bursts
            if on == 0 and rng.random() < 0.01:
                on = int(rng.integers(2, 10))
            if on > 0:
                bursts[i] = rng.uniform(0.5, 1.4)
                on -= 1
        demand = base * sector_gain * diurnal * weekly * (
            1.0 + noise + 0.35 * np.sin(t / 96 + phase) + bursts)
        demand = np.clip(demand, 1.0, None)
        frames.append(pd.DataFrame({
            "ts": ts, "sector": f"sec{s:02d}", "demand": demand.round(3)}))

    return pd.concat(frames, ignore_index=True)


def synthetic_capacity(n_sectors: int = 8, per_sector: float = 250.0) -> float:
    """Network capacity used for reservation clipping (sum over sectors)."""
    return per_sector * n_sectors
'''

FILES["ucra/data/ran_loader.py"] = r'''"""Loader for Zenodo 17815388 - Performance Management Counters from Live
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
'''

FILES["ucra/data/cttc_loader.py"] = r'''"""Loader for Zenodo 10610616 - CTTC B5G Network Slicing Dataset
(Farreras et al., Data in Brief 55:110738, 2024, CC-BY-4.0).

Each sample is a steady-state snapshot of a slicing simulation:
  - traffic_matrix : per-flow offered demand
  - slices         : eMBB / mMTC / URLLC with delta = reserved share
  - performance_matrix : PktsDrop, AvgDelay, p10-p90, Jitter per src-dst
  - topology_object: networkx graph (link bandwidth = capacity)

The loader converts samples into a slice-level frame:

    [sample, slice_type, reserved, offered, drops_ratio, avg_delay]

where reserved = delta * link_capacity. This is what UCRA's Stages 2-3
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
            f"{samples_dir} not found. Run: bash scripts/download_data.sh cttc")
    _ensure_datanetapi(samples_dir)

    try:
        from datanetAPI import datanetAPI  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "datanetAPI.py must sit inside the unzipped dataset folder "
            "(it ships with Zenodo 10610616)") from e

    reader = datanetAPI(str(samples_dir))
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
'''

FILES["ucra/features/__init__.py"] = r'''"""UCRA.features package."""
'''

FILES["ucra/features/windowing.py"] = r'''"""Windowing, chronological splits and scaling (Stage 1 preprocessing)."""
from __future__ import annotations

import numpy as np


def make_windows(series, seq_len: int, horizon: int):
    """Sliding windows from a 1-D series.

    Returns X (N, seq_len), Y (N, horizon) float32 arrays.
    """
    arr = np.asarray(series, dtype=np.float32)
    if len(arr) < seq_len + horizon:
        raise ValueError(f"series too short: {len(arr)} < {seq_len}+{horizon}")
    X, Y = [], []
    for i in range(len(arr) - seq_len - horizon + 1):
        X.append(arr[i:i + seq_len])
        Y.append(arr[i + seq_len:i + seq_len + horizon])
    return np.stack(X), np.stack(Y)


def chronological_split(n: int, fractions=(0.7, 0.15, 0.15)):
    """Return (train_idx, val_idx, test_idx) WITHOUT shuffling (time series!)."""
    a = int(n * fractions[0])
    b = int(n * (fractions[0] + fractions[1]))
    return np.arange(0, a), np.arange(a, b), np.arange(b, n)


class Scaler:
    """z-score scaler fitted on the TRAIN slice only (avoids leakage)."""

    def __init__(self):
        self.mu = 0.0
        self.sd = 1.0

    def fit(self, X: np.ndarray):
        self.mu = float(X.mean())
        self.sd = float(X.std() + 1e-8)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mu) / self.sd

    def inverse(self, X: np.ndarray) -> np.ndarray:
        return X * self.sd + self.mu
'''

FILES["ucra/models/__init__.py"] = r'''"""UCRA.models package."""
'''

FILES["ucra/models/quantile_lstm.py"] = r'''"""Quantile LSTM - Stage 1 (Uncertainty Estimation).

One network, direct quantile heads: input window -> {horizon x n_quantiles}.
Trained with the pinball (quantile) loss so the heads estimate the full
predictive distribution: q05 ... q99. ucra/core/uncertainty.py then turns
these into u_t (point), U_t (band width) and rho_t (risk indicator).

IMPORTANT: train in z-scored space (scale targets too). Pinball gradients
are bounded (~tau), so original-scale targets in the tens of thousands make
training stall. Invert the scaler after prediction.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class QuantileLSTM(nn.Module):
    def __init__(self, seq_len: int, horizon: int, quantiles: list[float],
                 hidden_size: int = 64, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.horizon = horizon
        self.quantiles = quantiles
        self.lstm = nn.LSTM(1, hidden_size, num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden_size, horizon * len(quantiles))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x.unsqueeze(-1))          # (B, L, H)
        last = out[:, -1, :]                         # (B, H)
        q = self.head(last)                          # (B, horizon*Q)
        return q.view(-1, self.horizon, len(self.quantiles))


def make_loss(quantiles: list[float]):
    """Build the pinball loss bound to the model's quantile levels."""
    taus = torch.tensor(quantiles, dtype=torch.float32).view(1, 1, -1)

    def loss_fn(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        # y_pred (B,H,Q), y_true (B,H) -> broadcast
        e = y_true.unsqueeze(-1) - y_pred
        return torch.mean(torch.maximum(taus * e, (taus - 1.0) * e))

    return loss_fn


def train_model(model: QuantileLSTM, Xtr, Ytr, Xva, Yva, cfg: dict,
                verbose: bool = True) -> dict:
    """Standard early-stopped training. Returns history dict."""
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)
    loss_fn = make_loss(cfg["quantiles"])
    opt = torch.optim.Adam(model.parameters(), lr=cfg["lr"])

    tr = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(Ytr)),
                    batch_size=cfg["batch_size"], shuffle=True)
    va = DataLoader(TensorDataset(torch.tensor(Xva), torch.tensor(Yva)),
                    batch_size=cfg["batch_size"])

    best_val, best_state, patience, hist = float("inf"), None, 0, {"train": [], "val": []}
    for ep in range(1, cfg["max_epochs"] + 1):
        model.train()
        tl = 0.0
        for xb, yb in tr:
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            l = loss_fn(model(xb), yb)
            l.backward()
            opt.step()
            tl += float(l) * len(xb)
        tl /= max(1, len(Xtr))

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for xb, yb in va:
                vl += float(loss_fn(model(xb.to(dev)), yb.to(dev))) * len(xb)
        vl /= max(1, len(Xva))
        hist["train"].append(tl)
        hist["val"].append(vl)

        if verbose:
            print(f"  epoch {ep:02d}  train {tl:.4f}  val {vl:.4f}")
        if vl < best_val - 1e-5:
            best_val, patience = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= cfg["patience"]:
                if verbose:
                    print(f"  early stop @ epoch {ep}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return hist


@torch.no_grad()
def predict_quantiles(model: QuantileLSTM, X: np.ndarray,
                      batch_size: int = 256) -> np.ndarray:
    """X (N, L) -> quantile predictions (N, horizon, Q) in the model's space."""
    model.eval()
    dev = next(model.parameters()).device
    outs = []
    for i in range(0, len(X), batch_size):
        xb = torch.tensor(X[i:i + batch_size], dtype=torch.float32, device=dev)
        outs.append(model(xb).cpu().numpy())
    return np.concatenate(outs, axis=0)
'''

FILES["ucra/core/__init__.py"] = r'''"""UCRA.core package."""
'''

FILES["ucra/core/uncertainty.py"] = r'''"""Stage 1 output assembly: quantile forecasts -> u_t, U_t, rho_t."""
from __future__ import annotations

import numpy as np


class UncertaintyState:
    """Rolling risk state of the network (feedback layer feeds it)."""

    def __init__(self, drift_window: int = 96):
        self.drift_window = drift_window
        self.violation_history: list[int] = []   # 1 = reservation missed demand
        self.recent_demand: list[float] = []
        self.baseline_mu: float | None = None    # train-scale stats
        self.baseline_sd: float = 1.0

    def set_reference(self, train_demand: np.ndarray):
        self.baseline_mu = float(np.mean(train_demand))
        self.baseline_sd = float(np.std(train_demand) + 1e-8)

    def observe(self, violated: bool, demand: float):
        self.violation_history.append(int(violated))
        self.recent_demand.append(float(demand))
        if len(self.violation_history) > self.drift_window:
            self.violation_history.pop(0)
        if len(self.recent_demand) > self.drift_window:
            self.recent_demand.pop(0)

    def rho(self) -> float:
        """Risk indicator rho_t in [0, 1.5]: recent violation rate + drift z."""
        if not self.violation_history:
            return 0.0
        viol_rate = float(np.mean(self.violation_history))
        z = 0.0
        if self.baseline_mu is not None and len(self.recent_demand) >= 8:
            recent = float(np.mean(self.recent_demand))
            z = abs(recent - self.baseline_mu) / self.baseline_sd
        return min(1.5, viol_rate + max(0.0, z - 2.0) / 4.0)


def extract_uncertainty(pred_q: np.ndarray, quantiles: list[float],
                        state: UncertaintyState) -> dict:
    """Turn (horizon, Q) quantile prediction into Stage 1 outputs at time t.

    Returns dict with:
      u_hat : point forecast (median) for the next slot
      spread: U_t = q99 - q50 (uncertainty width used by Phi)
      rho_t : risk indicator from the rolling state
    """
    qi = {round(q, 2): i for i, q in enumerate(quantiles)}
    q50 = pred_q[0, qi[0.5]]
    q99 = pred_q[0, qi[0.99]]
    return {
        "u_hat": float(q50),
        "spread": float(max(0.0, q99 - q50)),
        "rho_t": state.rho(),
        "q_vec": pred_q[0].tolist(),
    }
'''

FILES["ucra/core/transform.py"] = r'''"""Stage 2 (Phi: Uncertainty-to-Reservation) and Stage 4 (Update & Release).

Phi (see docs/METHODOLOGY.md):
    R_target = clip( q_tau + kappa * spread * (1 + rho_weight * rho_t),
                     min_headroom * u_hat, capacity * max_reserve_ratio )

Stage 4 smooths the target into R_{t+1} with EWMA + release hysteresis:
reservations drop only when headroom stays large (avoids oscillation).
"""
from __future__ import annotations

import numpy as np


def phi_transform(u_hat: float, spread: float, rho_t: float,
                  capacity: float, phi_cfg: dict) -> float:
    """Uncertainty triple (u_hat, U_t, rho_t) -> reservation R_t."""
    tau = phi_cfg["base_tau"]
    # q_tau approximated from the median and the spread ratio:
    # q_tau ~ q50 + (tau-0.5)/(0.99-0.5) * spread
    q_tau = u_hat + (tau - 0.5) / 0.49 * spread
    r = q_tau + phi_cfg["kappa"] * spread * (1.0 + phi_cfg["rho_weight"] * rho_t)
    floor = (1.0 + phi_cfg["min_headroom"]) * u_hat
    ceil = capacity * phi_cfg["max_reserve_ratio"]
    return float(np.clip(r, floor, ceil))


def update_reservation(r_prev: float, r_target: float, realized: float,
                       capacity: float, upd_cfg: dict) -> float:
    """Stage 4: EWMA-smoothed reservation with release hysteresis."""
    a = upd_cfg["ewma_alpha"]
    r_next = (1 - a) * r_prev + a * r_target
    # release only if unused headroom exceeds hysteresis threshold
    headroom_ratio = (r_prev - realized) / max(capacity, 1e-9)
    if headroom_ratio > upd_cfg["release_hysteresis"] and r_target < r_prev:
        r_next = (1 - a) * r_prev + a * max(r_target, realized)
    return float(max(0.0, r_next))
'''

FILES["ucra/core/allocate.py"] = r'''"""Stage 3 - Risk-Adaptive Allocation.

Distributes the reservation R_t across competing entities (sectors or
slices) proportionally to demand pressure and per-entity risk, keeping a
best-effort pool aside. Violation = demand that exceeds the allocated share.
"""
from __future__ import annotations

import numpy as np


def risk_adaptive_allocate(r_t: float, demands: np.ndarray,
                           risks: np.ndarray | None = None,
                           alloc_cfg: dict | None = None) -> dict:
    """Split R_t across entities given their current demand and risk.

    weights_i = demand_i^pressure_exp * (1 + risk_i)^risk_exp
    A_i = min(demand_i, R_t * w_i / sum(w)) + best-effort leftovers

    Returns dict with allocations A, unmet (violations) and the pool split.
    """
    cfg = alloc_cfg or {}
    p_exp = cfg.get("pressure_exponent", 1.0)
    r_exp = cfg.get("risk_exponent", 0.5)
    be_share = cfg.get("best_effort_share", 0.1)

    demands = np.asarray(demands, dtype=float)
    risks = np.zeros_like(demands) if risks is None \
        else np.asarray(risks, dtype=float)
    n = len(demands)
    if n == 0 or r_t <= 0:
        return {"alloc": np.zeros(n), "unmet": demands.copy(),
                "best_effort": 0.0}

    best_effort_pool = be_share * r_t
    reservable = r_t - best_effort_pool

    w = (np.maximum(demands, 0.0) ** p_exp) * ((1.0 + np.maximum(risks, 0.0)) ** r_exp)
    tot = float(w.sum())
    shares = (w / tot) if tot > 0 else np.full(n, 1.0 / n)

    alloc = np.minimum(demands, reservable * shares)
    # leftover from entities whose demand < share -> best effort
    leftover = reservable - float(alloc.sum())
    if leftover > 0:
        unmet_idx = np.where(demands > alloc)[0]
        if len(unmet_idx):
            add = np.minimum(demands[unmet_idx] - alloc[unmet_idx],
                             leftover * shares[unmet_idx] / max(shares[unmet_idx].sum(), 1e-9))
            alloc[unmet_idx] += add
    unmet = np.maximum(demands - alloc, 0.0)
    return {"alloc": alloc, "unmet": unmet,
            "best_effort": best_effort_pool}
'''

FILES["ucra/core/evolve.py"] = r'''"""Stage 5 - Self-Evolving Learning.

DriftMonitor watches the feedback (violation rate, demand z-score). When a
trigger fires, the EvolutionEngine fine-tunes theta_t -> theta_{t+1} on a
reservoir-sampled replay buffer of recent windows (continual learning
without catastrophic forgetting).
"""
from __future__ import annotations

import random

import numpy as np
import torch


class ReplayBuffer:
    """Reservoir sampling over (window, target) pairs."""

    def __init__(self, capacity: int = 512):
        self.capacity = capacity
        self.X: list[np.ndarray] = []
        self.Y: list[np.ndarray] = []
        self._n_seen = 0

    def push(self, x: np.ndarray, y: np.ndarray):
        self._n_seen += 1
        if len(self.X) < self.capacity:
            self.X.append(x)
            self.Y.append(y)
        else:
            j = random.randint(0, self._n_seen - 1)
            if j < self.capacity:
                self.X[j] = x
                self.Y[j] = y

    def arrays(self):
        if not self.X:
            return None, None
        return np.stack(self.X), np.stack(self.Y)


class DriftMonitor:
    def __init__(self, cfg: dict):
        self.violation_trigger = cfg["violation_trigger"]
        self.drift_zscore = cfg["drift_zscore"]
        self.window = cfg.get("drift_window", 96)
        self.violations: list[int] = []
        self.mu: float | None = None
        self.sd: float = 1.0

    def set_reference(self, train_demand: np.ndarray):
        self.mu = float(np.mean(train_demand))
        self.sd = float(np.std(train_demand) + 1e-8)

    def observe(self, violated: bool, demand: float) -> dict:
        self.violations.append(int(violated))
        if len(self.violations) > self.window:
            self.violations.pop(0)
        viol_rate = float(np.mean(self.violations))
        z = abs(demand - self.mu) / self.sd if self.mu is not None else 0.0
        return {"viol_rate": viol_rate, "z": z}

    def triggered(self, stats: dict) -> bool:
        return (stats["viol_rate"] > self.violation_trigger
                or stats["z"] > self.drift_zscore)


class EvolutionEngine:
    def __init__(self, model, cfg: dict, train_cfg: dict):
        self.model = model
        self.cfg = cfg                 # evolution section
        self.train_cfg = train_cfg     # model section (lr etc.)
        self.buffer = ReplayBuffer(cfg["replay_size"])
        self.n_updates = 0

    def remember(self, x: np.ndarray, y: np.ndarray):
        self.buffer.push(x, y)

    def finetune(self, loss_fn) -> dict:
        """theta_t -> theta_{t+1} on the replay buffer (few epochs)."""
        X, Y = self.buffer.arrays()
        if X is None:
            return {"updated": False}
        dev = next(self.model.parameters()).device
        opt = torch.optim.Adam(self.model.parameters(), lr=self.train_cfg["lr"] * 0.3)
        xt = torch.tensor(X, dtype=torch.float32, device=dev)
        yt = torch.tensor(Y, dtype=torch.float32, device=dev)
        self.model.train()
        for _ in range(self.cfg["finetune_epochs"]):
            opt.zero_grad()
            loss = loss_fn(self.model(xt), yt)
            loss.backward()
            opt.step()
        self.n_updates += 1
        self.model.eval()
        return {"updated": True, "replay": len(X),
                "loss": float(loss), "n_updates": self.n_updates}
'''

FILES["ucra/eval/__init__.py"] = r'''"""UCRA.eval package."""
'''

FILES["ucra/eval/metrics.py"] = r'''"""Feedback & Evaluation layer metrics (all computed on the TEST window)."""
from __future__ import annotations

import numpy as np


def reservation_error(R: np.ndarray, D: np.ndarray,
                      capacity: float) -> float:
    """Mean absolute gap between reservation and realized demand, normalised."""
    return float(np.mean(np.abs(R - D)) / max(capacity, 1e-9))


def violation_rate(R: np.ndarray, D: np.ndarray) -> float:
    """Risk exposure: fraction of slots where demand exceeded the reservation."""
    return float(np.mean(D > R))


def utilization(R: np.ndarray, D: np.ndarray) -> float:
    """Mean reserved-capacity utilization (capped at 1 per slot)."""
    return float(np.mean(np.minimum(D / np.maximum(R, 1e-9), 1.0)))


def waste(R: np.ndarray, D: np.ndarray, capacity: float) -> float:
    """Over-provisioned (unused reservation) share of capacity."""
    return float(np.mean(np.maximum(R - D, 0.0)) / max(capacity, 1e-9))


def summarize(R: np.ndarray, D: np.ndarray, capacity: float) -> dict:
    return {
        "reservation_error": reservation_error(R, D, capacity),
        "violation_rate": violation_rate(R, D),
        "utilization": utilization(R, D),
        "waste": waste(R, D, capacity),
    }
'''

FILES["ucra/eval/baselines.py"] = r'''"""Baseline reservation policies UCRA is compared against."""
from __future__ import annotations

import numpy as np


def static_peak(train_demand: np.ndarray, n_test: int) -> np.ndarray:
    """Reserve the historical peak everywhere (worst-case static sizing)."""
    return np.full(n_test, float(np.max(train_demand)))


def mean_forecast(pred_q: np.ndarray, quantiles: list[float]) -> np.ndarray:
    """Reserve the point forecast (median) only - ignores uncertainty."""
    qi = {round(q, 2): i for i, q in enumerate(quantiles)}
    return pred_q[:, 0, qi[0.5]].copy()


def oracle_quantile(test_demand: np.ndarray, horizon: int,
                    tau: float = 0.9) -> np.ndarray:
    """CHEATING upper bound: the true rolling quantile of the test window."""
    out = np.empty(len(test_demand))
    w = max(48, horizon * 4)
    for i in range(len(test_demand)):
        lo = max(0, i - w)
        out[i] = np.quantile(test_demand[lo:i + 1], tau) if i > 0 else test_demand[0]
    return out
'''

FILES["ucra/eval/plots.py"] = r'''"""Paper-grade figures (PNG, 150 dpi, constrained layout)."""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _save(fig, path: str | Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"[plots] saved {path}")


def plot_demand_vs_reservation(ts, demand, R_ucra, R_static, path):
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    ax.plot(ts, demand, lw=1.0, label="Realized demand", color="#1f77b4")
    ax.plot(ts, R_ucra, lw=1.2, label="UCRA reservation", color="#d62728")
    ax.plot(ts, R_static, lw=1.0, ls="--", label="Static-peak baseline", color="#7f7f7f")
    ax.set_xlabel("time")
    ax.set_ylabel("demand / reservation")
    ax.set_title("UCRA reservation tracks uncertain demand")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _save(fig, path)


def plot_quantile_band(ts, realized, pred_q, quantiles, path, n=240):
    """Predicted quantile fan chart vs realized demand (first n test slots)."""
    qi = {round(q, 2): i for i, q in enumerate(quantiles)}
    lo, hi, med = qi[0.05], qi[0.99], qi[0.5]
    fig, ax = plt.subplots(figsize=(11, 4), constrained_layout=True)
    tt = ts[:n]
    ax.fill_between(tt, pred_q[:n, 0, lo], pred_q[:n, 0, hi],
                    alpha=0.25, color="#1f77b4", label="q05-q99 band (U_t)")
    ax.plot(tt, pred_q[:n, 0, med], color="#1f77b4", lw=1.0, label="median forecast")
    ax.plot(tt, realized[:n], color="#ff7f0e", lw=0.9, label="realized")
    ax.set_xlabel("time")
    ax.set_ylabel("demand")
    ax.set_title("Stage 1: quantile uncertainty band vs realized demand")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _save(fig, path)


def plot_kappa_sweep(kappas, viol, util, path):
    """Risk-utility trade-off: kappa sweep (Phi risk aversion)."""
    fig, ax1 = plt.subplots(figsize=(7, 4), constrained_layout=True)
    ax1.plot(kappas, viol, "o-", color="#d62728", label="violation rate")
    ax1.set_xlabel("kappa (risk aversion in Phi)")
    ax1.set_ylabel("violation rate", color="#d62728")
    ax2 = ax1.twinx()
    ax2.plot(kappas, util, "s--", color="#1f77b4", label="utilization")
    ax2.set_ylabel("utilization", color="#1f77b4")
    ax1.set_title("Phi risk-utility trade-off")
    _save(fig, path)


def plot_training_history(hist: dict, path: str):
    fig, ax = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
    ax.plot(hist["train"], label="train pinball loss")
    ax.plot(hist["val"], label="val pinball loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Quantile LSTM training")
    ax.legend()
    _save(fig, path)


def plot_evolution(ts, viol_rolling, update_points, path, window=96):
    """Rolling violation rate with markers where self-evolution fired."""
    fig, ax = plt.subplots(figsize=(11, 3.2), constrained_layout=True)
    ax.plot(ts, viol_rolling, color="#9467bd", lw=1.2,
            label=f"rolling violation rate ({window}-slot)")
    for p in update_points:
        if 0 <= p < len(ts):
            ax.axvline(ts[p], color="#2ca02c", ls=":", lw=0.8)
    ax.set_xlabel("time")
    ax.set_ylabel("violation rate")
    ax.set_title("Stage 5: self-evolving fine-tuning events (green lines)")
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0))
    _save(fig, path)
'''


def main() -> None:
    for rel, content in FILES.items():
        target = Path(rel)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        print("  wrote", rel)
    print("\nPart A done (ucra package, 18 files). Now run Part B.")


if __name__ == "__main__":
    main()