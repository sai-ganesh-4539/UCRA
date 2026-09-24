"""Dataset dispatch: config -> canonical demand frame."""
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
