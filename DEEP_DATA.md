# DEEP_DATA.md — datasets, loaders, features, integrity

Where the numbers come from: provenance, schemas, the exact loader logic,
the feature pipeline, and every anti-leakage measure. Numbers quoted are
from your actual runs (they reproduced the sandbox reference almost
identically: 302,052 rows / 75 sectors / same 1,778-slot segment).

---

## Dataset A — Live RAN PM counters (primary)

| field | value |
|---|---|
| Zenodo record | 17815388 — "PM Counters from Live 5G, 4G and 2G RAN" |
| producers | STU Bratislava + Ericsson Slovakia |
| anchor paper | Nature Scientific Data 2026, DOI 10.1038/s41597-026-07723-0 |
| license | CC-BY-4.0 |
| files | Dataset_01.zip 8.33 MB · Dataset_02.zip 49.44 MB · Dataset_03.zip 709.42 MB (+ README) |
| nature | **Real commercial network**, sector-level Performance Management counters every 15 min |
| what UCRA uses | `Dataset_01` (config `data.ran.subset: dataset01`) |

### Raw schema (one CSV per radio technology per baseband)

```
Base station, Sector, Timestamp,
<2G|4G|5G> max active users DL/UL,  <tech> data volume DL/UL,
<tech> max RRC users,               <tech> RB utilization,
<tech> CQI rank 1..4,               <tech> RRC users,
<tech> active users UL/DL,          <tech> MIMO rank DL
```

Column names are **tech-prefixed** ("5G data volume DL", "4G data volume DL",
...) — the loader exploits this to auto-detect which technology a file holds.

### What your load produced (both machines agree)

- 302,052 canonical rows, 75 sectors, technologies 4G + 5G
- two observation campaigns: **March 2023** and **October 2023**, separated
  by a **193-day gap**
- working segment: **Oct 6 21:15 -> Oct 25 09:30 = 1,778 contiguous 15-min slots**

---

## `ucra/data/ran_loader.py` — function by function

### `_extract_if_needed(root)`
Unzips any `Dataset_*.zip` that has no matching folder yet (idempotent —
safe to call on every run).

### `_map_columns(cols)`
Regex `^(2G|4G|5G)\s` finds the technology prefix; then requires the five
columns UCRA actually needs (`{tech} data volume DL/UL`, `{tech} RB
utilization`, `{tech} active users DL/UL`). Files without the schema are
skipped with a printed notice — never crash on extra/unknown CSVs.

### `load_ran(data_dir, subset)` -> canonical long frame
Per CSV: parse timestamps (`errors="coerce"` + drop NaT), build sector ID as
`"<Base station>_S<Sector>"`, and emit one row per (ts, sector):

| column | construction |
|---|---|
| `data_volume` | DL volume + UL volume (NaN -> 0, then summed) |
| `utilization` | `{tech} RB utilization` as-is |
| `active_users` | DL + UL active users |
| `tech` | the prefix detected above |

Result frame: `[ts, sector, data_volume, utilization, active_users, tech]`.

### `load_demand_series(df, agg="sum")` — the anti-gap core

```
1. groupby(ts).sum()                       -> one network demand per slot
2. breaks  = index.diff() > 15 min         -> gap flags between campaigns
3. seg_id  = breaks.cumsum()               -> segment number per slot
4. keep only the LARGEST segment            -> Oct 6-25, 1,778 slots
5. asfreq("15min").interpolate(limit=4)    -> patch tiny internal holes
```

Steps 2-4 are the load-bearing decision: **windows must never cross the
193-day gap** — a 96-slot input window straddling March->October would mix
two traffic regimes and corrupt training. Keeping only the longest
gap-free segment is the cleanest possible guarantee (no synthetic bridging
of a 193-day hole).

### Where capacity comes from

`ucra/data/__init__.py -> load_dataset("ran")`:

```
capacity = max(segment demand) * 1.3  =  110,734.2 * 1.3  =  143,954.5
```

The 1.3 headroom factor is a heuristic (documented honestly in the docs
set): it represents physically usable capacity above the observed peak.
The `max_reserve_ratio: 0.95` ceiling then caps any reservation at
136,757 — so Phi can never claim more than 95% of that estimate.

`load_dataset` also returns `sector_frame` — the full long frame — which
`run_ucra.py` uses for per-sector Stage-3 allocation.

---

## Dataset B — CTTC B5G Network Slicing (multi-slice validation)

| field | value |
|---|---|
| Zenodo record | 10610616 |
| producer | CTTC (Centre Tecnologic de Telecomunicacions de Catalunya) |
| anchor paper | Data in Brief 55:110738, 2024 |
| license | CC-BY-4.0 |
| file | `slicing-simulations.zip`, 271,059,245 bytes, md5-verified |
| nature | simulated steady-state snapshots of a 5G slicing simulation |
| access | ships its own `datanetAPI.py` reader inside the zip |

Each sample carries: a **topology** (networkx graph; link `bandwidth` =
capacity), **slices** (eMBB / mMTC / URLLC with `delta` = reserved share of
link capacity and per-slice traffic **flows**), and a **performance matrix**
(PktsDrop, AvgDelay, p10-p90, Jitter per src-dst pair).

### `ucra/data/cttc_loader.py` — the schema war story

The current dataset build (v2.2.1) stores slices and flows as **plain
dicts**, while the official `datanetAPI` helpers expect **objects** with
attributes (`slice.type`, `flow.avgRate`) — calling them crashes with
`'dict' object has no attribute 'type'`. The loader therefore reads BOTH
shapes:

```python
stype  = sdict.get("type")  if isinstance(s, dict) else sdict.type
delta  = sdict.get("delta") if isinstance(s, dict) else sdict.delta
# flow rate: bandwidth (bps) -> avgRate -> rate -> traffic_string field 2
```

Per slice it aggregates offered load across all flows and averages
drop/delay over all src-dst pairs, emitting:

```
[sample, slice_type, delta, reserved, offered, link_cap, drops_ratio, avg_delay]
reserved = delta * link_cap        # link_cap = mean edge bandwidth
```

Corrupt samples are skipped with a logged notice, never fatal. Your
evaluation consumed **120 samples** (70/30 chronological split).

### CTTC-specific caveat (documented in docs/07 + 09)

`delta` (the operator's reservation) is measured **at the same snapshot**
as the offered load, so Stage-2 comparisons on CTTC use the
train-only-fitted Phi recipe; the kappa grid there forces `rho = 0` for the
same reason (no same-snapshot feedback contamination).

---

## Feature engineering — from frames to tensors

`ucra/features/windowing.py`, used identically by `train.py` and
`run_ucra.py`:

### `make_windows(series, seq_len=96, horizon=4)`

```
N = len(series) - 96 - 4 + 1  =  1,778 - 100  =  1,679 windows
X[i] = series[i      : i + 96]      # input:  previous 24 h
Y[i] = series[i + 96 : i + 100]     # target: next 1 h (4 slots)
```

`D_test = Y[ite][:, 0]` — evaluation uses the **first** future slot of each
window (the other 3 horizon steps are trained for but not scored in the
main loop).

### `chronological_split(1679, (0.7, 0.15, 0.15))`

```
train = 0    .. 1,175    (1,175 windows)
val   = 1,175.. 1,427    (  252 windows)
test  = 1,427.. 1,679    (  252 windows)
```

**No shuffling** — time series order is preserved everywhere. Only
DataLoader shuffles *within* the training set (batch composition, still
chronological at the window level).

### Two z-score scalers, both fitted on TRAIN only

| scaler | fitted on | applied to |
|---|---|---|
| `scaler` | `X[train]` (inputs) | all X splits |
| `scY` | `Y[train]` (targets) | Y for training; inverted after prediction |

Both are persisted in `outputs/scaler.npz` (`mu, sd, mu_y, sd_y`) so
`run_ucra.py` reloads the exact training-time scaling when it loads a
checkpoint. The `+1e-8` in `sd` guards division by constant series.

---

## Leakage audit (what could go wrong, and why it can't here)

| risk | protection |
|---|---|
| train/test crossing the 193-day campaign gap | longest-contiguous-segment extraction (loader) |
| future information in inputs | windows only look back 96 slots; targets start at t+1 |
| scaler peeking at test | both scalers fitted on train slice only |
| shuffled time split | `chronological_split` never shuffles |
| Stage-2 feedback using current-slot info | rho_t built from violations/demand of slots **before** t (CTTC delta caveat handled separately) |
| drift trigger peeking ahead | DriftMonitor window contains only past observations |

---

## Download integrity — `scripts/download_data.py`

Pure-stdlib downloader (no curl/wget dependency, PowerShell-friendly):

| feature | detail |
|---|---|
| baked manifest | exact byte sizes + md5 for every file |
| verify | md5 checked after download; mismatch -> non-zero exit |
| resume | HTTP Range requests; interrupted files continue from `.part` |
| retries | 6 attempts per file |
| skip-if-complete | existing verified files are not re-downloaded |
| usage | `python scripts/download_data.py` (RAN default) · `--cttc` · `--ran-all` · `--list` |

Baked RAN checksums (Dataset_01): 8,333,223 bytes; Dataset_02: 49,441,996;
Dataset_03: 709,421,773; CTTC zip: 271,059,245. Anyone in the world can
therefore reproduce **byte-identical** inputs without the 1+ GB ever
touching git.

`scripts/audit_data.py` then sanity-checks the series (length, NaN count,
peak vs capacity, prints the demand histogram to `outputs/audit_demand.png`)
— always run it first; it catches silent download corruption before
training wastes your time.

---

## The rejected candidates (why these two datasets won)

| candidate | verdict | reason |
|---|---|---|
| Zenodo 21625252 "Proactive Resource Allocation" | rejected | "augmented" data, no anchor paper, pre-baked allocation column |
| Zenodo 16265621 UE Traffic Model | backup | simulated scenarios, preprint-only anchor |
| 10610616 Milan-style demand (2013-14) | cite-only | underlying data a decade old |
| BurstGPT | backup | real traces, but LLM-serving domain, not 5G |
| Liverpool 5G HDD | optional 3rd | 1-3 s resolution bursts; kept as failsafe |
| **Zenodo 17815388 Live RAN** | **primary** | real commercial network, long horizon, Stages 1/4/5 |
| **Zenodo 10610616 CTTC** | **second** | slice logic + reservations, Stages 2-3 |

The pairing is deliberate: the RAN set exercises temporal uncertainty and
the drift loop on real data; CTTC exercises slice-level reservation logic
that the single-pool RAN series cannot. Together they cover all five stages
with no overlap gap.
