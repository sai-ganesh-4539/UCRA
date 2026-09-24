# 07 -- Data Sources: Provenance, Schemas, Loaders, Citations

Both datasets are real, published, CC-BY-4.0, and fetched from Zenodo by
`scripts/download_data.py` (pure Python stdlib -- works on any OS).
Datasets are NOT committed to git (`.gitignore` hard-excludes `data/`).

## 1. Why two datasets (division of labor)

| | RAN PM counters (17815388) | CTTC slicing (10610616) |
|---|---|---|
| Nature | commercial live network, 15-min PM counters | simulation steady-state snapshots |
| Unit of analysis | time (1,778-slot series) | snapshot x slice (independent samples) |
| Feeds stages | 1 (uncertainty), 4 (update), 5 (evolution) | 2-3 (reservation sizing vs violations) |
| Feedback signals | utilization, violations | violations, drops, delay |
| Temporal drift | yes (real days/weeks) | no (i.i.d. snapshots) |
| Headline metric | violation 0.0% @ util 71.7% | URLLC 29%->1.4%, eMBB 50.6%->0% |

Together they cover all five UCRA stages; neither alone could (one has no
slice semantics, the other has no time axis).

## 2. RAN dataset (Zenodo 17815388)

- Title: PM counters from live 5G, 4G and 2G RAN (STU Bratislava + Ericsson
  Slovakia), published 2026, Scientific Data, DOI 10.1038/s41597-026-07723-0,
  license CC-BY-4.0.
- Files: Dataset_01.zip (8,333,223 B), Dataset_02.zip (49,441,996 B),
  Dataset_03.zip (709,421,773 B), README (2,312 B). We use Dataset_01.
- Schema (verified against the real CSVs): one CSV per radio technology per
  baseband; columns include `Base station`, `Sector`, `Timestamp`, and
  tech-prefixed counters `<2G|4G|5G> data volume DL/UL`, `<tech> RB
  utilization`, `<tech> active users DL/UL`, `<tech> CQI rank 1..4`,
  `<tech> RRC users`, `<tech> MIMO rank DL`. Measurements every 15 min per
  sector.
- Campaign structure: separate observation campaigns separated by multi-day
  gaps (Mar 2023 and Oct 2023, ~193 days apart). This is why
  `load_demand_series` splits the resampled series at any gap > 15 min and
  keeps only the LONGEST gap-free segment -- a training window that silently
  crossed a 193-day hole would interpolate garbage and poison Stage 1. The
  longest segment: 1,778 slots, Oct 6 21:15 -> Oct 25 09:30, 2023. Internal
  holes of up to 4 slots are linearly interpolated (limit=4).

### Loader pipeline (`ucra/data/ran_loader.py`)

1. `_extract_if_needed`: unzip any `Dataset_*.zip` whose folder is missing.
2. Per CSV: `_map_columns` detects the technology prefix and requires that
   tech's 5 key columns; files without them are skipped (with a printed
   note).
3. Build the canonical long frame: `[ts, sector, data_volume (DL+UL),
   utilization, active_users (DL+UL), tech]` -> 302,052 rows, 75 sectors,
   techs {4G, 5G} for Dataset_01.
4. `load_demand_series`: groupby ts -> sum over sectors -> strict 15-min
   grid -> split at gaps -> longest segment -> interpolate (limit 4).

### Capacity estimate (honesty box)

The PM data does not publish physical sector capacities. run_ucra uses
`C = 1.3 x max(observed demand)` = 1.3 x ~110,734 = 143,954.5 -- a
deliberate headroom factor so the ceiling in Phi is essentially never
binding (0.95C ~ 136,757 ~ 1.24x observed peak). Treat C as a
normalization constant, not a measured quantity; say so in the paper
(doc 14). Changing the 1.3 factor rescales all normalized metrics
uniformly and changes no ranking.

## 3. CTTC dataset (Zenodo 10610616)

- Title: CTTC B5G network slicing dataset (Farreras et al.), Data in Brief
  55:110738, 2024, license CC-BY-4.0. Files: slicing-simulations.zip
  (271,059,245 B) + datanetAPI.py (42,915 B, the dataset's own reader).
- Each sample is one steady-state simulation snapshot:
  - `topology_object`: networkx graph; edge attribute `bandwidth` = link
    capacity. We take `link_cap = mean(edge bandwidths)`.
  - traffic matrix: per-flow offered demand.
  - `slices`: list of eMBB / mMTC / URLLC slice entries, each with `delta`
    (reserved share, 0..1) and `flows`.
  - `performance_matrix`: per src-dst pair: PktsDrop, AvgDelay, p10-p90,
    Jitter.
- Derived row per slice: `reserved = delta * link_cap`;
  `offered = sum of flow rates`; plus drops_ratio / avg_delay aggregated
  over pairs. Frame columns: [sample, slice_type, delta, reserved, offered,
  link_cap, drops_ratio, avg_delay].

### The schema saga (why the loader is "dict-tolerant")

The dataset's own `datanetAPI.py` expects jsonpickle-decoded OBJECTS
(`slice.type`, `slice.flows`, flow `avgRate`). The current build of the
archive (v2.2.1) instead delivers plain PYTHON DICTS with `bandwidth` (bps)
on flows -- so the vendor reader crashes with "'dict' object has no
attribute 'type'". Our loader reads BOTH shapes (dict or object), and
extracts each flow's rate from `bandwidth`, falling back to `avgRate`,
`rate`, then to the 2nd comma-field of `traffic_string`. It also imports
the dataset's class under the correct name `DatanetAPI` (capital D -- the
lowercase import was bug #1, fixed in Part C). Corrupt samples are skipped
with a printed note instead of aborting the run.

Your parse: 120 samples -> frame (2,908 rows, 8 cols); slice counts
mMTC 1,201 / URLLC 1,106 / eMBB 601; test split (samples > 70% cut) yields
URLLC n=276, eMBB n=164, mMTC n=296 -- the n's in your results JSON.

## 4. Synthetic generator (`ucra/data/synthetic.py`)

15-min multi-sector series: `base * sector_gain * diurnal * weekly * (1 +
AR(1) noise + slow sinusoid + ON/OFF bursts)`, diurnal amplitude 0.45
(period 96), weekly 0.15, AR(1) rho 0.85, bursts start with p=0.01 per slot
and last 2-10 slots with height 0.5-1.4. Used by the smoke test and for
sanity runs without any download. `synthetic_capacity` = 250 * n_sectors.

## 5. Downloader behavior (`scripts/download_data.py`)

- md5 verification against baked-in checksums; skip-if-complete (so re-runs
  are instant); HTTP Range resume of `.part` files; 6 retries with backoff.
- `--list` shows known files/sizes/checksums; `--cttc` adds the CTTC pair;
  `--ran-all` adds Dataset_02/03; `--root` redirects the data dir.
- After CTTC download it unzips (idempotent, marker-file based) and pip
  installs `jsonpickle` (imported by the dataset's datanetAPI.py). On your
  machine pip printed "Defaulting to user installation" -- harmless.

## 6. Ready-to-paste citations

```bibtex
@dataset{ran_pm_counters_2026,
  title   = {{PM} counters from live {5G}, {4G} and {2G} {RAN}},
  author  = {Lehoczky, Jan and others},        % fill full author list from the Zenodo page
  year    = {2026},
  doi     = {10.5281/zenodo.17815388},
  note    = {Scientific Data, DOI 10.1038/s41597-026-07723-0. CC-BY-4.0}
}

@dataset{cttc_slicing_2024,
  title   = {{CTTC} {B5G} network slicing dataset},
  author  = {Farreras, Miquel and others},     % fill full author list from the Zenodo page
  year    = {2024},
  doi     = {10.5281/zenodo.10610616},
  note    = {Data in Brief 55:110738. CC-BY-4.0}
}
```

(CC-BY-4.0 requires attribution -- the two entries above satisfy it; also
cite the Scientific Data paper for the RAN set.)
