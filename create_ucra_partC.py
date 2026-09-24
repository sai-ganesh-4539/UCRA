#!/usr/bin/env python3
# ============================================================
# UCRA - Part C generator
# writes:  scripts/download_data.py            (new)
#          ucra/data/cttc_loader.py            (bug-fix overwrite)
#
# Why 1: scripts/download_data.sh (from Part B) needs bash/curl,
# which Windows PowerShell does not have. This Python downloader
# is pure stdlib and runs anywhere - resume + MD5 verified.
# Why 2: the Part A cttc_loader imported the wrong class name
# (datanetAPI instead of DatanetAPI) - fixed here.
#
# Usage:  save this file as create_ucra_partC.py in E:\UCRA
#         python create_ucra_partC.py
# ============================================================
from pathlib import Path

CTTC_LOADER_PY = r'''"""Loader for Zenodo 10610616 - CTTC B5G Network Slicing Dataset
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
'''

DOWNLOAD_PY = r'''#!/usr/bin/env python3
"""UCRA data downloader - pure-stdlib, Windows/PowerShell safe.

Fetches the two core UCRA datasets from Zenodo with resume + MD5 checks:

  Dataset 2 (default) : PM Counters from Live 5G/4G/2G RAN - Zenodo 17815388
                        Dataset_01.zip (8.3 MB) + README.md
  Dataset 1 (--cttc)  : CTTC B5G Network Slicing - Zenodo 10610616
                        slicing-simulations.zip (271 MB) + datanetAPI.py

Usage (run from the repo root, any OS):
  python scripts/download_data.py              # RAN Dataset_01 (enough for train/run)
  python scripts/download_data.py --cttc       # + CTTC 271 MB (Stage 2-3 slice eval)
  python scripts/download_data.py --ran-all    # + RAN Datasets 02/03 (~758 MB)
  python scripts/download_data.py --list       # show remote file lists only
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

UA = {"User-Agent": "UCRA-downloader/1.0"}
RETRIES = 6

# key: (exact bytes, md5 hex) - verified against the Zenodo API, 2026-09-24
RECORDS = {
    "ran": ("https://zenodo.org/records/17815388/files", {
        "Dataset_01.zip": (8333223,   "cba4e8c2887c0d9382c6a63736fe5ad5"),
        "Dataset_02.zip": (49441996,  "7d9c1c019e5d18bf318b4e4ad9393d91"),
        "Dataset_03.zip": (709421773, "f9e3517853f6a21f3b49ad089f1b8da0"),
        "README.md":      (2312,      "092277c2bb7bd59a6635185456f3f406"),
    }),
    "cttc": ("https://zenodo.org/records/10610616/files", {
        "slicing-simulations.zip": (271059245, "2a4a9a47bdeaa0c02304c40115c56273"),
        "datanetAPI.py":           (42915,     "f43037e2b83f7996ef26f4c30abe7f9e"),
    }),
}


def md5_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def complete(path: Path, size, md5) -> bool:
    if size is not None and (not path.exists() or path.stat().st_size != size):
        return False
    if md5 is None:
        return path.exists()
    return md5_file(path) == md5


def fetch(url: str, dest: Path, size, md5, label: str) -> Path:
    """Streaming download with resume, retries and MD5 verification."""
    if complete(dest, size, md5):
        print(f"  [skip] {label}: already complete")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    last = None
    for attempt in range(1, RETRIES + 1):
        try:
            offset = part.stat().st_size if part.exists() else 0
            req = urllib.request.Request(url, headers=UA)
            if offset:
                req.add_header("Range", "bytes={}-".format(offset))
            with urllib.request.urlopen(req, timeout=60) as resp:
                if getattr(resp, "status", 200) == 200 and offset:
                    offset = 0                      # server ignored Range
                f_mode = "ab" if offset else "wb"
                h = hashlib.md5()
                if offset:                          # reseed hash with part file
                    with open(part, "rb") as f0:
                        for chunk in iter(lambda: f0.read(1 << 20), b""):
                            h.update(chunk)
                done, t0 = offset, time.time()
                with open(part, f_mode) as f:
                    while True:
                        chunk = resp.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        h.update(chunk)
                        done += len(chunk)
                        if time.time() - t0 > 0.5:
                            t0 = time.time()
                            mb = done / 1e6
                            if size:
                                print("\r  [dl..] {0}: {1:8.1f} MB ({2:5.1f}%)"
                                      .format(label, mb, 100.0 * done / size),
                                      end="", flush=True)
                            else:
                                print("\r  [dl..] {0}: {1:8.1f} MB"
                                      .format(label, mb), end="", flush=True)
                print("\r" + " " * 76 + "\r", end="")   # clear the progress line
            got = md5_file(part)
            if md5 and got != md5:
                raise IOError("md5 mismatch: got {}, want {}".format(got, md5))
            if size is not None and part.stat().st_size != size:
                raise IOError("size mismatch: {} != {}"
                              .format(part.stat().st_size, size))
            part.replace(dest)
            print("  [ ok ] {0}: {1:,} bytes, md5 verified".format(
                label, dest.stat().st_size))
            return dest
        except Exception as exc:
            last = exc
            wait = 3 * attempt
            print("  [retry {}/{}] {}: {} - retrying in {}s".format(
                attempt, RETRIES, label, exc, wait))
            time.sleep(wait)
    raise RuntimeError("download failed after {} attempts: {} ({})"
                       .format(RETRIES, label, last))


def unzip(zip_path: Path, target: Path) -> None:
    """Idempotent extraction guarded by a marker file."""
    if (target / ".ucra_extracted").exists():
        print("  [skip] {} already extracted".format(target))
        return
    target.mkdir(parents=True, exist_ok=True)
    print("  [unzip] {} -> {}".format(zip_path.name, target))
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(target)
    (target / ".ucra_extracted").write_text("ok", encoding="ascii")


def job_ran(root: Path, all_subsets: bool) -> None:
    base, files = RECORDS["ran"]
    keys = ["Dataset_01.zip", "README.md"]
    if all_subsets:
        keys += ["Dataset_02.zip", "Dataset_03.zip"]
    for key in keys:
        name = "README_dataset.md" if key == "README.md" else key
        fetch("{}/{}?download=1".format(base, key),
              root / "data" / "ran" / name,
              files[key][0], files[key][1], "RAN " + key)
    print("  [note] ran_loader extracts Dataset_*.zip automatically at load time")


def job_cttc(root: Path) -> None:
    base, files = RECORDS["cttc"]
    zip_path = fetch("{}/slicing-simulations.zip?download=1".format(base),
                     root / "data" / "cttc" / "slicing-simulations.zip",
                     files["slicing-simulations.zip"][0],
                     files["slicing-simulations.zip"][1],
                     "CTTC slicing-simulations.zip")
    target = root / "data" / "cttc" / "slicing-simulations"
    unzip(zip_path, target)
    api = target / "datanetAPI.py"
    if not api.exists():
        fetch("{}/datanetAPI.py?download=1".format(base), api,
              files["datanetAPI.py"][0], files["datanetAPI.py"][1],
              "CTTC datanetAPI.py")
    try:
        import jsonpickle  # noqa: F401
    except ImportError:
        print("  [pip ] installing jsonpickle (needed by datanetAPI.py)")
        try:
            subprocess.check_call([sys.executable, "-m", "pip",
                                   "install", "jsonpickle"])
        except Exception as exc:
            print("  [warn] could not install jsonpickle ({}); "
                  "run: pip install jsonpickle".format(exc))


def list_remote() -> None:
    for rec in ("17815388", "10610616"):
        url = "https://zenodo.org/api/records/" + rec
        with urllib.request.urlopen(urllib.request.Request(url, headers=UA),
                                    timeout=60) as r:
            meta = json.load(r)
        print("record {}: {}".format(rec, meta["metadata"]["title"]))
        for f in meta.get("files", []):
            print("  {:30s} {:>13,} bytes  {}".format(
                f["key"], f["size"], f["checksum"]))


def main() -> None:
    ap = argparse.ArgumentParser(description="UCRA dataset downloader")
    ap.add_argument("--cttc", action="store_true",
                    help="also fetch CTTC B5G slicing dataset (271 MB)")
    ap.add_argument("--ran-all", action="store_true",
                    help="fetch all RAN subsets (~758 MB extra)")
    ap.add_argument("--list", action="store_true",
                    help="list remote Zenodo files and exit")
    ap.add_argument("--root", default=None,
                    help="repo root (default: parent of scripts/)")
    args = ap.parse_args()

    if args.list:
        list_remote()
        return

    root = Path(args.root) if args.root else Path(__file__).resolve().parents[1]
    print("[ucra] datasets live under: {}".format(root / "data"))
    job_ran(root, args.ran_all)
    if args.cttc:
        job_cttc(root)

    print("")
    print("[ucra] data ready. Next steps:")
    print("  python scripts/audit_data.py   --config configs/default.yaml")
    print("  python scripts/train.py        --config configs/default.yaml")
    print("  python scripts/run_ucra.py     --config configs/default.yaml --sweep")
    print("  python scripts/make_figures.py --config configs/default.yaml")


if __name__ == "__main__":
    main()
'''


def main() -> None:
    root = Path.cwd()
    (root / "scripts").mkdir(exist_ok=True)
    out = root / "scripts" / "download_data.py"
    out.write_text(DOWNLOAD_PY, encoding="utf-8")
    print("  wrote scripts/download_data.py")

    (root / "ucra" / "data").mkdir(parents=True, exist_ok=True)
    out2 = root / "ucra" / "data" / "cttc_loader.py"
    out2.write_text(CTTC_LOADER_PY, encoding="utf-8")
    print("  wrote ucra/data/cttc_loader.py  (fixed DatanetAPI import)")

    print("")
    print("Part C done (2 files). Now run:")
    print("  python scripts/download_data.py")
    print("  python scripts/audit_data.py  --config configs/default.yaml")
    print("  python scripts/train.py       --config configs/default.yaml")


if __name__ == "__main__":
    main()