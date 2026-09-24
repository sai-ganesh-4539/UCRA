#!/usr/bin/env bash
# UCRA dataset downloader (resumable). Usage:
#   bash scripts/download_data.sh          # RAN Dataset_01 (8 MB) + CTTC dataset (271 MB)
#   RAN_ALL=1 bash scripts/download_data.sh  # all three RAN subsets (~767 MB)
#   SKIP_CTTC=1 bash scripts/download_data.sh
set -e
cd "$(dirname "$0")/.."

ZENODO_RAN="https://zenodo.org/records/17815388/files"
ZENODO_CTTC="https://zenodo.org/records/10610616/files"

dl() {  # dl <url> <out>
  echo "[download] $2"
  curl -L -C - --retry 5 --retry-delay 3 -o "$2" "$1"
}

mkdir -p data/ran data/cttc

# ---- Dataset 2: Live RAN PM counters (real commercial network) ----
dl "${ZENODO_RAN}/Dataset_01.zip" data/ran/Dataset_01.zip
if [ "${RAN_ALL:-0}" = "1" ]; then
  dl "${ZENODO_RAN}/Dataset_02.zip" data/ran/Dataset_02.zip
  dl "${ZENODO_RAN}/Dataset_03.zip" data/ran/Dataset_03.zip
fi
dl "${ZENODO_RAN}/README.md" data/ran/README_dataset.md || true

# ---- Dataset 1: CTTC B5G slicing (simulated, slice-level) ----
if [ "${SKIP_CTTC=0}" != "1" ]; then
  dl "${ZENODO_CTTC}/slicing-simulations.zip" data/cttc/slicing-simulations.zip
  # unzip; datanetAPI.py ships inside the archive, if not, fetch it separately
  (cd data/cttc && unzip -n -q slicing-simulations.zip -d slicing-simulations)
  if [ ! -f data/cttc/slicing-simulations/datanetAPI.py ]; then
    dl "${ZENODO_CTTC}/datanetAPI.py" data/cttc/slicing-simulations/datanetAPI.py
  fi
  pip install jsonpickle --quiet || true
fi

echo "[download] done. Files:"
du -sh data/ran data/cttc 2>/dev/null || true
