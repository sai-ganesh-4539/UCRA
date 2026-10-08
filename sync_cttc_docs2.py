#!/usr/bin/env python3
"""sync_cttc_docs2.py -- one leftover file: UCRA_EXPLAINED.md

The first sync covered README, the 4 DEEP guides and 6 docs/ guides, but
the root-level UCRA_EXPLAINED.md still shows the old partial-download
CTTC numbers (120 snapshots). This small addendum patches exactly that
file to the full-dataset numbers (250 snapshots).

Run from the repo root (where README.md lives):
    python sync_cttc_docs2.py

Safe to re-run: already-patched files are skipped.
"""
import sys
from pathlib import Path

PATCHES = [
    ("UCRA_EXPLAINED.md",
     "**CTTC multi-slice (120 snapshots, operator vs UCRA):**",
     "**CTTC multi-slice (250 snapshots, operator vs UCRA):**"),
    ("UCRA_EXPLAINED.md",
     "| URLLC | 29.0% | 15.2% | **1.4%** |\n"
     "| eMBB | 50.6% | 12.2% | **0.0%** |\n"
     "| mMTC | 5.4% | 12.2% | **4.4%** |",
     "| URLLC | 26.2% | 9.9% | **1.0%** |\n"
     "| eMBB | 44.2% | 5.3% | **1.2%** |\n"
     "| mMTC | 5.4% | 11.4% | **1.6%** |"),
]


def apply_one(root, path, old, new):
    p = root / path
    if not p.exists():
        return "FILE-MISSING"
    raw = p.read_bytes().decode("utf-8")
    for eol in ("\n", "\r\n"):
        o = old.replace("\n", eol)
        if o in raw:
            if raw.count(o) != 1:
                return "NOT-UNIQUE (" + str(raw.count(o)) + " matches)"
            p.write_bytes(raw.replace(o, new.replace("\n", eol)).encode("utf-8"))
            return "patched"
    for eol in ("\n", "\r\n"):
        if new.replace("\n", eol) in raw:
            return "skip (already patched)"
    return "NOT-FOUND"


def main():
    root = Path(__file__).resolve().parent
    print("[sync2] repo root:", root)
    ok = skip = bad = 0
    for path, old, new in PATCHES:
        status = apply_one(root, path, old, new)
        tag = "[ ok ]" if status == "patched" else \
              ("[skip]" if status.startswith("skip") else "[FAIL]")
        print("{} {:22s} {}".format(tag, path, status))
        if status == "patched":
            ok += 1
        elif status.startswith("skip"):
            skip += 1
        else:
            bad += 1
    print("-" * 60)
    print("[sync2] patched {} | skipped {} | failed {}".format(ok, skip, bad))
    if bad:
        print("[sync2] some patches did not apply - reply with this output.")
        return 1
    print("[sync2] UCRA_EXPLAINED.md now matches the full-dataset run too.")
    print()
    print("Next (one line at a time):")
    print("  git add -A")
    print('  git commit -m "docs: sync UCRA_EXPLAINED CTTC table to 250-snapshot run"')
    print("  git push")
    return 0


if __name__ == "__main__":
    sys.exit(main())