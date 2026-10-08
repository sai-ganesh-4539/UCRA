#!/usr/bin/env python3
"""sync_cttc_docs.py -- align every doc with the FULL-dataset CTTC run.

Background: the CTTC numbers first published in the README and the docs
came from a partial download (120 snapshots). Your machine now runs the
FULL dataset (default 250 snapshots), so outputs/cttc_results.json holds
the authoritative numbers:

    URLLC  operator 26.2% -> ucra_phi 1.0%   (n=577)
    eMBB   operator 44.2% -> ucra_phi 1.2%   (n=337)
    mMTC   operator  5.4% -> ucra_phi 1.6%   (n=630)

This script patches every stale CTTC figure in README.md, the four
DEEP_*.md guides and the affected docs/ guides. Nothing else is touched.

Run it from the repo root (E:\UCRA):
    python sync_cttc_docs.py

Safe to re-run: already-patched files are skipped.
"""
import sys
from pathlib import Path

# (file, old, new) -- applied in order; each must match exactly once.
PATCHES = [
    # ---------- README.md ----------
    ("README.md",
     "Zenodo 10610616, 120 snapshots, 70/30 split",
     "Zenodo 10610616, 250 snapshots, 70/30 split"),
    ("README.md",
     "| URLLC | 29.0% violations | 15.2% | **1.4%** |\n"
     "| eMBB | 50.6% violations | 12.2% | **0.0%** |\n"
     "| mMTC | 5.4% violations | 12.2% | **4.4%** |",
     "| URLLC | 26.2% violations | 9.9% | **1.0%** |\n"
     "| eMBB | 44.2% violations | 5.3% | **1.2%** |\n"
     "| mMTC | 5.4% violations | 11.4% | **1.6%** |"),
    # ---------- DEEP_RESULTS.md ----------
    ("DEEP_RESULTS.md",
     "| URLLC | 29.0% | 15.2% | **1.4%** |\n"
     "| eMBB | 50.6% | 12.2% | **0.0%** |\n"
     "| mMTC | 5.4% | 12.2% | **4.4%** |",
     "| URLLC | 26.2% | 9.9% | **1.0%** |\n"
     "| eMBB | 44.2% | 5.3% | **1.2%** |\n"
     "| mMTC | 5.4% | 11.4% | **1.6%** |"),
    ("DEEP_RESULTS.md",
     "ucra_phi 0.804 / 0.741 / 0.823 for URLLC/eMBB/mMTC",
     "ucra_phi 0.836 / 0.782 / 0.844 for URLLC/eMBB/mMTC"),
    ("DEEP_RESULTS.md",
     "slice) in ~1 of 3.5 snapshots and eMBB half the time",
     "slice) in ~1 of 4 snapshots and eMBB nearly half the time"),
    ("DEEP_RESULTS.md",
     "URLLC still violates 15.2%.",
     "URLLC still violates 9.9%."),
    ("DEEP_RESULTS.md",
     "buffer is what buys 1.4% / 0.0%.",
     "buffer is what buys 1.0% / 1.2%."),
    ("DEEP_RESULTS.md",
     "- mMTC (4.4%) is the one slice where Phi does not dominate everything:\n"
     "  its offered distribution is heavy-tailed and spiky; the kappa sweep in\n"
     "  the same JSON shows the frontier if you want mMTC violations lower.",
     "- mMTC (1.6%): in the first partial-data run (120 snapshots) this was\n"
     "  the stubborn slice at 4.4%; on the full 250-snapshot set Phi dominates\n"
     "  here too. Its offered distribution stays heavy-tailed and spiky -- the\n"
     "  kappa sweep in the same JSON shows the frontier for tighter mMTC sizing."),
    # ---------- DEEP_CODE.md ----------
    ("DEEP_CODE.md",
     "#   -> per-slice table: operator 29/50.6/5.4 vs ucra_phi 1.4/0.0/4.4",
     "#   -> per-slice table: operator 26.2/44.2/5.4 vs ucra_phi 1.0/1.2/1.6"),
    # ---------- DEEP_DATA.md ----------
    ("DEEP_DATA.md",
     "evaluation consumed **120 samples** (70/30 chronological split).",
     "evaluation consumed **250 samples** (70/30 chronological split)."),
    # ---------- docs/07_data_sources.md ----------
    ("docs/07_data_sources.md",
     "URLLC 29%->1.4%, eMBB 50.6%->0% |",
     "URLLC 26%->1.0%, eMBB 44%->1.2% |"),
    ("docs/07_data_sources.md",
     "Your parse: 120 samples -> frame (2,908 rows, 8 cols); slice counts\n"
     "mMTC 1,201 / URLLC 1,106 / eMBB 601; test split (samples > 70% cut) yields\n"
     "URLLC n=276, eMBB n=164, mMTC n=296 -- the n's in your results JSON.",
     "Your parse: 250 samples -> frame (5,547 rows, 8 cols); slice counts\n"
     "mMTC 2,304 / URLLC 2,048 / eMBB 1,195; test split (samples > 70% cut) yields\n"
     "URLLC n=577, eMBB n=337, mMTC n=630 -- the n's in your results JSON.\n"
     "(An earlier partial-download run parsed 120 samples / n=276,164,296; the\n"
     "full-dataset numbers used everywhere in the docs supersede it.)"),
    # ---------- docs/09_results_interpretation.md ----------
    ("docs/09_results_interpretation.md",
     "eval (run_cttc_eval.py, 120 snapshots, 70/30)",
     "eval (run_cttc_eval.py, 250 snapshots, 70/30)"),
    ("docs/09_results_interpretation.md",
     "| URLLC | 276 | 29.0% viol / 56.1% util / 43.9% over | 15.2% / 46.6% / 53.4% | **1.4%** / 19.6% / 80.4% |\n"
     "| eMBB | 164 | 50.6% / 78.2% / 21.8% | 12.2% / 56.7% / 43.3% | **0.0%** / 25.9% / 74.1% |\n"
     "| mMTC | 296 | 5.4% / 15.3% / 84.7% | 12.2% / 30.6% / 69.4% | **4.4%** / 17.7% / 82.3% |",
     "| URLLC | 577 | 26.2% viol / 50.8% util / 49.2% over | 9.9% / 38.7% / 61.3% | **1.0%** / 16.4% / 83.6% |\n"
     "| eMBB | 337 | 44.2% / 73.0% / 27.0% | 5.3% / 41.0% / 59.0% | **1.2%** / 21.8% / 78.2% |\n"
     "| mMTC | 630 | 5.4% / 17.9% / 82.1% | 11.4% / 36.8% / 63.2% | **1.6%** / 15.6% / 84.4% |"),
    ("docs/09_results_interpretation.md",
     "under-reserves URLLC (29.0%) and eMBB\n"
     "  (50.6%) badly, while over-provisioning mMTC by 84.7% of its reservation.",
     "under-reserves URLLC (26.2%) and eMBB\n"
     "  (44.2%) badly, while over-provisioning mMTC by 82.1% of its reservation."),
    ("docs/09_results_interpretation.md",
     "still 12-15% violations",
     "still 5-11% violations"),
    ("docs/09_results_interpretation.md",
     "- **ucra_phi**: 1.4% / 0.0% / 4.4% violations. The cost: over-provision\n"
     "  ~74-82% of the reservation (note: relative to R, not to capacity), i.e.\n"
     "  utilization 18-26%.",
     "- **ucra_phi**: 1.0% / 1.2% / 1.6% violations. The cost: over-provision\n"
     "  ~78-84% of the reservation (note: relative to R, not to capacity), i.e.\n"
     "  utilization 16-22%."),
    ("docs/09_results_interpretation.md",
     "- **mMTC nuance worth stating honestly:** operator (5.4%) beats static_q90\n"
     "  (12.2%) here, because the dataset's mMTC delta reservations are already\n"
     "  generous relative to that slice's spiky-but-tiny loads. UCRA (4.4%)\n"
     "  edges out both, but the headline story on mMTC is \"comparable\n"
     "  violations\", not \"big win\".",
     "- **mMTC nuance worth stating honestly:** operator (5.4%) still beats\n"
     "  static_q90 (11.4%) here, because the dataset's mMTC delta reservations\n"
     "  are already generous relative to that slice's spiky-but-tiny loads. UCRA\n"
     "  (1.6%) now clearly beats both, but from an already-low 5.4% baseline --\n"
     "  the headline win on mMTC is modest next to URLLC/eMBB."),
    ("docs/09_results_interpretation.md",
     "operator-style violations from 29-51% to 0-4.4% on URLLC/eMBB at",
     "operator-style violations from 26-44% to 1.0-1.2% on URLLC/eMBB at"),
    # ---------- docs/02_pipeline_walkthrough.md ----------
    ("docs/02_pipeline_walkthrough.md",
     "### `python scripts/run_cttc_eval.py --config configs/default.yaml --max-samples 120`",
     "### `python scripts/run_cttc_eval.py --config configs/default.yaml`"),
    ("docs/02_pipeline_walkthrough.md",
     "Parses 120 CTTC steady-state snapshots (271 MB dataset, ~2,908 slice rows),",
     "Parses 250 CTTC steady-state snapshots (271 MB dataset, ~5,547 slice rows),"),
    ("docs/02_pipeline_walkthrough.md",
     "| run_cttc_eval --max-samples 120 | ~2-4 min (parsing dominates) |",
     "| run_cttc_eval (250 samples) | ~4-8 min (parsing dominates) |"),
    # ---------- docs/12_experiments_guide.md ----------
    ("docs/12_experiments_guide.md",
     "python scripts/run_cttc_eval.py --config configs/default.yaml --max-samples 120\n"
     "python -m pytest tests/test_smoke.py -q",
     "python scripts/run_cttc_eval.py --config configs/default.yaml  # default 250\n"
     "python -m pytest tests/test_smoke.py -q"),
    ("docs/12_experiments_guide.md",
     "demo updates [96, 120, 144]; CTTC URLLC 1.4% /\n"
     "eMBB 0.0% / mMTC 4.4%. Tolerances:",
     "demo updates [96, 120, 144]; CTTC URLLC 1.0% /\n"
     "eMBB 1.2% / mMTC 1.6%. Tolerances:"),
    ("docs/12_experiments_guide.md",
     "2. `--max-samples 250` on CTTC (the parser is linear; doubles the test\n"
     "   rows).",
     "2. DONE: the default CTTC run now parses 250 samples -- doc 09's numbers\n"
     "   already reflect this (up from 120 in the first partial-data run)."),
    # ---------- docs/13_faq_troubleshooting.md ----------
    ("docs/13_faq_troubleshooting.md",
     "A: Parsing dominates; use `--max-samples 120` (your current setting). 250\n"
     "is the ceiling we consider \"plenty\" (doc 12 section 9).",
     "A: Parsing dominates; the reference numbers use the default 250 samples\n"
     "(considered \"plenty\", doc 12 section 9). Use `--max-samples 100` for a\n"
     "quick pass -- numbers shift slightly with fewer snapshots."),
    # ---------- docs/14_paper_map.md ----------
    ("docs/14_paper_map.md",
     "CTTC slices: 29-51% -> 0-4.4%",
     "CTTC slices: 26-44% -> 1.0-1.6%"),
]


def apply_one(root: Path, path: str, old: str, new: str) -> str:
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


def main() -> int:
    root = Path(__file__).resolve().parent
    print("[sync] repo root:", root)
    ok = skip = bad = 0
    for path, old, new in PATCHES:
        status = apply_one(root, path, old, new)
        tag = "[ ok ]" if status == "patched" else \
              ("[skip]" if status.startswith("skip") else "[FAIL]")
        print("{} {:34s} {}".format(tag, path, status))
        if status == "patched":
            ok += 1
        elif status.startswith("skip"):
            skip += 1
        else:
            bad += 1
    print("-" * 60)
    print("[sync] patched {} | skipped {} | failed {}".format(ok, skip, bad))
    if bad:
        print("[sync] some patches did not apply - reply with this output.")
        return 1
    print("[sync] all docs now match outputs/cttc_results.json (250 snapshots).")
    print()
    print("Next (run one line at a time):")
    print("  git add -A")
    print('  git commit -m "docs: sync CTTC tables to full-dataset run (250 snapshots)"')
    print("  git push")
    return 0


if __name__ == "__main__":
    sys.exit(main())