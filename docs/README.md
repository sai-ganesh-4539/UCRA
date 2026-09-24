# UCRA Documentation -- Complete Guide Set

This folder explains **everything** in the UCRA repository: what the algorithm
is, how every stage works, why each design decision was made, what every file
does, what every number in your results means, and how to extend or defend the
work. Start here and follow one of the reading orders below.

The companion file [`METHODOLOGY.md`](METHODOLOGY.md) is the compact
diagram-block-to-code mapping table used when writing the paper.

## Suggested reading orders

**"I ran it -- now I want to understand what I built"** (recommended first pass)

    01_big_picture  ->  02_pipeline_walkthrough  ->  09_results_interpretation
    (glossary 15 open in a second tab at all times)

**"I need to defend this in a paper / to a reviewer"**

    01 -> 03 (Stage 1 math) -> 04 (Phi math) -> 08 (metrics + baselines)
    -> 09 (results) -> 14 (paper map + limitations) -> 07 (dataset provenance)

**"I want to modify or extend the code"**

    10 (code reference) -> 11 (config guide) -> 03/04/05/06 (stage internals)
    -> 12 (experiments guide)

**"Something broke"**

    13_faq_troubleshooting  -- every error we hit during the build is listed
    there with the exact fix.

## The 16 guides

| # | File | What it covers |
|---|------|----------------|
| 01 | [01_big_picture.md](01_big_picture.md) | The problem, the idea, the 5-stage loop, a slot-by-slot narrative walk-through |
| 02 | [02_pipeline_walkthrough.md](02_pipeline_walkthrough.md) | Every command, every output file, data-flow with the real numbers from your run |
| 03 | [03_stage1_uncertainty.md](03_stage1_uncertainty.md) | Quantile LSTM, pinball loss, z-space training, coverage, rho_t |
| 04 | [04_stage2_phi_stage4_update.md](04_stage2_phi_stage4_update.md) | The Phi equation term by term, worked example, EWMA + hysteresis release |
| 05 | [05_stage3_allocation.md](05_stage3_allocation.md) | Risk-adaptive allocation weights, best-effort pool, leftover recycling |
| 06 | [06_stage5_self_evolution.md](06_stage5_self_evolution.md) | Drift triggers, reservoir replay, fine-tuning, your drift-demo JSON decoded |
| 07 | [07_data_sources.md](07_data_sources.md) | Both Zenodo datasets: schemas, loaders, capacity estimate, citations, BibTeX |
| 08 | [08_metrics_baselines.md](08_metrics_baselines.md) | The 4 metrics and 3 baselines: formulas, intuition, what "good" looks like |
| 09 | [09_results_interpretation.md](09_results_interpretation.md) | YOUR numbers (RAN + drift demo + CTTC) explained cell by cell |
| 10 | [10_code_reference.md](10_code_reference.md) | Every module, class and function with signatures and gotchas |
| 11 | [11_config_guide.md](11_config_guide.md) | Every knob in configs/default.yaml + tuning recipes |
| 12 | [12_experiments_guide.md](12_experiments_guide.md) | Reproduction checklist, ablations, sweep grids, seed variance |
| 13 | [13_faq_troubleshooting.md](13_faq_troubleshooting.md) | All real errors hit during the build and their fixes |
| 14 | [14_paper_map.md](14_paper_map.md) | Repo artifacts mapped to paper sections, claims, limitations, naming note |
| 15 | [15_glossary.md](15_glossary.md) | Every term used anywhere, in plain language |

## How the docs were written

Every equation, file name, function signature and result number in these
guides was verified directly against the repository source code and the actual
run outputs (RAN pipeline, drift demo, CTTC evaluation) on the date of
generation. If you change the code, re-check docs 03-06 and 09 first.
