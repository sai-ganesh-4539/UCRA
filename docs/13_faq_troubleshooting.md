# 13 -- FAQ and Troubleshooting (Every Real Error We Hit)

These are not hypothetical: each item below actually happened during the
build/validation of this repo. Find your symptom, get the cause and the fix.

## Data & downloads

**Q: The download keeps dying / resumes from scratch.**
A: It does not -- `download_data.py` resumes `.part` files via HTTP Range
requests (we tested a 3 MB partial resume) and retries 6 times. Just re-run
the same command. If a file is complete it is skipped via size + md5.

**Q: `pip` says "Defaulting to user installation because normal
site-packages is not writeable".**
A: Harmless Windows notice; the package installs into your user site and
works.

**Q: `[ran_loader] skip (no tech columns): <file>.csv`**
A: Expected. Some CSVs in the archive do not match the tech-prefixed PM
schema; the loader skips them loudly and continues. Only worry if ALL files
are skipped -> then you are pointing at the wrong data root.

**Q: `FileNotFoundError: No dataset files under data/ran...`**
A: Run `python scripts/download_data.py` from the repo root first (or
check `--root`).

**Q: Why is my series only 1,778 slots when the audit says 302,052 rows?**
A: 302,052 = rows across ALL sectors x technologies x campaigns. After
summing sectors and cutting at the 193-day campaign gap, the longest
gap-free 15-min segment is 1,778 slots. The gap cutting is deliberate
(doc 07 section 2) -- do not "fix" it.

## CTTC

**Q: `'dict' object has no attribute 'type'` (crash inside datanetAPI).**
A: The dataset build ships plain dicts where its own reader expects
jsonpickle objects. Fixed: `ucra/data/cttc_loader.py` reads both shapes.
If you see this, your copy of the loader predates Part D -- re-run
`create_ucra_partD.py`.

**Q: `ImportError: datanetAPI.py must sit inside the unzipped dataset
folder`.**
A: Run `python scripts/download_data.py --cttc` (it downloads, unzips, and
pip-installs jsonpickle), then re-run the eval.

**Q: The class is `datanetAPI` or `DatanetAPI`?**
A: `DatanetAPI` (capital D). The lowercase import was a real bug we hit and
fixed in Part C.

**Q: CTTC eval is slow / memory heavy.**
A: Parsing dominates; use `--max-samples 120` (your current setting). 250
is the ceiling we consider "plenty" (doc 12 section 9).

## Training (Stage 1)

**Q: Training stalls -- loss flat around 60-ish, predictions near zero.**
A: You are training on RAW-scale targets. The pinball loss has bounded
gradients; at target scale ~50,000 it cannot climb. The repo trains in
z-space (scY) and inverts afterwards -- keep it that way. This was real
bug #2 of the project.

**Q: `UserWarning: ... variable needed for gradient computation ...
tl += float(l) * len(xb)`.**
A: Historical; fixed via `float(l.detach())` in Part D. If you still see
it, your quantile_lstm.py predates Part D.

**Q: Coverage came out 96.0% but a friend running the same repo gets
93.25%.**
A: Cross-platform torch nondeterminism (2.13 Windows vs 2.14 Linux in our
case), same seed. Both healthy. Report mean +- sd over seeds (doc 12
section 8).

**Q: I changed model config but results look like the old model.**
A: `outputs/model.pt` still exists, so run_ucra loads it. Delete the
checkpoint and retrain (doc 11, last bullet).

## Running the loop

**Q: `evolution_updates: 0` -- is Stage 5 broken?**
A: No. Zero violations + no >3-sigma drift -> the trigger never fires. To
see Stage 5 work, run `scripts/demo_evolution.py` (doc 06).

**Q: Stage 5 never fired in the demo either / fired at odd times.**
A: Trigger checks happen every `eval_every` slots and need the 96-slot
rolling violation window to exceed 0.15. With the default demo cadence
(24) on a +35% surge, updates land at [96, 120, 144]. See the derivation
in doc 06 section 6.

**Q: Violations cluster where `R_target` was already high but `R` lagged.**
A: Stage 4 smoothing too slow -> lower `update.ewma_alpha` (diagnose via
ucra_log.csv columns R vs R_target, doc 04 section 5).

## Git / environment (Windows)

**Q: `git push` fails with a network/DNS error but PyPI works.**
A: Transient DNS. Retry later (it worked for you on the next push:
cf334f1 and cdde350 are on origin/main).

**Q: PowerShell vs bash commands in the README.**
A: Everything repo-provided is pure Python on purpose: use
`python scripts/download_data.py` (not the .sh). The .sh exists only for
Unix convenience.

## Reproducibility sanity checks

- Same config + same checkpoint -> identical loop results (Stages 2-4 are
  deterministic numpy).
- Fresh train on another machine -> coverage within a few points; loop
  metrics nearly identical because Phi reacts to quantile SHAPES, which
  are stable.
- `python -m pytest tests/test_smoke.py -q` must always pass -- it
  exercises every stage in seconds on synthetic data.
