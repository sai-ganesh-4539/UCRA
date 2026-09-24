# 14 -- Paper Map: From Repo Artifacts to a Submission

## 1. Section plan with repo sources

| Paper section | Content | Source in repo |
|---|---|---|
| Abstract | 5-stage closed loop; real RAN: 0% viol @ 71.7% util; drift: -56% relative violations; CTTC slices: 29-51% -> 0-4.4% | README results; doc 09 |
| 1 Introduction | reservation dilemma; uncertainty as first-class input; contributions list | doc 01 sections 1-2 |
| 2 Related work | static sizing, point-forecast policies, quantile forecasting, continual learning; **naming-collision sentence below** | docs/METHODOLOGY.md; doc 01 section 6 |
| 3 Methodology | the five stages + feedback; equations 1-5 = phi_transform, update_reservation, rho_t, weights, trigger | docs 03-06; diagram in doc 01 section 3 |
| 4 Experimental setup | both datasets (provenance, license), capacity heuristic, splits, compute (CPU, 3 min) | doc 07; doc 02 |
| 5 Results | RAN table; kappa frontier; drift A/B; CTTC per-slice table | doc 09 tables |
| 6 Discussion & limitations | below | doc 09 section 4; doc 07 honesty box |
| Reproducibility statement | config + seeds + commit + Zenodo DOIs | doc 12 section 8, 10 |

**Required related-work sentence (acronym collision):** a 2020 IEEE paper
"User-Centric Context-Aware Resource Allocation for Network Slicing"
(IEEE Access 8, DOI 10.1109/ACCESS.2020.3046198) already uses "UCRA". Use:
*"Unlike the user-centric context-aware resource allocation of [2020],
UCRA here denotes an uncertainty-driven reservation transformation with
self-evolving risk adaptation; the two address complementary problems
(slice admission vs. capacity reservation)."* Or rename before submission
(U2RA / URTA / UnRes are free).

## 2. Figure inventory (PNG, 150 dpi, paper-ready)

| File | Suggested paper role |
|---|---|
| fig_demand_vs_reservation.png | Fig. 4: main qualitative result |
| fig_quantile_band.png | Fig. 3: Stage 1 uncertainty visualization |
| fig_kappa_sweep.png | Fig. 5: risk-utility frontier |
| fig_drift_demo.png | Fig. 6: Stage 5 A/B (two panels) |
| fig_cttc_slices.png | Fig. 7: per-slice policy comparison |
| fig_self_evolution.png | appendix or merge into Fig. 6 |
| history.png / audit_demand.png | appendix: training health / data overview |

## 3. Table inventory

| Table | Build from |
|---|---|
| T1 datasets | doc 07 sections 2-3 |
| T2 main RAN results | ucra_results.json (doc 09 section 1) |
| T3 drift demo | drift_demo.json (doc 09 section 2) |
| T4 CTTC per-slice | cttc_results.json (doc 09 section 3) |
| T5 kappa sweep | --sweep + cttc_results.json kappa_sweep (doc 12 section 4) |
| T6 ablations | doc 12 section 5 (config-only runs) |

## 4. Limitations to state (reviewers always ask)

1. **Capacity is a heuristic** (1.3 x observed peak) -- the PM dataset does
   not publish physical capacity; all normalized metrics share it, rankings
   are unaffected (doc 07).
2. **Single network segment / 63-hour test window** on the RAN side;
   generalization across sites is future work (doc 12 section 9 gives the
   extension path).
3. **CTTC snapshots are i.i.d.**, not a time series -- Stage 2-3 evidence
   there is distributional, not temporal; the rho term in the CTTC eval
   uses current offered load as a stress proxy (doc 09 section 3 caveat +
   the train-only variant recipe in doc 12 section 7).
4. **Quantile crossing is not enforced** structurally (doc 03 section 8).
5. **Stage 5 is trigger-based fine-tuning**, not a formal drift detector
   with guarantees; the demo is a controlled A/B, not a field study.
6. **Single-step reservations** (horizon step 0) -- multi-step lookahead is
   scaffolded (horizon=4) but unused.

## 5. Before you submit (repo polish checklist)

- [ ] Tag the exact commit used for the paper: `git tag v1.0-paper && git push --tags`
- [ ] Pin requirements (torch version you validated on) in requirements.txt
- [ ] Archive the repo on Zenodo (get a DOI; cite it in the paper)
- [ ] Fill the full author lists in the two BibTeX entries (doc 07 section 6)
- [ ] Decide the acronym (keep UCRA + collision sentence, or rename)
- [ ] Run the 3-seed variance pass and put mean +- sd in T2/T3 (doc 12
      section 8)
- [ ] Add the "Phi (train-only)" fourth policy to T4 (doc 12 section 7)
