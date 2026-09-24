# 15 -- Glossary (Plain Language, No Jargon Left Behind)

**URLLC / eMBB / mMTC** -- the three 5G slice families: ultra-reliable
low-latency (factory robots, remote surgery), enhanced mobile broadband
(video, browsing), massive machine-type communication (millions of tiny
sensors). Strictness order: URLLC > eMBB > mMTC.

**RAN** -- radio access network: base stations and their sectors, the part
of the network that talks to phones.

**PM counter** -- performance-management statistic a live network reports
every 15 min: data volumes, RB utilization, active users, CQI, ...

**RB / CQI / RRC / MIMO rank** -- resource block (radio spectrum unit);
channel-quality indicator (signal quality 1-15); radio connection state;
number of spatial streams. All appear in the RAN dataset columns.

**Sector** -- one antenna beam of a base station. Dataset_01: 75 of them.

**DL / UL** -- downlink / uplink traffic.

**Campaign gap** -- the multi-day hole between the dataset's observation
periods (Mar vs Oct 2023). Never model across it (doc 07).

**Slot** -- one 15-min decision step. The unit of everything in UCRA.

**Reservation R_t** -- capacity UCRA commits for slot t before demand is
realized. The central quantity of the whole project.

**Offered vs realized demand** -- CTTC: the traffic the flows ask for;
RAN: the data volume actually observed in the slot.

**delta (CTTC)** -- the slice's reserved share (0..1) in the dataset;
`reserved = delta x link_cap`.

**link_cap (CTTC)** -- mean edge `bandwidth` across the sample's topology
graph; our stand-in for slice capacity.

**Quantile / q_tau** -- the value below which tau of the probability mass
lies. q90 of demand = "level that demand stays under 90% of the time".

**Pinball loss** -- the training loss whose minimizer IS the tau-quantile
(doc 03 section 5). Asymmetric: under-prediction costs tau, over-prediction
costs 1-tau.

**Coverage** -- fraction of realized values inside the predicted
[q05, q99] band. Calibration check for Stage 1 (nominal 94%).

**Quantile crossing** -- when predicted q75 > q99 etc. (heads are
independent). Guarded, not structurally forbidden (doc 03 section 8).

**Z-space / z-score** -- subtract mean, divide by std. Two scalers: one
for inputs, one (scY) for targets. Training in z-space is what un-stalls
pinball learning (doc 03 section 3).

**Leakage** -- any future/test information sneaking into training. The
split is chronological and scalers are fit on train only, so there is
none (except the deliberately-cheating oracle).

**u_hat, U_t (spread), rho_t** -- the Stage 1 triple: point forecast
(median), uncertainty width (q99 - q50), and risk indicator from recent
violations + drift (doc 03 section 10).

**kappa** -- risk aversion: how many units of reservation per unit of
uncertainty spread. The frontier dial (doc 04 section 3).

**tau (base_tau)** -- which quantile Phi anchors on (0.9).

**Phi** -- the transformation (u_hat, U_t, rho_t) -> R_target with floor
and ceiling. The "U-to-R" in UCRA.

**Headroom / floor** -- min_headroom: R never drops below 1.05 x point
forecast.

**Ceiling (max_reserve_ratio)** -- R never exceeds 95% of capacity.

**EWMA** -- exponentially weighted moving average; Stage 4's smoother
(alpha 0.3: each step moves 30% toward the target).

**Hysteresis (release)** -- reservations shrink only when >10% of capacity
stayed idle AND the target is lower; and never below what was actually
used. Prevents oscillation (doc 04 section 4).

**Best-effort pool** -- 10% of R_t kept out of per-entity guarantees.

**Best-effort share / pressure exponent / risk exponent** -- Stage 3
weights: proportional-to-demand with an optional risk premium
(doc 05 section 2).

**Reservoir sampling** -- equal inclusion probability for every past item
in a fixed-size buffer; the anti-forgetting memory of Stage 5 (doc 06
section 3).

**Catastrophic forgetting** -- fine-tuning on only-new data erasing old
knowledge; the replay buffer prevents it.

**Drift / drift z-score** -- the world's statistics changing / how many
training-sigmas the recent demand mean is away from the training mean.

**Trigger** -- Stage 5 fires if rolling violations > 15% OR |z| > 3
(doc 06 section 2).

**Fine-tune** -- 3 gentle epochs (lr x 0.3) on the replay buffer.

**Oracle** -- a baseline that cheats (sees the test window) to define the
achievable envelope (doc 08 section 3).

**Frontier / knee** -- the (violation, utilization) trade-off curve and
its elbow: the cheapest kappa with ~0 violations.

**Capacity C** -- 1.3 x observed peak demand (heuristic stand-in for the
physical limit, doc 07 honesty box).

**Reservation error / utilization / waste / violation rate** -- the four
metrics (doc 08 section 1).

**md5 / HTTP Range** -- checksum for download verification / the header
that enables resuming partial downloads.

**CC-BY-4.0** -- the datasets' license: use freely WITH attribution (doc
07 section 6).

**Zenodo DOI** -- the persistent identifier of each dataset (10.5281/
zenodo.17815388 RAN, 10.5281/zenodo.10610616 CTTC).

**datanetAPI** -- the CTTC dataset's own Python reader; our loader wraps
it dict-tolerantly (doc 07 section 3).

**jsonpickle** -- the package datanetAPI.py imports; auto-installed by
the downloader.

**PPBP-flavoured bursts** -- Poisson Pareto burst process style:
sudden ON/OFF traffic surges with heavy-tailed lengths; what the
synthetic generator mimics.

**Checkpoint (model.pt)** -- saved network weights + architecture config;
loaded by run_ucra/demo/evolution so training happens once.

**Early stopping / patience** -- stop training when validation loss stops
improving for `patience` epochs; restore the best epoch.
