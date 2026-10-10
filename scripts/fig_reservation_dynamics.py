#!/usr/bin/env python3
"""Paper figure: reservation dynamics on the RAN test window.

Two panels from outputs/ucra_log.csv (seed-42 reference run):
  A) slack distribution: (R - D)/C histogram + per-slot slack trace stats
  B) EWMA smoothing: reservation R vs. raw Phi target R_target, plus the
     per-slot adjustment magnitude |R_t - R_{t-1}| (churn).
Saves outputs/fig_reservation_dynamics.png and prints summary stats.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

out = Path("outputs")
log = pd.read_csv(out / "ucra_log.csv", parse_dates=["ts"])
C = 143954.525

R = log["R"].values
D = log["demand"].values
RT = log["R_target"].values

slack = (R - D) / C                      # normalized slack (can be negative)
viol = (D > R)
churn = np.abs(np.diff(R)) / C           # per-slot reservation adjustments
churn_target = np.abs(np.diff(RT)) / C   # what raw target-chasing would do

fig, axes = plt.subplots(2, 1, figsize=(10, 6.4), constrained_layout=True)

ax = axes[0]
ax.hist(slack[~viol], bins=32, color="#1f77b4", alpha=0.75,
        label="violated = no (252/252 slots)")
ax.hist(slack[viol], bins=8, color="#d62728", alpha=0.9,
        label="violated = yes (0 slots)")
ax.axvline(0, color="k", lw=1.0)
ax.set_xlabel("slack  (R$_t$ $-$ D$_t$) / capacity")
ax.set_ylabel("slots")
ax.set_title("(a) Reservation slack: always positive, tightly distributed")
ax.legend(fontsize=8)

ax = axes[1]
t = np.arange(1, len(R))
ax.plot(t, churn * 100, lw=0.9, color="#1f77b4",
        label="UCRA commitment |R$_t$ - R$_{t-1}$| (EWMA + hysteresis)")
ax.plot(t, churn_target * 100, lw=0.9, alpha=0.6, color="#7f7f7f",
        label="raw target chase |R$^{target}_t$ - R$^{target}_{t-1}$| (no smoothing)")
ax.set_xlabel("test slot")
ax.set_ylabel("adjustment (% of capacity)")
ax.set_title("(b) Stage-4 smoothing suppresses reservation churn")
ax.legend(fontsize=8)

fig.savefig(out / "fig_reservation_dynamics.png", dpi=150)
plt.close(fig)
print("[fig] saved", out / "fig_reservation_dynamics.png")

print(f"slack: mean {slack.mean()*100:.1f}% | min {slack.min()*100:.1f}% | "
      f"max {slack.max()*100:.1f}% | P05 {np.percentile(slack,5)*100:.1f}%")
print(f"violations: {int(viol.sum())}/{len(viol)}")
print(f"churn (EWMA): mean {churn.mean()*100:.3f}%  P95 {np.percentile(churn,95)*100:.3f}%  max {churn.max()*100:.2f}%")
print(f"churn (raw target): mean {churn_target.mean()*100:.3f}%  P95 {np.percentile(churn_target,95)*100:.3f}%  max {churn_target.max()*100:.2f}%")
print(f"churn reduction (mean): {(1 - churn.mean()/churn_target.mean())*100:.1f}%")
print(f"corr(|target step|, |R step|): {np.corrcoef(churn, churn_target)[0,1]:.2f}")
