import ucraResults from "./ucra_results.json";
import cttcResults from "./cttc_results.json";
import driftDemo from "./drift_demo.json";
import trainMetrics from "./train_metrics.json";
import seriesJson from "./ucra-series.json";

/* ------------------------------------------------------------------ */
/* Types                                                               */
/* ------------------------------------------------------------------ */

export interface RunPoint {
  t: number;
  demand: number;
  reservation: number;
  uHat: number;
  spread: number;
}

export interface DriftPoint {
  violFrozen: number;
  violEvolving: number;
  rollFrozen: number | null;
  rollEvolving: number | null;
  demandDrift: number;
}

export interface SweepPoint {
  k: number;
  viol: number; // %
  util: number; // %
  over: number; // %
}

export interface SliceResult {
  id: string;
  n: number;
  operator: { viol: number; util: number; over: number };
  staticQ90: { viol: number; util: number; over: number };
  ucra: { viol: number; util: number; over: number };
  sweep: SweepPoint[];
}

/* ------------------------------------------------------------------ */
/* Real logged run (ucra_log.csv - 252 test slots)                     */
/* ------------------------------------------------------------------ */

const meta = seriesJson.meta as {
  n: number;
  capacity: number;
  surgeStart: number;
  shift: number;
  updates: number[];
  frozenCalib: number;
  evolvingScales: number[];
  frozenPostCount: number;
  evolvingCount: number;
  tickLabels: Record<string, string>;
};

export const N_SLOTS = meta.n;
export const CAPACITY = meta.capacity;
export const SURGE_START = meta.surgeStart;
export const SHIFT = meta.shift;
export const UPDATES = meta.updates;
export const TICK_LABELS = meta.tickLabels;

export const RUN: RunPoint[] = seriesJson.log.t.map((t: number, i: number) => ({
  t,
  demand: seriesJson.log.demand[i],
  reservation: seriesJson.log.reservation[i],
  uHat: seriesJson.log.uHat[i],
  spread: seriesJson.log.spread[i],
}));

export const DRIFT: DriftPoint[] = seriesJson.drift.violFrozen.map(
  (_: number, i: number) => ({
    violFrozen: seriesJson.drift.violFrozen[i],
    violEvolving: seriesJson.drift.violEvolving[i],
    rollFrozen: seriesJson.drift.rollFrozen[i],
    rollEvolving: seriesJson.drift.rollEvolving[i],
    demandDrift: seriesJson.drift.demandDrift[i],
  }),
);

/** Reservation path the frozen agent would hold under surge (calibrated). */
export const FROZEN_R = RUN.map((p) => p.reservation * meta.frozenCalib);
const segScale = (t: number) =>
  t < UPDATES[0] ? 0 : t < UPDATES[1] ? 1 : t < UPDATES[2] ? 2 : 3;
export const EVOLVING_R = RUN.map(
  (p, t) => p.reservation * meta.frozenCalib * meta.evolvingScales[segScale(t)],
);

/* ------------------------------------------------------------------ */
/* RAN head-to-head (ucra_results.json)                                */
/* ------------------------------------------------------------------ */

const pct = (x: number) => Math.round(x * 1000) / 10;

export type MethodId = "ucra" | "static_peak" | "mean_forecast" | "oracle";

export interface MethodRow {
  id: MethodId;
  name: string;
  viol: number; // % of slots violated
  util: number; // % of reservation actually used
  waste: number; // % of capacity wasted
  verdict: string;
  accent: string; // tailwind color token base
}

export const RAN: { rows: MethodRow[]; capacity: number } = {
  capacity: pct(ucraResults.capacity),
  rows: [
    {
      id: "ucra",
      name: "UCRA",
      viol: pct(ucraResults.ucra.violation_rate),
      util: pct(ucraResults.ucra.utilization),
      waste: pct(ucraResults.ucra.waste),
      verdict: "Safe and efficient",
      accent: "emerald",
    },
    {
      id: "static_peak",
      name: "Static peak",
      viol: pct(ucraResults.static_peak.violation_rate),
      util: pct(ucraResults.static_peak.utilization),
      waste: pct(ucraResults.static_peak.waste),
      verdict: "Safe, but burns capacity",
      accent: "amber",
    },
    {
      id: "mean_forecast",
      name: "Mean forecast",
      viol: pct(ucraResults.mean_forecast.violation_rate),
      util: pct(ucraResults.mean_forecast.utilization),
      waste: pct(ucraResults.mean_forecast.waste),
      verdict: "Lean, but breaks the SLA",
      accent: "rose",
    },
    {
      id: "oracle",
      name: "Oracle quantile",
      viol: pct(ucraResults.oracle_quantile.violation_rate),
      util: pct(ucraResults.oracle_quantile.utilization),
      waste: pct(ucraResults.oracle_quantile.waste),
      verdict: "Sees the future - still 16.3%",
      accent: "zinc",
    },
  ],
};

/* ------------------------------------------------------------------ */
/* CTTC public-data validation (cttc_results.json)                     */
/* ------------------------------------------------------------------ */

const sweepOf = (sw: Record<string, { violation_rate: number; utilization: number; over_provision: number }>): SweepPoint[] =>
  Object.entries(sw)
    .map(([k, v]) => ({
      k: parseFloat(k),
      viol: pct(v.violation_rate),
      util: pct(v.utilization),
      over: pct(v.over_provision),
    }))
    .sort((a, b) => a.k - b.k);

const sliceOf = (id: string, d: (typeof cttcResults)["URLLC"]): SliceResult => ({
  id,
  n: d.ucra_phi.n,
  operator: {
    viol: pct(d.operator.violation_rate),
    util: pct(d.operator.utilization),
    over: pct(d.operator.over_provision),
  },
  staticQ90: {
    viol: pct(d.static_q90.violation_rate),
    util: pct(d.static_q90.utilization),
    over: pct(d.static_q90.over_provision),
  },
  ucra: {
    viol: pct(d.ucra_phi.violation_rate),
    util: pct(d.ucra_phi.utilization),
    over: pct(d.ucra_phi.over_provision),
  },
  sweep: sweepOf(d.kappa_sweep),
});

export const CTTC: { slices: SliceResult[]; snapshots: number } = {
  slices: [
    sliceOf("URLLC", cttcResults.URLLC),
    sliceOf("eMBB", cttcResults.eMBB),
    sliceOf("mMTC", cttcResults.mMTC),
  ],
  snapshots: 250,
};

/* ------------------------------------------------------------------ */
/* Drift demo + training (drift_demo.json / train_metrics.json)        */
/* ------------------------------------------------------------------ */

const dd = driftDemo as {
  frozen: { viol_post_surge: number; utilization_post_surge: number; viol_overall: number };
  evolving: {
    viol_post_surge: number;
    utilization_post_surge: number;
    viol_overall: number;
    updates: number;
  };
};

export const DRIFT_KPI = {
  frozenPost: pct(dd.frozen.viol_post_surge),
  evolvingPost: pct(dd.evolving.viol_post_surge),
  frozenUtil: pct(dd.frozen.utilization_post_surge),
  evolvingUtil: pct(dd.evolving.utilization_post_surge),
  frozenOverall: pct(dd.frozen.viol_overall),
  evolvingOverall: pct(dd.evolving.viol_overall),
  updates: dd.evolving.updates,
};

export const TRAIN = {
  pinball: Math.round(trainMetrics.test_pinball * 10) / 10,
  coverage: pct(trainMetrics.coverage_90pct),
  nTrain: trainMetrics.n_train,
  nTest: trainMetrics.n_test,
};

/* ------------------------------------------------------------------ */
/* Config constants (configs/default.yaml)                             */
/* ------------------------------------------------------------------ */

export const CFG = {
  quantiles: [0.05, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99],
  baseTau: 0.9,
  kappa: 0.5,
  rhoWeight: 0.3,
  minHeadroom: 0.05,
  maxReserve: 0.95,
  ewmaAlpha: 0.3,
  releaseHysteresis: 0.1,
  driftWindow: 96,
  violationTrigger: 0.15,
  replaySize: 512,
  finetuneEpochs: 3,
  params: 52252,
  seqLen: 96,
  hidden: 64,
  layers: 2,
};

export const DATASETS = {
  ran: { rows: 302052, sectors: 75, slots: 1778, doi: "Zenodo 17815388" },
  cttc: { snapshots: 250, doi: "Zenodo 10610616" },
};

/** Replace with the real repository URL before publishing. */
export const REPO_URL = "https://github.com/your-username/UCRA";

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

/** Linear interpolation across measured kappa sweep points. */
export function sweepAt(sweep: SweepPoint[], k: number): SweepPoint {
  if (k <= sweep[0].k) return sweep[0];
  const last = sweep[sweep.length - 1];
  if (k >= last.k) return last;
  let i = 0;
  while (i < sweep.length - 1 && sweep[i + 1].k < k) i++;
  const a = sweep[i];
  const b = sweep[i + 1];
  const f = (k - a.k) / (b.k - a.k);
  return {
    k,
    viol: a.viol + f * (b.viol - a.viol),
    util: a.util + f * (b.util - a.util),
    over: a.over + f * (b.over - a.over),
  };
}

export const fmt1 = (x: number) =>
  (Math.round(x * 10) / 10).toFixed(1).replace(/\.0$/, x === 0 ? ".0" : "");
