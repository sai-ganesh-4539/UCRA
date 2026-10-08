"use client";

import { motion } from "framer-motion";
import { ArrowDown, FlaskConical, Radio, Zap } from "lucide-react";
import { CountUp, Panel } from "./shared";
import { CAPACITY, N_SLOTS, RUN, TRAIN, CFG } from "@/lib/ucra";

/* Mini sparkline of the real logged run (demand vs reservation) */
function RunSparkline() {
  const W = 900;
  const H = 150;
  const max = 150000;
  const xs = (i: number) => (i / (N_SLOTS - 1)) * W;
  const ys = (v: number) => H - (v / max) * H;

  const path = (key: "demand" | "reservation") =>
    RUN.map((p, i) => `${i === 0 ? "M" : "L"}${xs(i).toFixed(1)},${ys(p[key]).toFixed(1)}`).join(" ");

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="h-28 w-full sm:h-36"
      preserveAspectRatio="none"
      aria-hidden
    >
      <defs>
        <linearGradient id="sparkFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#34d399" stopOpacity="0.18" />
          <stop offset="100%" stopColor="#34d399" stopOpacity="0" />
        </linearGradient>
      </defs>
      <path
        d={`${path("reservation")} L${W},${H} L0,${H} Z`}
        fill="url(#sparkFill)"
      />
      <path d={path("demand")} fill="none" stroke="#d4d4d8" strokeWidth="1.4" opacity="0.85" />
      <path d={path("reservation")} fill="none" stroke="#34d399" strokeWidth="1.8" />
    </svg>
  );
}

const STATS = [
  {
    label: "SLA violations",
    value: 0.0,
    suffix: "%",
    decimals: 1,
    hint: "252 real test slots",
  },
  {
    label: "Avg. utilization",
    value: 71.7,
    suffix: "%",
    decimals: 1,
    hint: "reserved capacity actually used",
  },
  {
    label: "90%-band coverage",
    value: TRAIN.coverage,
    suffix: "%",
    decimals: 1,
    hint: "calibrated quantile LSTM",
  },
  {
    label: "Model parameters",
    value: CFG.params,
    suffix: "",
    decimals: 0,
    hint: "2-layer quantile LSTM",
  },
];

export function Hero() {
  return (
    <section id="top" className="relative overflow-hidden pt-28 pb-16 sm:pt-36">
      {/* backdrop glow */}
      <div
        aria-hidden
        className="pointer-events-none absolute inset-0 grid-bg opacity-70"
      />
      <div
        aria-hidden
        className="pointer-events-none absolute -top-40 left-1/2 h-96 w-[42rem] -translate-x-1/2 rounded-full bg-emerald-500/10 blur-3xl"
      />

      <div className="relative mx-auto max-w-6xl px-4 sm:px-6">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7 }}
          className="mx-auto max-w-3xl text-center"
        >
          <span className="inline-flex items-center gap-2 rounded-full border border-emerald-400/25 bg-emerald-400/10 px-3 py-1 text-xs font-medium text-emerald-300">
            <Radio className="h-3.5 w-3.5" />
            Uncertainty-aware capacity reservation for network slicing
          </span>

          <h1 className="mt-6 text-4xl font-bold leading-[1.08] tracking-tight text-zinc-50 sm:text-6xl">
            Reserve the network
            <br />
            <span className="bg-gradient-to-r from-emerald-300 via-emerald-400 to-teal-300 bg-clip-text text-transparent">
              before demand happens.
            </span>
          </h1>

          <p className="mx-auto mt-6 max-w-2xl text-base leading-relaxed text-zinc-400 sm:text-lg">
            UCRA forecasts demand uncertainty with a quantile LSTM, transforms it
            into a risk-dialled reservation, and fine-tunes itself when traffic
            drifts. On the real 5G RAN trace it holds the SLA with zero
            violations — while wasting{" "}
            <span className="text-zinc-200">2.8&times; less capacity</span> than
            static peak provisioning.
          </p>

          <div className="mt-8 flex flex-wrap items-center justify-center gap-3">
            <a
              href="#lab"
              className="inline-flex items-center gap-2 rounded-xl bg-emerald-400 px-5 py-3 text-sm font-semibold text-emerald-950 shadow-lg shadow-emerald-500/20 transition-transform hover:scale-[1.03]"
            >
              <FlaskConical className="h-4 w-4" />
              Open the &kappa; lab
            </a>
            <a
              href="#drift"
              className="inline-flex items-center gap-2 rounded-xl border border-white/10 bg-white/[0.05] px-5 py-3 text-sm font-semibold text-zinc-100 transition-colors hover:bg-white/[0.1]"
            >
              <Zap className="h-4 w-4 text-amber-300" />
              Break it: inject a surge
            </a>
          </div>
        </motion.div>

        {/* stat cards */}
        <div className="mt-14 grid grid-cols-2 gap-3 sm:gap-4 lg:grid-cols-4">
          {STATS.map((s, i) => (
            <motion.div
              key={s.label}
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.15 + i * 0.08 }}
            >
              <Panel className="p-4 sm:p-5">
                <p className="text-2xl font-bold text-zinc-50 sm:text-3xl">
                  <CountUp
                    value={s.value}
                    decimals={s.decimals}
                    suffix={s.suffix}
                    className="tabular-nums"
                  />
                </p>
                <p className="mt-1 text-sm font-medium text-emerald-300">{s.label}</p>
                <p className="mt-0.5 text-xs text-zinc-500">{s.hint}</p>
              </Panel>
            </motion.div>
          ))}
        </div>

        {/* sparkline card */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.5 }}
        >
          <Panel className="mt-4 overflow-hidden">
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/[0.06] px-4 py-3 text-xs text-zinc-500 sm:px-5">
              <span className="font-medium text-zinc-300">
                Real logged run — {N_SLOTS} test slots, 15-min resolution
              </span>
              <span className="flex items-center gap-4">
                <span className="flex items-center gap-1.5">
                  <span className="h-0.5 w-4 rounded bg-zinc-300" /> demand
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="h-0.5 w-4 rounded bg-emerald-400" /> UCRA
                    reservation
                </span>
                <span>capacity {Math.round(CAPACITY).toLocaleString()}</span>
              </span>
            </div>
            <div className="px-2 pb-2 pt-3 sm:px-3">
              <RunSparkline />
            </div>
          </Panel>
        </motion.div>

        <div className="mt-8 flex justify-center">
          <a
            href="#problem"
            className="flex items-center gap-2 text-xs text-zinc-500 transition-colors hover:text-zinc-300"
          >
            <ArrowDown className="h-3.5 w-3.5 animate-bounce" />
            why this is hard
          </a>
        </div>
      </div>
    </section>
  );
}
