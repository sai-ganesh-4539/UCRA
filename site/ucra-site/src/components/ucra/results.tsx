"use client";

import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { CheckCircle2, CircleSlash } from "lucide-react";
import { Reveal, SectionHeading, Panel } from "./shared";
import { CAPACITY, N_SLOTS, RAN, RUN, TICK_LABELS, type MethodRow } from "@/lib/ucra";

const METHOD_COLOR: Record<string, string> = {
  emerald: "#34d399",
  amber: "#fbbf24",
  rose: "#fb7185",
  zinc: "#a1a1aa",
};

/* ---------------- Real run chart ---------------- */

function RunChart() {
  const W = 1000;
  const H = 360;
  const M = { l: 8, r: 8, t: 14, b: 26 };
  const yMax = 155000;

  const { demandPath, reservPath, capY, ticks } = useMemo(() => {
    const xs = (i: number) => M.l + (i / (N_SLOTS - 1)) * (W - M.l - M.r);
    const ys = (v: number) => M.t + (1 - v / yMax) * (H - M.t - M.b);
    const line = (key: "demand" | "reservation") =>
      RUN.map(
        (p, i) => `${i === 0 ? "M" : "L"}${xs(i).toFixed(1)},${ys(p[key]).toFixed(1)}`,
      ).join(" ");
    const ticks = Object.entries(TICK_LABELS).map(([t, label]) => ({
      x: xs(parseInt(t)),
      label,
    }));
    return {
      demandPath: line("demand"),
      reservPath: line("reservation"),
      capY: ys(CAPACITY),
      ticks,
    };
  }, []);

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="h-64 w-full sm:h-80"
      preserveAspectRatio="none"
      role="img"
      aria-label="Realized demand versus UCRA reservation over 252 test slots"
    >
      <defs>
        <linearGradient id="resFill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor="#34d399" stopOpacity="0.22" />
          <stop offset="100%" stopColor="#34d399" stopOpacity="0" />
        </linearGradient>
      </defs>

      {/* gridlines */}
      {[0.25, 0.5, 0.75].map((f) => {
        const y = M.t + f * (H - M.t - M.b);
        return (
          <line
            key={f}
            x1={M.l}
            x2={W - M.r}
            y1={y}
            y2={y}
            stroke="#ffffff"
            strokeOpacity="0.05"
            strokeWidth="1"
          />
        );
      })}

      {/* capacity */}
      <line
        x1={M.l}
        x2={W - M.r}
        y1={capY}
        y2={capY}
        stroke="#fbbf24"
        strokeOpacity="0.55"
        strokeWidth="1.2"
        strokeDasharray="6 5"
      />
      <text x={M.l + 6} y={capY - 6} fill="#fbbf24" fillOpacity="0.75" fontSize="11">
        cell capacity {Math.round(CAPACITY).toLocaleString()}
      </text>

      <path d={`${reservPath} L${W - M.r},${H - M.b} L${M.l},${H - M.b} Z`} fill="url(#resFill)" />
      <path d={demandPath} fill="none" stroke="#d4d4d8" strokeWidth="1.5" strokeOpacity="0.9" />
      <path d={reservPath} fill="none" stroke="#34d399" strokeWidth="1.9" />

      {ticks.map((tk) => (
        <text
          key={tk.label}
          x={tk.x}
          y={H - 8}
          fill="#71717a"
          fontSize="11"
          textAnchor="center"
        >
          {tk.label}
        </text>
      ))}
    </svg>
  );
}

/* ---------------- Metric bars ---------------- */

type MetricKey = "viol" | "waste" | "util";
const METRICS: { key: MetricKey; label: string; unit: string; lowerBetter: boolean; blurb: string }[] = [
  {
    key: "viol",
    label: "Safety",
    unit: "% slots violating the SLA",
    lowerBetter: true,
    blurb: "Share of 15-min slots where demand exceeded the reservation. Lower is better; 0% means the SLA held all week.",
  },
  {
    key: "waste",
    label: "Efficiency",
    unit: "% capacity wasted",
    lowerBetter: true,
    blurb: "Reserved-but-unused capacity relative to the cell. This is the money metric — idle radio is sunk cost.",
  },
  {
    key: "util",
    label: "Utilization",
    unit: "% of reservation used",
    lowerBetter: false,
    blurb: "How much of the reservation real demand actually consumed. High utilization with 0 violations is the sweet spot.",
  },
];

function MetricBars() {
  const [metric, setMetric] = useState<MetricKey>("viol");
  const active = METRICS.find((m) => m.key === metric)!;
  const max = Math.max(...RAN.rows.map((r) => r[metric]), 1);

  return (
    <Panel className="p-5 sm:p-6">
      <div className="flex flex-wrap items-center gap-2">
        {METRICS.map((m) => (
          <button
            key={m.key}
            onClick={() => setMetric(m.key)}
            aria-pressed={metric === m.key}
            className={`rounded-lg px-3 py-1.5 text-sm font-medium transition-colors ${
              metric === m.key
                ? "bg-emerald-400 text-emerald-950"
                : "border border-white/10 bg-white/[0.04] text-zinc-300 hover:bg-white/[0.09]"
            }`}
          >
            {m.label}
          </button>
        ))}
      </div>
      <p className="mt-2 text-xs text-zinc-500">{active.blurb}</p>

      <div className="mt-5 space-y-4">
        {RAN.rows.map((r) => (
          <BarRow key={r.id} row={r} metric={metric} max={max} />
        ))}
      </div>
      <p className="mt-4 text-xs text-zinc-500">{active.unit}</p>
    </Panel>
  );
}

function BarRow({
  row,
  metric,
  max,
}: {
  row: MethodRow;
  metric: MetricKey;
  max: number;
}) {
  const color = METHOD_COLOR[row.accent];
  const val = row[metric];
  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between gap-2 text-sm">
        <span className="font-medium text-zinc-200">{row.name}</span>
        <span className="tabular-nums text-zinc-400">
          {val.toFixed(1)}%
        </span>
      </div>
      <div className="h-2.5 overflow-hidden rounded-full bg-white/[0.06]">
        <motion.div
          className="h-full rounded-full"
          style={{ backgroundColor: color }}
          initial={{ width: 0 }}
          whileInView={{ width: `${Math.max(2, (val / max) * 100)}%` }}
          viewport={{ once: true }}
          transition={{ duration: 0.9, ease: "easeOut" }}
        />
      </div>
    </div>
  );
}

/* ---------------- Section ---------------- */

export function Results() {
  return (
    <section id="results" className="relative py-20 sm:py-28">
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        <SectionHeading
          kicker="Head-to-head on the RAN trace"
          title="The receipts: one week, 252 slots, four policies"
          sub="Real sector-level 5G PM counters (75 sectors aggregated, 15-min slots). UCRA's reservation hugs demand — never under it, never far above."
        />

        <Reveal className="mt-12">
          <Panel className="overflow-hidden">
            <div className="flex flex-wrap items-center justify-between gap-2 border-b border-white/[0.06] px-4 py-3 text-xs sm:px-5">
              <span className="font-medium text-zinc-300">
                Realized demand vs UCRA reservation
              </span>
              <span className="flex items-center gap-4 text-zinc-500">
                <span className="flex items-center gap-1.5">
                  <span className="h-0.5 w-4 rounded bg-zinc-300" /> demand
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="h-0.5 w-4 rounded bg-emerald-400" /> reservation
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="h-0.5 w-4 rounded bg-amber-300/70" /> capacity
                </span>
              </span>
            </div>
            <div className="px-2 py-3 sm:px-3">
              <RunChart />
            </div>
            <div className="border-t border-white/[0.06] px-4 py-3 text-xs text-zinc-500 sm:px-5">
              Zero violating slots in the whole window — the reservation (green)
              tracks the diurnal cycle and always stays above realized demand
              (grey) yet well below the static-peak strategy&apos;s flat
              over-provisioning.
            </div>
          </Panel>
        </Reveal>

        <div className="mt-6 grid gap-6 lg:grid-cols-5">
          <Reveal className="lg:col-span-3" delay={0.05}>
            <MetricBars />
          </Reveal>

          <Reveal className="lg:col-span-2" delay={0.1}>
            <Panel className="h-full p-5 sm:p-6">
              <h3 className="text-sm font-semibold uppercase tracking-wide text-zinc-400">
                Scorecard
              </h3>
              <div className="mt-4 space-y-3">
                {RAN.rows.map((r) => (
                  <div
                    key={r.id}
                    className="flex items-start justify-between gap-3 rounded-xl border border-white/[0.06] bg-white/[0.02] p-3"
                  >
                    <div className="flex items-start gap-2.5">
                      {r.id === "ucra" ? (
                        <CheckCircle2 className="mt-0.5 h-4 w-4 shrink-0 text-emerald-400" />
                      ) : (
                        <CircleSlash className="mt-0.5 h-4 w-4 shrink-0 text-zinc-600" />
                      )}
                      <div>
                        <p className="text-sm font-medium text-zinc-100">
                          {r.name}
                        </p>
                        <p className="text-xs text-zinc-500">{r.verdict}</p>
                      </div>
                    </div>
                    <div className="shrink-0 text-right text-xs tabular-nums text-zinc-400">
                      <p>
                        <span className="text-zinc-500">viol</span>{" "}
                        {r.viol.toFixed(1)}%
                      </p>
                      <p>
                        <span className="text-zinc-500">waste</span>{" "}
                        {r.waste.toFixed(1)}%
                      </p>
                      <p>
                        <span className="text-zinc-500">util</span>{" "}
                        {r.util.toFixed(1)}%
                      </p>
                    </div>
                  </div>
                ))}
              </div>
              <p className="mt-4 text-xs leading-relaxed text-zinc-500">
                Static peak matches UCRA&apos;s zero violations but leaves{" "}
                <span className="text-amber-300">30.5%</span> of the cell idle.
                Mean forecast is lean yet misses half of all slots. UCRA is the
                only policy that is simultaneously safe and efficient.
              </p>
            </Panel>
          </Reveal>
        </div>
      </div>
    </section>
  );
}
