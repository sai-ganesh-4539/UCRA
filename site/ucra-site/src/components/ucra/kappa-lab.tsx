"use client";

import { useMemo, useState } from "react";
import { motion } from "framer-motion";
import { Gauge, ShieldAlert, Boxes } from "lucide-react";
import { Reveal, SectionHeading, Panel } from "./shared";
import { CTTC, CFG, sweepAt, type SliceResult } from "@/lib/ucra";

const SLICE_COLOR: Record<string, string> = {
  URLLC: "#34d399",
  eMBB: "#fbbf24",
  mMTC: "#2dd4bf",
};

const SLICE_DESC: Record<string, string> = {
  URLLC: "Ultra-reliable low latency — tightest SLA, tiny payloads",
  eMBB: "Enhanced mobile broadband — heavy, bursty video traffic",
  mMTC: "Massive machine-type — huge device counts, small packets",
};

/* ------------- frontier chart ------------- */

function Frontier({
  slice,
  k,
}: {
  slice: SliceResult;
  k: number;
}) {
  const W = 560;
  const H = 360;
  const M = { l: 46, r: 18, t: 18, b: 42 };
  const yMax = Math.max(slice.operator.viol, slice.sweep[0].viol) + 8;
  const xMax = 100;

  const px = (over: number) => M.l + (over / xMax) * (W - M.l - M.r);
  const py = (viol: number) => M.t + (1 - viol / yMax) * (H - M.t - M.b);
  const color = SLICE_COLOR[slice.id] ?? "#34d399";

  const cur = sweepAt(slice.sweep, k);

  const dPath = slice.sweep
    .map((p, i) => `${i === 0 ? "M" : "L"}${px(p.over).toFixed(1)},${py(p.viol).toFixed(1)}`)
    .join(" ");

  return (
    <svg
      viewBox={`0 0 ${W} ${H}`}
      className="h-72 w-full sm:h-80"
      role="img"
      aria-label={`Safety-efficiency frontier for ${slice.id}`}
    >
      {/* grid */}
      {[0, 0.25, 0.5, 0.75, 1].map((f) => (
        <line
          key={f}
          x1={M.l}
          x2={W - M.r}
          y1={M.t + f * (H - M.t - M.b)}
          y2={M.t + f * (H - M.t - M.b)}
          stroke="#fff"
          strokeOpacity="0.05"
        />
      ))}

      {/* axes */}
      <line x1={M.l} x2={M.l} y1={M.t} y2={H - M.b} stroke="#3f3f46" />
      <line x1={M.l} x2={W - M.r} y1={H - M.b} y2={H - M.b} stroke="#3f3f46" />
      <text x={M.l - 8} y={M.t + 10} fill="#71717a" fontSize="11" textAnchor="end">
        viol %
      </text>
      <text x={W - M.r} y={H - M.b + 28} fill="#71717a" fontSize="11" textAnchor="end">
        over-provisioned capacity % (worse to the right)
      </text>
      {[0, Math.round(yMax / 2), Math.round(yMax)].map((v) => (
        <text key={v} x={M.l - 8} y={py(v) + 4} fill="#71717a" fontSize="10" textAnchor="end">
          {v}
        </text>
      ))}

      {/* frontier path */}
      <path d={dPath} fill="none" stroke={color} strokeWidth="2" strokeOpacity="0.9" />

      {/* measured points */}
      {slice.sweep.map((p, i) => {
        const prev = i > 0 ? px(slice.sweep[i - 1].over) : -100;
        const x = px(p.over);
        return (
          <g key={p.k}>
            <circle cx={x} cy={py(p.viol)} r="3.5" fill="#0b0f0d" stroke={color} strokeWidth="1.6" />
            {x - prev > 44 && (
              <text x={x} y={py(p.viol) - 8} fill="#a1a1aa" fontSize="9" textAnchor="middle">
                {"\u03BA"}={p.k}
              </text>
            )}
          </g>
        );
      })}

      {/* reference baselines */}
      <g>
        <line
          x1={px(slice.operator.over)}
          y1={py(slice.operator.viol) - 7}
          x2={px(slice.operator.over)}
          y2={py(slice.operator.viol) + 7}
          stroke="#fb7185"
          strokeWidth="2"
        />
        <line
          x1={px(slice.operator.over) - 7}
          y1={py(slice.operator.viol)}
          x2={px(slice.operator.over) + 7}
          y2={py(slice.operator.viol)}
          stroke="#fb7185"
          strokeWidth="2"
        />
        <text x={px(slice.operator.over) + 9} y={py(slice.operator.viol) + 4} fill="#fb7185" fontSize="10">
          operator
        </text>

        <line
          x1={px(slice.staticQ90.over)}
          y1={py(slice.staticQ90.viol) - 7}
          x2={px(slice.staticQ90.over)}
          y2={py(slice.staticQ90.viol) + 7}
          stroke="#a1a1aa"
          strokeWidth="2"
        />
        <line
          x1={px(slice.staticQ90.over) - 7}
          y1={py(slice.staticQ90.viol)}
          x2={px(slice.staticQ90.over) + 7}
          y2={py(slice.staticQ90.viol)}
          stroke="#a1a1aa"
          strokeWidth="2"
        />
        <text x={px(slice.staticQ90.over) + 9} y={py(slice.staticQ90.viol) + 4} fill="#a1a1aa" fontSize="10">
          static q90
        </text>
      </g>

      {/* current point */}
      <motion.g
        animate={{ x: px(cur.over), y: py(cur.viol) }}
        transition={{ type: "spring", stiffness: 260, damping: 26 }}
      >
        <circle r="12" fill={color} fillOpacity="0.15" />
        <circle r="5.5" fill={color} stroke="#0b0f0d" strokeWidth="2" />
      </motion.g>
    </svg>
  );
}

/* ------------- section ------------- */

export function KappaLab() {
  const [sliceId, setSliceId] = useState("URLLC");
  const [k, setK] = useState(CFG.kappa);
  const slice = CTTC.slices.find((s) => s.id === sliceId)!;
  const cur = useMemo(() => sweepAt(slice.sweep, k), [slice, k]);
  const color = SLICE_COLOR[slice.id];

  return (
    <section id="lab" className="relative py-20 sm:py-28">
      <div
        aria-hidden
        className="pointer-events-none absolute inset-x-0 top-0 h-px bg-gradient-to-r from-transparent via-emerald-400/30 to-transparent"
      />
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        <SectionHeading
          kicker="Interactive lab"
          title={<>The risk dial: pick your &kappa;</>}
          sub="Every measured operating point from the CTTC public slicing dataset is a dot on the frontier. Slide kappa to move UCRA between paranoid (0) and economical (2) — watch safety buy efficiency, or the reverse."
        />

        <Reveal className="mt-12">
          <Panel className="p-5 sm:p-7">
            {/* slice tabs */}
            <div className="flex flex-wrap items-center gap-2">
              {CTTC.slices.map((s) => (
                <button
                  key={s.id}
                  onClick={() => setSliceId(s.id)}
                  aria-pressed={sliceId === s.id}
                  className={`rounded-xl px-4 py-2 text-sm font-medium transition-colors ${
                    sliceId === s.id
                      ? "text-zinc-950"
                      : "border border-white/10 bg-white/[0.04] text-zinc-300 hover:bg-white/[0.09]"
                  }`}
                  style={
                    sliceId === s.id
                      ? { backgroundColor: SLICE_COLOR[s.id] }
                      : undefined
                  }
                >
                  {s.id}
                  <span className="ml-2 text-xs opacity-70">n={s.n}</span>
                </button>
              ))}
              <span className="ml-auto hidden text-xs text-zinc-500 sm:block">
                {SLICE_DESC[slice.id]}
              </span>
            </div>

            <div className="mt-6 grid gap-8 lg:grid-cols-2">
              {/* left: slider + readouts */}
              <div>
                <div className="flex items-baseline justify-between">
                  <label
                    htmlFor="kappa-slider"
                    className="text-sm font-medium text-zinc-300"
                  >
                    Risk multiplier &kappa;
                  </label>
                  <span
                    className="rounded-lg px-2.5 py-1 font-mono text-lg font-bold tabular-nums"
                    style={{ color, backgroundColor: `${color}1a` }}
                  >
                    {k.toFixed(2)}
                  </span>
                </div>
                <input
                  id="kappa-slider"
                  type="range"
                  min={0}
                  max={2}
                  step={0.01}
                  value={k}
                  onChange={(e) => setK(parseFloat(e.target.value))}
                  className="mt-3 w-full accent-emerald-400"
                  style={{ accentColor: color }}
                  aria-valuetext={`kappa ${k.toFixed(2)}`}
                />
                <div className="mt-1 flex justify-between font-mono text-[10px] text-zinc-600">
                  {[0, 0.25, 0.5, 0.75, 1, 1.5, 2].map((m) => (
                    <button
                      key={m}
                      onClick={() => setK(m)}
                      className="transition-colors hover:text-zinc-300"
                    >
                      {m}
                    </button>
                  ))}
                </div>
                <p className="mt-3 text-xs leading-relaxed text-zinc-500">
                  Reservation rule: R = q<sub>0.9</sub> + &kappa; &middot; spread
                  &middot; (1 + 0.3&rho;). Dots on the chart are the 7 measured
                  &kappa; settings from <code className="text-zinc-400">run_cttc_eval.py</code>; the
                  slider interpolates between them.
                </p>

                {/* readouts */}
                <div className="mt-5 grid grid-cols-3 gap-3">
                  <Readout
                    icon={ShieldAlert}
                    label="violations"
                    value={cur.viol}
                    tone={color}
                    compare={`operator ${slice.operator.viol}%`}
                  />
                  <Readout
                    icon={Gauge}
                    label="utilization"
                    value={cur.util}
                    tone={color}
                    compare={`operator ${slice.operator.util}%`}
                  />
                  <Readout
                    icon={Boxes}
                    label="over-provision"
                    value={cur.over}
                    tone={color}
                    compare={`static q90 ${slice.staticQ90.over}%`}
                  />
                </div>

                <div className="mt-5 rounded-xl border border-white/[0.06] bg-white/[0.02] p-4 text-xs leading-relaxed text-zinc-400">
                  <span className="font-semibold text-zinc-200">Reading it: </span>
                  low &kappa; keeps spare capacity tight but lets bursts through
                  (violations rise). High &kappa; makes the slice nearly
                  violation-free at the cost of idle headroom. UCRA&apos;s
                  published operating point is &kappa;=0.5.
                </div>
              </div>

              {/* right: frontier */}
              <div className="rounded-xl border border-white/[0.06] bg-black/20 p-2">
                <Frontier slice={slice} k={k} />
              </div>
            </div>
          </Panel>
        </Reveal>
      </div>
    </section>
  );
}

function Readout({
  icon: Icon,
  label,
  value,
  tone,
  compare,
}: {
  icon: typeof Gauge;
  label: string;
  value: number;
  tone: string;
  compare: string;
}) {
  return (
    <div className="rounded-xl border border-white/[0.06] bg-white/[0.02] p-3">
      <div className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wide text-zinc-500">
        <Icon className="h-3.5 w-3.5" style={{ color: tone }} />
        {label}
      </div>
      <motion.p
        key={Math.round(value * 10)}
        initial={{ opacity: 0.4 }}
        animate={{ opacity: 1 }}
        className="mt-1.5 text-xl font-bold tabular-nums text-zinc-50"
      >
        {value.toFixed(1)}%
      </motion.p>
      <p className="mt-1 text-[10px] leading-tight text-zinc-600">{compare}</p>
    </div>
  );
}
