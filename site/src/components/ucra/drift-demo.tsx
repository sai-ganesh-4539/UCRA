"use client";

import { useEffect, useMemo, useRef, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Zap, BrainCircuit, RotateCcw, TrendingDown } from "lucide-react";
import { Reveal, SectionHeading, Panel, CountUp } from "./shared";
import {
  CAPACITY,
  CFG,
  DRIFT,
  DRIFT_KPI,
  EVOLVING_R,
  FROZEN_R,
  N_SLOTS,
  RUN,
  SURGE_START,
  TICK_LABELS,
  UPDATES,
} from "@/lib/ucra";

type Phase = "idle" | "surging" | "frozenDone" | "evolving" | "done";

const SPEED_MS = 14; // per 2-slot step
const W = 1000;
const H_TOP = 210;
const H_BOT = 190;
const M = { l: 8, r: 8, t: 12, b: 22 };
const Y_MAX = 155000;
const TRIGGER = CFG.violationTrigger * 100;

function xs(i: number) {
  return M.l + (i / (N_SLOTS - 1)) * (W - M.l - M.r);
}
function ysTop(v: number) {
  return M.t + (1 - v / Y_MAX) * (H_TOP - M.t - M.b);
}

function line(arr: (number | null)[], y: (v: number) => number, upto: number) {
  let d = "";
  let started = false;
  for (let i = 0; i <= upto && i < arr.length; i++) {
    const v = arr[i];
    if (v == null) continue;
    d += `${started ? "L" : "M"}${xs(i).toFixed(1)},${y(v).toFixed(1)} `;
    started = true;
  }
  return d.trim();
}

export function DriftDemo() {
  const [phase, setPhase] = useState<Phase>("idle");
  const [slot, setSlot] = useState(0);
  const timer = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopTimer = () => {
    if (timer.current) {
      clearInterval(timer.current);
      timer.current = null;
    }
  };

  useEffect(() => stopTimer, []);

  useEffect(() => {
    if (phase !== "surging" && phase !== "evolving") return;
    timer.current = setInterval(() => {
      setSlot((s) => {
        if (s + 2 >= N_SLOTS) {
          stopTimer();
          setPhase(phase === "surging" ? "frozenDone" : "done");
          return N_SLOTS - 1;
        }
        return s + 2;
      });
    }, SPEED_MS);
    return stopTimer;
  }, [phase]);

  const start = (p: Phase) => {
    stopTimer();
    setSlot(p === "surging" ? SURGE_START : 0);
    setPhase(p);
  };

  const reset = () => {
    stopTimer();
    setPhase("idle");
    setSlot(0);
  };

  const upto = phase === "idle" ? SURGE_START : slot;
  const surged = phase !== "idle";
  const showEvolving = phase === "evolving" || phase === "done";

  const demandPath = useMemo(
    () => line(surged ? DRIFT.map((d) => d.demandDrift) : RUN.map((r) => r.demand), ysTop, upto),
    [surged, upto],
  );
  const frozenRPath = useMemo(() => line(FROZEN_R, ysTop, upto), [upto]);
  const evolRPath = useMemo(
    () => (showEvolving ? line(EVOLVING_R, ysTop, upto) : ""),
    [showEvolving, upto],
  );

  const ysBot = (v: number) => M.t + (1 - v / 32) * (H_BOT - M.t - M.b);
  const rollF = useMemo(
    () => line(DRIFT.map((d) => (d.rollFrozen == null ? null : d.rollFrozen * 100)), ysBot, upto),
    [upto],
  );
  const rollE = useMemo(
    () =>
      showEvolving
        ? line(DRIFT.map((d) => (d.rollEvolving == null ? null : d.rollEvolving * 100)), ysBot, upto)
        : "",
    [showEvolving, upto],
  );

  return (
    <section id="drift" className="relative py-20 sm:py-28">
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        <SectionHeading
          kicker="Stage 5 — self-evolution"
          title="Break it live: +35% flash crowd"
          sub="At slot 37 of the test window, demand jumps by a third — a stadium event, a viral stream. A frozen model keeps under-reserving. UCRA's drift monitor catches it, fine-tunes, and recovers."
        />

        <Reveal className="mt-12">
          <Panel className="overflow-hidden">
            {/* control bar */}
            <div className="flex flex-wrap items-center gap-3 border-b border-white/[0.06] px-4 py-3 sm:px-5">
              {phase === "idle" && (
                <button
                  onClick={() => start("surging")}
                  className="inline-flex items-center gap-2 rounded-xl bg-rose-400 px-4 py-2 text-sm font-semibold text-rose-950 shadow-lg shadow-rose-500/20 transition-transform hover:scale-[1.03]"
                >
                  <Zap className="h-4 w-4" />
                  Inject +35% surge
                </button>
              )}
              {phase === "surging" && (
                <span className="inline-flex items-center gap-2 text-sm text-rose-300">
                  <span className="h-2 w-2 animate-ping rounded-full bg-rose-400" />
                  flash crowd running — slot {slot}/{N_SLOTS}
                </span>
              )}
              {phase === "frozenDone" && (
                <>
                  <button
                    onClick={() => start("evolving")}
                    className="inline-flex items-center gap-2 rounded-xl bg-emerald-400 px-4 py-2 text-sm font-semibold text-emerald-950 shadow-lg shadow-emerald-500/20 transition-transform hover:scale-[1.03]"
                  >
                    <BrainCircuit className="h-4 w-4" />
                    Now enable Stage-5 self-evolution
                  </button>
                  <button
                    onClick={reset}
                    className="inline-flex items-center gap-2 rounded-xl border border-white/10 px-3 py-2 text-sm text-zinc-400 hover:bg-white/[0.06]"
                  >
                    <RotateCcw className="h-3.5 w-3.5" /> Reset
                  </button>
                </>
              )}
              {phase === "evolving" && (
                <span className="inline-flex items-center gap-2 text-sm text-emerald-300">
                  <span className="h-2 w-2 animate-ping rounded-full bg-emerald-400" />
                  Stage-5 live — drift trigger at {TRIGGER}% &middot; fine-tuning
                </span>
              )}
              {phase === "done" && (
                <button
                  onClick={reset}
                  className="inline-flex items-center gap-2 rounded-xl border border-white/10 px-3 py-2 text-sm text-zinc-300 hover:bg-white/[0.06]"
                >
                  <RotateCcw className="h-3.5 w-3.5" /> Replay
                </button>
              )}
              <span className="ml-auto text-xs text-zinc-500">
                surge at slot {SURGE_START} &middot; +{Math.round(35)}% demand
              </span>
            </div>

            {/* charts */}
            <div className="space-y-1 px-2 py-3 sm:px-3">
              {/* top: demand vs reservations */}
              <p className="px-2 text-[11px] font-medium uppercase tracking-wide text-zinc-500">
                demand vs reservation
              </p>
              <svg
                viewBox={`0 0 ${W} ${H_TOP}`}
                className="h-40 w-full sm:h-48"
                preserveAspectRatio="none"
                aria-label="Drifted demand against frozen and evolving reservations"
              >
                <line x1={xs(SURGE_START)} x2={xs(SURGE_START)} y1={M.t} y2={H_TOP - M.b} stroke="#fb7185" strokeDasharray="4 4" strokeOpacity="0.6" />
                <line x1={M.l} x2={W - M.r} y1={ysTop(CAPACITY)} y2={ysTop(CAPACITY)} stroke="#fbbf24" strokeOpacity="0.4" strokeDasharray="6 5" strokeWidth="1.1" />
                <path d={frozenRPath} fill="none" stroke="#fb7185" strokeWidth="1.7" strokeOpacity="0.85" />
                {evolRPath && (
                  <path d={evolRPath} fill="none" stroke="#34d399" strokeWidth="1.7" />
                )}
                <path d={demandPath} fill="none" stroke="#d4d4d8" strokeWidth="1.5" />

                {/* violating slots under frozen policy */}
                {surged &&
                  DRIFT.slice(0, upto + 1).map((d, i) =>
                    d.violFrozen === 1 ? (
                      <circle key={i} cx={xs(i)} cy={ysTop(d.demandDrift)} r="2.4" fill="#fb7185" />
                    ) : null,
                  )}
              </svg>

              {/* bottom: rolling violation */}
              <p className="px-2 pt-2 text-[11px] font-medium uppercase tracking-wide text-zinc-500">
                rolling violation rate (96-slot window)
              </p>
              <svg
                viewBox={`0 0 ${W} ${H_BOT}`}
                className="h-36 w-full sm:h-44"
                preserveAspectRatio="none"
                aria-label="Rolling violation rate for frozen versus self-evolving agent"
              >
                <line x1={M.l} x2={W - M.r} y1={ysBot(TRIGGER)} y2={ysBot(TRIGGER)} stroke="#71717a" strokeDasharray="3 4" strokeWidth="1" />
                <text x={M.l + 6} y={ysBot(TRIGGER) - 5} fill="#a1a1aa" fontSize="11">
                  Stage-5 trigger {TRIGGER}%
                </text>

                {surged &&
                  UPDATES.map((u) =>
                    showEvolving && u <= upto ? (
                      <g key={u}>
                        <line x1={xs(u)} x2={xs(u)} y1={M.t} y2={H_BOT - M.b} stroke="#34d399" strokeDasharray="2 4" strokeOpacity="0.8" />
                        <text x={xs(u)} y={H_BOT - M.b - 6} fill="#34d399" fontSize="11" textAnchor="middle">
                          update
                        </text>
                      </g>
                    ) : null,
                  )}

                <path d={rollF} fill="none" stroke="#fb7185" strokeWidth="1.8" />
                {rollE && <path d={rollE} fill="none" stroke="#34d399" strokeWidth="1.8" />}

                {Object.entries(TICK_LABELS).map(([t, label]) => (
                  <text key={label} x={xs(parseInt(t))} y={H_BOT - 6} fill="#52525b" fontSize="10" textAnchor="center">
                    {label}
                  </text>
                ))}
              </svg>

              <div className="flex flex-wrap items-center gap-4 px-2 pt-1 text-xs text-zinc-500">
                <span className="flex items-center gap-1.5">
                  <span className="h-0.5 w-4 rounded bg-zinc-300" /> demand (+surge)
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="h-0.5 w-4 rounded bg-rose-400" /> frozen
                </span>
                <span className="flex items-center gap-1.5">
                  <span className="h-0.5 w-4 rounded bg-emerald-400" /> self-evolving
                </span>
              </div>
            </div>
          </Panel>
        </Reveal>

        {/* KPI cards */}
        <div className="mt-6 grid gap-4 md:grid-cols-3">
          <AnimatePresence>
            {(phase === "frozenDone" || phase === "evolving" || phase === "done") && (
              <motion.div
                key="frozen"
                initial={{ opacity: 0, y: 16 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.5 }}
              >
                <Panel className="h-full border-rose-400/20 p-5">
                  <p className="text-xs font-semibold uppercase tracking-wide text-rose-300">
                    Frozen agent (Stages 1-4)
                  </p>
                  <p className="mt-3 text-3xl font-bold tabular-nums text-zinc-50">
                    <CountUp value={DRIFT_KPI.frozenPost} suffix="%" />
                  </p>
                  <p className="text-xs text-zinc-500">post-surge violation rate</p>
                  <p className="mt-3 text-xs leading-relaxed text-zinc-400">
                    Never re-learns. The surge keeps breaking through at{" "}
                    {DRIFT_KPI.frozenUtil}% utilization, and the rolling rate
                    never drops back under the {TRIGGER}% trigger.
                  </p>
                </Panel>
              </motion.div>
            )}
            {phase === "done" && (
              <>
                <motion.div
                  key="evolving"
                  initial={{ opacity: 0, y: 16 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.5, delay: 0.15 }}
                >
                  <Panel className="h-full border-emerald-400/20 p-5">
                    <p className="text-xs font-semibold uppercase tracking-wide text-emerald-300">
                      Self-evolving UCRA
                    </p>
                    <p className="mt-3 text-3xl font-bold tabular-nums text-zinc-50">
                      <CountUp value={DRIFT_KPI.evolvingPost} suffix="%" />
                    </p>
                    <p className="text-xs text-zinc-500">post-surge violation rate</p>
                    <p className="mt-3 text-xs leading-relaxed text-zinc-400">
                      Drift monitor fires at the {TRIGGER}% line; the quantile
                      LSTM fine-tunes {DRIFT_KPI.updates} times (slots{" "}
                      {UPDATES.join(" · ")}) and violations fall by{" "}
                      {(DRIFT_KPI.frozenPost - DRIFT_KPI.evolvingPost).toFixed(1)}{" "}
                      points.
                    </p>
                  </Panel>
                </motion.div>
                <motion.div
                  key="delta"
                  initial={{ opacity: 0, y: 16 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.5, delay: 0.3 }}
                >
                  <Panel className="h-full p-5">
                    <p className="text-xs font-semibold uppercase tracking-wide text-teal-300">
                      The evolution dividend
                    </p>
                    <p className="mt-3 flex items-center gap-2 text-3xl font-bold tabular-nums text-zinc-50">
                      <TrendingDown className="h-7 w-7 text-emerald-400" />
                      &minus;56%
                    </p>
                    <p className="text-xs text-zinc-500">violations vs frozen</p>
                    <p className="mt-3 text-xs leading-relaxed text-zinc-400">
                      Same model, same data, same surge — the only difference is
                      Stage 5: a replay buffer ({CFG.replaySize} windows) and{" "}
                      {CFG.finetuneEpochs}-epoch fine-tunes whenever the
                      trigger fires.
                    </p>
                  </Panel>
                </motion.div>
              </>
            )}
          </AnimatePresence>
        </div>

        <p className="mt-6 text-center text-xs leading-relaxed text-zinc-600">
          Official aggregates from <code>outputs/drift_demo.json</code>: frozen{" "}
          {DRIFT_KPI.frozenOverall}% overall / {DRIFT_KPI.frozenPost}% post-surge,
          evolving {DRIFT_KPI.evolvingOverall}% / {DRIFT_KPI.evolvingPost}%. Per-slot
          timelines are reconstructed from the logged reservation series and
          calibrated to those exact counts (57 vs 25 violations).
        </p>
      </div>
    </section>
  );
}
