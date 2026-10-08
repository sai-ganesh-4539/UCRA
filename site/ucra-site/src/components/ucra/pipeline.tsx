"use client";

import {
  Waves,
  Sigma,
  Split,
  RefreshCcw,
  BrainCircuit,
  Database,
} from "lucide-react";
import { Reveal, SectionHeading, Panel } from "./shared";
import { CFG, DATASETS, TRAIN } from "@/lib/ucra";

const STAGES = [
  {
    icon: Waves,
    n: "01",
    title: "Uncertainty estimation",
    body: "A 2-layer quantile LSTM (64 hidden units, 24 h input window) predicts 7 demand quantiles — q05 to q99 — trained with pinball loss.",
    chips: [`${CFG.params.toLocaleString()} params`, `pinball ${TRAIN.pinball}`, `q05–q99`],
  },
  {
    icon: Sigma,
    n: "02",
    title: "Uncertainty → reservation (φ)",
    body: "The φ-transform converts spread into headroom: R = q₀.₉ + κ · spread · (1 + 0.3·ρₜ), clipped to keep ≥5% headroom and ≤95% of cell capacity.",
    chips: [`base τ = 0.9`, `κ = 0.5`, `min headroom 5%`],
  },
  {
    icon: Split,
    n: "03",
    title: "Risk-adaptive allocation",
    body: "Reserved capacity is split across slices and sectors weighted by demand pressure and each slice's risk exponent, holding back a 10% best-effort pool.",
    chips: [`pressure w = 1.0`, `risk exp 0.5`, `10% best-effort`],
  },
  {
    icon: RefreshCcw,
    n: "04",
    title: "Update & release",
    body: "Reservations move smoothly (EWMA α=0.3) and release headroom only when it stays >10% unused — no thrashing between slots.",
    chips: [`EWMA α = 0.3`, `hysteresis 10%`, `drift window 96`],
  },
  {
    icon: BrainCircuit,
    n: "05",
    title: "Self-evolving learning",
    body: "A monitor watches the rolling violation rate. Cross 15% (or demand z>3) and the model fine-tunes on a 512-window replay buffer — drift handled in-loop.",
    chips: [`trigger 15%`, `replay 512`, `3-epoch tune`],
  },
];

export function Pipeline() {
  return (
    <section id="pipeline" className="relative py-20 sm:py-28">
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        <SectionHeading
          kicker="How it works"
          title="Five stages, one closed loop"
          sub="Forecast the distribution, convert uncertainty into headroom, allocate it, move it smoothly, and let the agent retrain itself when the world changes."
        />

        <div className="mt-12 grid gap-4 md:grid-cols-2 lg:grid-cols-5">
          {STAGES.map((s, i) => (
            <Reveal key={s.n} delay={i * 0.08} className="h-full">
              <Panel className="relative flex h-full flex-col p-5">
                <span className="absolute right-4 top-4 font-mono text-xs text-zinc-600">
                  {s.n}
                </span>
                <span className="inline-flex h-10 w-10 items-center justify-center rounded-xl bg-emerald-400/10 ring-1 ring-emerald-400/25">
                  <s.icon className="h-5 w-5 text-emerald-400" />
                </span>
                <h3 className="mt-4 text-base font-semibold leading-snug text-zinc-100">
                  {s.title}
                </h3>
                <p className="mt-2 flex-1 text-[13px] leading-relaxed text-zinc-400">
                  {s.body}
                </p>
                <div className="mt-4 flex flex-wrap gap-1.5">
                  {s.chips.map((c) => (
                    <span
                      key={c}
                      className="rounded-md border border-white/[0.08] bg-white/[0.04] px-2 py-0.5 font-mono text-[10px] text-zinc-400"
                    >
                      {c}
                    </span>
                  ))}
                </div>
              </Panel>
            </Reveal>
          ))}
        </div>

        {/* dataset strip */}
        <Reveal delay={0.15}>
          <Panel className="mt-6 flex flex-col items-start gap-4 p-5 sm:flex-row sm:items-center sm:justify-between sm:p-6">
            <div className="flex items-start gap-3">
              <span className="inline-flex h-10 w-10 shrink-0 items-center justify-center rounded-xl bg-teal-400/10 ring-1 ring-teal-400/25">
                <Database className="h-5 w-5 text-teal-300" />
              </span>
              <div>
                <p className="text-sm font-semibold text-zinc-100">
                  Trained and evaluated on public traces — nothing synthetic in
                  the headline numbers
                </p>
                <p className="mt-1 text-xs leading-relaxed text-zinc-500">
                  Live 5G/4G/2G RAN PM counters ({DATASETS.ran.rows.toLocaleString()}{" "}
                  rows, {DATASETS.ran.sectors} sectors, {DATASETS.ran.slots} slots,{" "}
                  {DATASETS.ran.doi}) for Stages 1/4/5; CTTC B5G slicing
                  simulations ({DATASETS.cttc.snapshots} snapshots,{" "}
                  {DATASETS.cttc.doi}) for slice-level reservation; Liverpool 5G
                  HDD as optional failsafe.
                </p>
              </div>
            </div>
            <div className="grid shrink-0 grid-cols-3 gap-4 text-center sm:pl-6">
              {[
                { v: DATASETS.ran.rows.toLocaleString(), l: "RAN rows" },
                { v: `${DATASETS.ran.sectors}`, l: "sectors" },
                { v: `${DATASETS.cttc.snapshots}`, l: "CTTC snaps" },
              ].map((s) => (
                <div key={s.l}>
                  <p className="text-lg font-bold tabular-nums text-zinc-50">
                    {s.v}
                  </p>
                  <p className="text-[10px] uppercase tracking-wide text-zinc-500">
                    {s.l}
                  </p>
                </div>
              ))}
            </div>
          </Panel>
        </Reveal>
      </div>
    </section>
  );
}
