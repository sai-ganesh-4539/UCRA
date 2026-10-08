"use client";

import { BatteryLow, Gauge, Eye, ArrowRight } from "lucide-react";
import { Reveal, SectionHeading, Panel } from "./shared";
import { RAN } from "@/lib/ucra";

const CARDS = [
  {
    icon: BatteryLow,
    tone: "text-amber-300",
    ring: "ring-amber-400/25 bg-amber-400/10",
    big: `${RAN.rows[1].waste}%`,
    bigLabel: "of capacity wasted",
    title: "Play it safe, burn money",
    body: `Provisioning for the historic peak (static peak) never violates the SLA — but it idles ${RAN.rows[1].waste}% of the cell's capacity on average. You pay for radio capacity that almost never carries a single bit.`,
  },
  {
    icon: Gauge,
    tone: "text-rose-300",
    ring: "ring-rose-400/25 bg-rose-400/10",
    big: `${RAN.rows[2].viol}%`,
    bigLabel: "of slots violate the SLA",
    title: "Play it lean, break the promise",
    body: `Reserving the mean forecast looks efficient — only ${RAN.rows[2].waste}% waste — but half of all slots (${RAN.rows[2].viol}%) burst past the reservation. Every violation is a dropped URLLC packet or a stalled video session.`,
  },
  {
    icon: Eye,
    tone: "text-zinc-300",
    ring: "ring-zinc-400/25 bg-zinc-400/10",
    big: `${RAN.rows[3].viol}%`,
    bigLabel: "violations - with foresight",
    title: "Even an oracle struggles",
    body: `Give a policy perfect knowledge of the future demand distribution and it still misses ${RAN.rows[3].viol}% of slots. Uncertainty is not noise to average away — it must be reserved for, explicitly.`,
  },
];

export function Problem() {
  return (
    <section id="problem" className="relative py-20 sm:py-28">
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        <SectionHeading
          kicker="The problem"
          title="Every fixed policy fails one way or another"
          sub="Network demand is bursty, diurnal, and drifting. Pick a reservation strategy and you inherit its failure mode — the operator's dilemma is choosing which one you can afford."
        />

        <div className="mt-12 grid gap-4 md:grid-cols-3">
          {CARDS.map((c, i) => (
            <Reveal key={c.title} delay={i * 0.1}>
              <Panel className="flex h-full flex-col p-6">
                <span
                  className={`inline-flex h-10 w-10 items-center justify-center rounded-xl ring-1 ${c.ring}`}
                >
                  <c.icon className={`h-5 w-5 ${c.tone}`} />
                </span>
                <p className="mt-5 text-4xl font-bold tabular-nums text-zinc-50">
                  {c.big}
                </p>
                <p className="mt-1 text-xs font-medium uppercase tracking-wide text-zinc-500">
                  {c.bigLabel}
                </p>
                <h3 className="mt-4 text-lg font-semibold text-zinc-100">
                  {c.title}
                </h3>
                <p className="mt-2 text-sm leading-relaxed text-zinc-400">
                  {c.body}
                </p>
              </Panel>
            </Reveal>
          ))}
        </div>

        <Reveal delay={0.2}>
          <div className="mx-auto mt-10 flex max-w-3xl items-center gap-4 rounded-2xl border border-emerald-400/20 bg-emerald-400/[0.06] p-5">
            <ArrowRight className="h-5 w-5 shrink-0 text-emerald-400" />
            <p className="text-sm leading-relaxed text-zinc-300">
              <span className="font-semibold text-emerald-300">UCRA&apos;s bet:</span>{" "}
              stop picking one operating point. Forecast the whole demand
              <em> distribution</em>, convert measured uncertainty into headroom
              with a tunable risk dial (&kappa;), and keep fine-tuning the model
              as traffic drifts —{" "}
              <span className="text-zinc-100">
                {RAN.rows[0].viol}% violations at {RAN.rows[0].util}% utilization,
                wasting only {RAN.rows[0].waste}%.
              </span>
            </p>
          </div>
        </Reveal>
      </div>
    </section>
  );
}
