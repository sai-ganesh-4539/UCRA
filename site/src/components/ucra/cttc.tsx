"use client";

import { useState } from "react";
import { motion } from "framer-motion";
import { Globe2 } from "lucide-react";
import { Reveal, SectionHeading, Panel } from "./shared";
import { CTTC, type SliceResult } from "@/lib/ucra";

const SLICE_COLOR: Record<string, string> = {
  URLLC: "#34d399",
  eMBB: "#fbbf24",
  mMTC: "#2dd4bf",
};
const SLICE_BLURB: Record<string, string> = {
  URLLC: "577 test snapshots — mission-critical traffic where every violation is a fault",
  eMBB: "337 test snapshots — the operator misses almost half of all eMBB windows today",
  mMTC: "630 test snapshots — massive IoT: huge device counts, modest per-device demand",
};

type Metric = "viol" | "over";

function GroupedBars({ slice, metric }: { slice: SliceResult; metric: Metric }) {
  const bars = [
    { name: "Operator", v: slice.operator[metric], color: "#a1a1aa" },
    { name: "Static q90", v: slice.staticQ90[metric], color: "#fbbf24" },
    { name: "UCRA (κ=0.5)", v: slice.ucra[metric], color: "#34d399" },
  ];
  const max = Math.max(...bars.map((b) => b.v), 1);

  return (
    <div className="space-y-5">
      {bars.map((b, i) => (
        <div key={b.name}>
          <div className="mb-1.5 flex items-baseline justify-between text-sm">
            <span className="font-medium text-zinc-200">{b.name}</span>
            <span className="tabular-nums text-zinc-400">{b.v.toFixed(1)}%</span>
          </div>
          <div className="h-8 overflow-hidden rounded-lg bg-white/[0.05]">
            <motion.div
              className="flex h-full items-center justify-end rounded-lg pr-2"
              style={{ backgroundColor: `${b.color}cc` }}
              initial={{ width: 0 }}
              animate={{ width: `${Math.max(4, (b.v / max) * 100)}%` }}
              transition={{ duration: 0.7, delay: i * 0.08, ease: "easeOut" }}
            >
              {b.v / max > 0.22 && (
                <span className="text-[11px] font-semibold text-zinc-900/80">
                  {b.name === "Operator" ? "today" : ""}
                </span>
              )}
            </motion.div>
          </div>
        </div>
      ))}
    </div>
  );
}

export function Cttc() {
  const [sliceId, setSliceId] = useState("URLLC");
  const [metric, setMetric] = useState<Metric>("viol");
  const slice = CTTC.slices.find((s) => s.id === sliceId)!;

  const reduction =
    metric === "viol"
      ? Math.round((1 - slice.ucra.viol / slice.operator.viol) * 100)
      : null;

  return (
    <section id="cttc" className="relative py-20 sm:py-28">
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        <SectionHeading
          kicker="Public-data validation"
          title="Same story on the CTTC 5G slicing dataset"
          sub={`250 network-slicing simulation snapshots (Zenodo 10610616), evaluated per slice class against the operator's own reservation and a static 90th-percentile policy.`}
        />

        <Reveal className="mt-12">
          <Panel className="p-5 sm:p-7">
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
                </button>
              ))}
              <div className="ml-auto flex rounded-xl border border-white/10 p-1">
                {(
                  [
                    { k: "viol" as Metric, label: "SLA violations" },
                    { k: "over" as Metric, label: "Over-provisioning" },
                  ]
                ).map((m) => (
                  <button
                    key={m.k}
                    onClick={() => setMetric(m.k)}
                    className={`rounded-lg px-3 py-1.5 text-xs font-medium transition-colors ${
                      metric === m.k
                        ? "bg-white/[0.12] text-zinc-100"
                        : "text-zinc-500 hover:text-zinc-300"
                    }`}
                  >
                    {m.label}
                  </button>
                ))}
              </div>
            </div>

            <p className="mt-3 text-xs text-zinc-500">{SLICE_BLURB[slice.id]}</p>

            <div className="mt-6 grid gap-8 lg:grid-cols-2">
              <div>
                <GroupedBars slice={slice} metric={metric} />
                <p className="mt-5 text-xs leading-relaxed text-zinc-500">
                  {metric === "viol" ? (
                    <>
                      UCRA cuts violations by{" "}
                      <span className="font-semibold text-emerald-300">
                        {reduction}%
                      </span>{" "}
                      versus the operator&apos;s current reservation on{" "}
                      {slice.id} (n={slice.n} held-out snapshots).
                    </>
                  ) : (
                    <>
                      On these thin public slices UCRA buys its{" "}
                      {slice.ucra.viol.toFixed(1)}% violation rate with spare
                      headroom — that headroom is exactly what the{" "}
                      &kappa; dial above trades away when a slice can afford
                      more risk.
                    </>
                  )}
                </p>
              </div>

              {/* mini table */}
              <div className="overflow-hidden rounded-xl border border-white/[0.06]">
                <table className="w-full text-sm">
                  <thead>
                    <tr className="border-b border-white/[0.08] bg-white/[0.03] text-left text-xs uppercase tracking-wide text-zinc-500">
                      <th className="px-4 py-3 font-medium">Policy</th>
                      <th className="px-4 py-3 text-right font-medium">Violations</th>
                      <th className="px-4 py-3 text-right font-medium">Over-prov.</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-white/[0.05]">
                    <tr>
                      <td className="px-4 py-3 text-zinc-300">Operator</td>
                      <td className="px-4 py-3 text-right tabular-nums text-zinc-400">
                        {slice.operator.viol.toFixed(1)}%
                      </td>
                      <td className="px-4 py-3 text-right tabular-nums text-zinc-400">
                        {slice.operator.over.toFixed(1)}%
                      </td>
                    </tr>
                    <tr>
                      <td className="px-4 py-3 text-zinc-300">Static q90</td>
                      <td className="px-4 py-3 text-right tabular-nums text-zinc-400">
                        {slice.staticQ90.viol.toFixed(1)}%
                      </td>
                      <td className="px-4 py-3 text-right tabular-nums text-zinc-400">
                        {slice.staticQ90.over.toFixed(1)}%
                      </td>
                    </tr>
                    <tr className="bg-emerald-400/[0.05]">
                      <td className="px-4 py-3 font-medium text-emerald-300">
                        UCRA (&kappa;=0.5)
                      </td>
                      <td className="px-4 py-3 text-right font-semibold tabular-nums text-emerald-300">
                        {slice.ucra.viol.toFixed(1)}%
                      </td>
                      <td className="px-4 py-3 text-right font-semibold tabular-nums text-emerald-300">
                        {slice.ucra.over.toFixed(1)}%
                      </td>
                    </tr>
                  </tbody>
                </table>
                <div className="flex items-center gap-2 border-t border-white/[0.06] px-4 py-3 text-xs text-zinc-500">
                  <Globe2 className="h-3.5 w-3.5" />
                  Across all three slices UCRA lands at{" "}
                  {CTTC.slices.map((s) => `${s.ucra.viol.toFixed(1)}%`).join(" / ")}{" "}
                  — an order of magnitude under today&apos;s operator policy.
                </div>
              </div>
            </div>
          </Panel>
        </Reveal>
      </div>
    </section>
  );
}
