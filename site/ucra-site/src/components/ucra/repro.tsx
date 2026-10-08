"use client";

import { useState } from "react";
import { Check, Copy, Github, TerminalSquare } from "lucide-react";
import { Reveal, SectionHeading, Panel } from "./shared";
import { DATASETS, REPO_URL } from "@/lib/ucra";

const COMMANDS = [
  "git clone https://github.com/your-username/UCRA.git && cd UCRA",
  "pip install -r requirements.txt",
  "python scripts/download_data.py          # Zenodo: RAN + CTTC traces",
  "python scripts/audit_data.py             # 302,052 rows / 75 sectors sanity",
  "python scripts/train.py                  # quantile LSTM -> outputs/model.pt",
  "python scripts/run_ucra.py               # head-to-head vs baselines",
  "python scripts/demo_evolution.py         # +35% surge drift demo",
  "python scripts/run_cttc_eval.py          # per-slice kappa sweep",
  "python scripts/make_figures.py           # every figure on this page",
];

const DATASET_CARDS = [
  {
    name: "Live 5G/4G/2G RAN PM counters",
    meta: `${DATASETS.ran.rows.toLocaleString()} rows · ${DATASETS.ran.sectors} sectors · ${DATASETS.ran.slots} slots · ${DATASETS.ran.doi}`,
    role: "Stage 1/4/5 — demand uncertainty, update rule, self-evolution",
  },
  {
    name: "CTTC B5G network slicing simulations",
    meta: `${DATASETS.cttc.snapshots} snapshots · ${DATASETS.cttc.doi}`,
    role: "Stage 2/3 — slice-level reservation & violations",
  },
  {
    name: "Liverpool 5G urban deployment (HDD)",
    meta: "Nature Sci Data 2025 · optional failsafe",
    role: "Backup real-world demand trace",
  },
];

export function Repro() {
  const [copied, setCopied] = useState(false);

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(COMMANDS.join("\n"));
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      /* clipboard unavailable */
    }
  };

  return (
    <section id="repo" className="relative py-20 sm:py-28">
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        <SectionHeading
          kicker="Reproduce it"
          title="Every number here comes from the repo"
          sub="No cherry-picking: the pipeline is scripted end-to-end, from raw Zenodo archives to the figures and JSON that power this page."
        />

        <div className="mt-12 grid gap-6 lg:grid-cols-5">
          <Reveal className="lg:col-span-3">
            <Panel className="overflow-hidden">
              <div className="flex items-center justify-between border-b border-white/[0.06] px-4 py-3">
                <span className="flex items-center gap-2 text-sm font-medium text-zinc-300">
                  <TerminalSquare className="h-4 w-4 text-emerald-400" />
                  quickstart
                </span>
                <button
                  onClick={copy}
                  className="inline-flex items-center gap-1.5 rounded-lg border border-white/10 px-2.5 py-1.5 text-xs text-zinc-400 transition-colors hover:bg-white/[0.08] hover:text-zinc-200"
                  aria-label="Copy commands"
                >
                  {copied ? (
                    <>
                      <Check className="h-3.5 w-3.5 text-emerald-400" /> copied
                    </>
                  ) : (
                    <>
                      <Copy className="h-3.5 w-3.5" /> copy
                    </>
                  )}
                </button>
              </div>
              <div className="max-h-96 overflow-y-auto p-4 font-mono text-xs leading-6 scrollbar-thin">
                {COMMANDS.map((c) => (
                  <p key={c} className="whitespace-pre">
                    <span className="select-none text-emerald-500">$ </span>
                    <span className="text-zinc-300">{c}</span>
                  </p>
                ))}
              </div>
            </Panel>
          </Reveal>

          <Reveal className="lg:col-span-2" delay={0.1}>
            <div className="flex h-full flex-col gap-4">
              <Panel className="p-5">
                <h3 className="text-sm font-semibold uppercase tracking-wide text-zinc-400">
                  Datasets
                </h3>
                <div className="mt-4 space-y-4">
                  {DATASET_CARDS.map((d) => (
                    <div key={d.name} className="border-l-2 border-emerald-400/40 pl-3">
                      <p className="text-sm font-medium text-zinc-200">{d.name}</p>
                      <p className="mt-0.5 font-mono text-[10px] text-zinc-500">
                        {d.meta}
                      </p>
                      <p className="mt-1 text-xs text-zinc-400">{d.role}</p>
                    </div>
                  ))}
                </div>
              </Panel>
              <a
                href={REPO_URL}
                target="_blank"
                rel="noreferrer"
                className="group flex items-center justify-between rounded-2xl border border-emerald-400/25 bg-emerald-400/[0.07] p-5 transition-colors hover:bg-emerald-400/[0.12]"
              >
                <div>
                  <p className="text-sm font-semibold text-emerald-300">
                    Full source, docs & deep-dives
                  </p>
                  <p className="mt-1 text-xs text-zinc-400">
                    15 docs guides · 4 DEEP_*.md · tests · MIT license
                  </p>
                </div>
                <Github className="h-6 w-6 text-emerald-300 transition-transform group-hover:scale-110" />
              </a>
            </div>
          </Reveal>
        </div>
      </div>
    </section>
  );
}

export function Footer() {
  return (
    <footer className="mt-auto border-t border-white/[0.06] bg-black/30">
      <div className="mx-auto flex max-w-6xl flex-col items-center justify-between gap-3 px-4 py-8 text-center sm:flex-row sm:px-6 sm:text-left">
        <p className="text-xs text-zinc-500">
          <span className="font-semibold text-zinc-300">UCRA</span> —
          Uncertainty-to-Reservation Conversion for Self-Evolving Risk-Adaptive
          Resource Allocation. Research demo; all metrics from{" "}
          <code className="text-zinc-400">outputs/*.json</code> of the public
          pipeline.
        </p>
        <a
          href="#top"
          className="text-xs text-zinc-500 transition-colors hover:text-emerald-300"
        >
          back to top ↑
        </a>
      </div>
    </footer>
  );
}
