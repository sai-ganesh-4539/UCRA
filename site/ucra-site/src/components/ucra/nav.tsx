"use client";

import { useEffect, useState } from "react";
import { Github, Activity } from "lucide-react";
import { REPO_URL } from "@/lib/ucra";

const LINKS = [
  { href: "#problem", label: "Problem" },
  { href: "#results", label: "Results" },
  { href: "#lab", label: "Kappa Lab" },
  { href: "#drift", label: "Drift Demo" },
  { href: "#cttc", label: "CTTC" },
  { href: "#pipeline", label: "How it works" },
];

export function Nav() {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 24);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <header
      className={`fixed inset-x-0 top-0 z-50 transition-colors duration-300 ${
        scrolled
          ? "border-b border-white/[0.06] bg-[#070b09]/85 backdrop-blur-xl"
          : "bg-transparent"
      }`}
    >
      <nav
        aria-label="Main"
        className="mx-auto flex h-14 max-w-6xl items-center justify-between px-4 sm:px-6"
      >
        <a href="#top" className="flex items-center gap-2 font-semibold text-zinc-50">
          <span className="flex h-7 w-7 items-center justify-center rounded-lg bg-emerald-400/15 ring-1 ring-emerald-400/30">
            <Activity className="h-4 w-4 text-emerald-400" />
          </span>
          UCRA
        </a>
        <div className="hidden items-center gap-1 md:flex">
          {LINKS.map((l) => (
            <a
              key={l.href}
              href={l.href}
              className="rounded-lg px-3 py-2 text-sm text-zinc-400 transition-colors hover:bg-white/[0.06] hover:text-zinc-100"
            >
              {l.label}
            </a>
          ))}
        </div>
        <a
          href={REPO_URL}
          target="_blank"
          rel="noreferrer"
          className="flex items-center gap-2 rounded-lg border border-white/10 bg-white/[0.04] px-3 py-1.5 text-sm text-zinc-200 transition-colors hover:bg-white/[0.1]"
        >
          <Github className="h-4 w-4" />
          <span className="hidden sm:inline">GitHub</span>
        </a>
      </nav>
    </header>
  );
}
