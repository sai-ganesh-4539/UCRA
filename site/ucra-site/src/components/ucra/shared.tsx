"use client";

import { motion, useInView } from "framer-motion";
import { useEffect, useRef, useState, type ReactNode } from "react";

/* Scroll-reveal wrapper */
export function Reveal({
  children,
  delay = 0,
  className,
}: {
  children: ReactNode;
  delay?: number;
  className?: string;
}) {
  return (
    <motion.div
      className={className}
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-80px" }}
      transition={{ duration: 0.6, delay, ease: [0.21, 0.47, 0.32, 0.98] }}
    >
      {children}
    </motion.div>
  );
}

export function SectionHeading({
  kicker,
  title,
  sub,
}: {
  kicker: string;
  title: ReactNode;
  sub?: ReactNode;
}) {
  return (
    <Reveal className="mx-auto max-w-3xl text-center">
      <p className="text-xs font-semibold uppercase tracking-[0.2em] text-emerald-400">
        {kicker}
      </p>
      <h2 className="mt-3 text-3xl font-bold tracking-tight text-zinc-50 sm:text-4xl">
        {title}
      </h2>
      {sub ? (
        <p className="mt-4 text-base leading-relaxed text-zinc-400">{sub}</p>
      ) : null}
    </Reveal>
  );
}

/* Animated number that counts up when scrolled into view */
export function CountUp({
  value,
  decimals = 1,
  suffix = "",
  duration = 1400,
  className,
}: {
  value: number;
  decimals?: number;
  suffix?: string;
  duration?: number;
  className?: string;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true, margin: "-40px" });
  const [display, setDisplay] = useState("0");

  useEffect(() => {
    if (!inView) return;
    let raf = 0;
    const start = performance.now();
    const tick = (now: number) => {
      const p = Math.min(1, (now - start) / duration);
      const eased = 1 - Math.pow(1 - p, 3);
      setDisplay((value * eased).toFixed(decimals));
      if (p < 1) raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [inView, value, decimals, duration]);

  return (
    <span ref={ref} className={className}>
      {display}
      {suffix}
    </span>
  );
}

export function Panel({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`rounded-2xl border border-white/[0.08] bg-white/[0.03] ${className}`}
    >
      {children}
    </div>
  );
}
