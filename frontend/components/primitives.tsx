"use client";

import { motion } from "framer-motion";
import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import type { HTMLAttributes, ReactNode } from "react";

export function Skeleton({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn(
        "animate-pulse rounded-2xl bg-gradient-to-r from-white/[0.06] via-white/[0.04] to-white/[0.06]",
        className
      )}
      {...props}
    />
  );
}

export function Badge({ children, className, variant = "default", ...props }: HTMLAttributes<HTMLSpanElement> & { variant?: "default" | "success" | "warning" | "danger" | "accent" }) {
  const variants = {
    default: "bg-white/[0.08] text-slate-200 ring-white/10",
    success: "bg-emerald-500/10 text-emerald-300 ring-emerald-300/20",
    warning: "bg-amber-500/10 text-amber-200 ring-amber-200/20",
    danger: "bg-rose-500/10 text-rose-200 ring-rose-200/20",
    accent: "bg-accent-400/10 text-accent-200 ring-accent-300/20",
  };

  return (
    <span
      className={cn("inline-flex items-center rounded-full px-3 py-1 text-xs font-semibold ring-1", variants[variant], className)}
      {...props}
    >
      {children}
    </span>
  );
}

export function StatCard({ label, value, suffix, icon, delay = 0 }: { label: string; value: number; suffix?: string; icon: ReactNode; delay?: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true }}
      transition={{ delay, duration: 0.5, ease: [0.22, 1, 0.36, 1] }}
      className="relative overflow-hidden rounded-2xl border border-slate-200 dark:border-white/10 bg-white/60 dark:bg-white/[0.04] p-5 backdrop-blur-xl"
    >
      <div className="absolute inset-0 bg-gradient-to-br from-white/[0.04] dark:from-white/[0.04] to-transparent" />
      <div className="relative flex items-start justify-between">
        <div>
          <p className="text-xs uppercase tracking-[0.2em] text-slate-500 dark:text-slate-400">{label}</p>
          <div className="mt-2 flex items-baseline gap-1">
            <CountUp target={value} />
            {suffix && <span className="text-lg text-slate-500 dark:text-slate-400">{suffix}</span>}
          </div>
        </div>
        <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-accent-400/10 text-accent-300 ring-1 ring-accent-300/20">
          {icon}
        </div>
      </div>
    </motion.div>
  );
}

function CountUp({ target, duration = 2000 }: { target: number; duration?: number }) {
  const [count, setCount] = useState(0);

  useEffect(() => {
    let start = 0;
    const startTime = performance.now();

    function step(currentTime: number) {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
      const eased = 1 - Math.pow(1 - progress, 3);
      setCount(Math.floor(eased * target));

      if (progress < 1) {
        requestAnimationFrame(step);
      }
    }

    requestAnimationFrame(step);
  }, [target, duration]);

  return <span className="text-3xl font-semibold text-slate-900 dark:text-white">{count.toLocaleString()}</span>;
}
