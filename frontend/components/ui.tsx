"use client";

import { MoonStar, SunMedium } from "lucide-react";
import { motion, type HTMLMotionProps } from "framer-motion";
import { cn } from "@/lib/utils";
import type { HTMLAttributes, ReactNode } from "react";
import { useTheme } from "@/components/theme-provider";

export function GlassCard({ className, children, ...props }: HTMLAttributes<HTMLDivElement> & { children: ReactNode }) {
  return (
    <div
      className={cn(
        "rounded-3xl border border-slate-200 dark:border-white/10 bg-white/60 dark:bg-white/[0.04] p-5 shadow-glow backdrop-blur-xl",
        className,
      )}
      {...props}
    >
      {children}
    </div>
  );
}

export function SectionHeading({ eyebrow, title, description }: { eyebrow: string; title: string; description: string }) {
  return (
    <div className="max-w-3xl space-y-3">
      <p className="text-xs font-semibold uppercase tracking-[0.32em] text-accent-300/80 dark:text-accent-300/80">{eyebrow}</p>
      <h2 className="font-heading text-3xl tracking-tight text-slate-900 dark:text-white md:text-5xl">{title}</h2>
      <p className="max-w-2xl text-sm leading-7 text-slate-700 dark:text-slate-300 md:text-base">{description}</p>
    </div>
  );
}

export function PrimaryButton({ children, className, ...props }: HTMLMotionProps<"button">) {
  return (
    <motion.button
      whileHover={{ y: -2, scale: 1.01 }}
      whileTap={{ scale: 0.98 }}
      className={cn(
        "inline-flex items-center justify-center rounded-full bg-gradient-to-r from-accent-500 to-cyan-400 px-5 py-3 text-sm font-semibold text-slate-950 shadow-[0_12px_40px_rgba(24,189,224,0.28)] transition focus:outline-none focus:ring-2 focus:ring-accent-300/70 focus:ring-offset-2 focus:ring-offset-slate-50 dark:focus:ring-offset-slate-950",
        className,
      )}
      {...props}
    >
      {children}
    </motion.button>
  );
}

export function GhostButton({ children, className, ...props }: HTMLMotionProps<"button">) {
  return (
    <motion.button
      whileHover={{ y: -2 }}
      whileTap={{ scale: 0.98 }}
      className={cn(
        "inline-flex items-center justify-center rounded-full border border-slate-200 dark:border-white/12 bg-slate-100 dark:bg-white/5 px-5 py-3 text-sm font-semibold text-slate-900 dark:text-white transition hover:border-slate-300 dark:hover:border-white/20 hover:bg-slate-200 dark:hover:bg-white/[0.08] focus:outline-none focus:ring-2 focus:ring-accent-300/70 focus:ring-offset-2 focus:ring-offset-slate-50 dark:focus:ring-offset-slate-950",
        className,
      )}
      {...props}
    >
      {children}
    </motion.button>
  );
}

export function ThemeToggle() {
  const { theme, toggle } = useTheme();

  return (
    <button
      type="button"
      onClick={toggle}
      className="inline-flex h-11 w-11 items-center justify-center rounded-full border border-slate-200 dark:border-white/10 bg-white/80 dark:bg-white/[0.05] text-slate-900 dark:text-white shadow-glow transition hover:bg-slate-200 dark:hover:bg-white/[0.12]"
      aria-label="Toggle theme"
    >
      {theme === "dark" ? <SunMedium size={18} /> : <MoonStar size={18} />}
    </button>
  );
}

export function RiskPill({ value }: { value: string }) {
  const styles: Record<string, string> = {
    low: "bg-emerald-400/10 text-emerald-300 ring-emerald-300/20",
    moderate: "bg-amber-400/10 text-amber-200 ring-amber-200/20",
    high: "bg-rose-400/10 text-rose-200 ring-rose-200/20",
    unknown: "bg-slate-400/10 text-slate-200 ring-slate-200/20",
  };

  return <span className={cn("inline-flex rounded-full px-3 py-1 text-xs font-semibold ring-1", styles[value] ?? styles.unknown)}>{value}</span>;
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