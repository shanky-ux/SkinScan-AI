"use client";

import { motion } from "framer-motion";
import { Activity, AlertTriangle, CheckCircle2, Info } from "lucide-react";
import { GlassCard, Badge, SectionHeading } from "@/components/ui";
import type { ClassItem, SeverityLevel } from "@/lib/types";
import { useMemo, useState } from "react";
import { filterOptions } from "@/lib/data";

export function DiseaseLibrary({ classes }: { classes: ClassItem[] }) {
  const [filter, setFilter] = useState<SeverityLevel | "all">("all");

  const filtered = useMemo(() => {
    return classes.filter((item) => filter === "all" || item.disease_info.severity_level === filter);
  }, [classes, filter]);

  return (
    <section className="space-y-8">
      <SectionHeading
        eyebrow="Disease reference"
        title="A compact reference library for the 9 classes"
        description="Each card keeps the model label, condition summary, severity signal, and the recommended next step together in one readable place."
      />

      <div className="flex flex-wrap gap-3">
        {filterOptions.map((option) => (
          <button
            key={option.value}
            type="button"
            onClick={() => setFilter(option.value)}
            className={`rounded-full px-4 py-2 text-sm font-semibold transition-all duration-200 ${
              filter === option.value ? "bg-accent-400 text-slate-950 shadow-[0_8px_24px_rgba(24,189,224,0.25)]" : "bg-slate-100 dark:bg-white/[0.05] text-slate-900 dark:text-white hover:bg-slate-200 dark:hover:bg-white/[0.1]"
            }`}
          >
            {option.label}
          </button>
        ))}
      </div>

      <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
        {filtered.map((item, index) => (
          <motion.div key={item.class_name} initial={{ opacity: 0, y: 18 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: index * 0.04 }}>
            <GlassCard className="h-full transition-all duration-200 hover:border-slate-300 dark:hover:border-white/20 hover:bg-slate-100 dark:hover:bg-white/[0.06]">
              <div className="flex items-start justify-between gap-4">
                <div className="flex items-center gap-3">
                  <div className={`flex h-10 w-10 shrink-0 items-center justify-center rounded-xl ${
                    item.disease_info.severity_level === "high" ? "bg-rose-500/10 text-rose-300 ring-1 ring-rose-300/20" :
                    item.disease_info.severity_level === "moderate" ? "bg-amber-500/10 text-amber-200 ring-1 ring-amber-200/20" :
                    item.disease_info.severity_level === "low" ? "bg-emerald-500/10 text-emerald-200 ring-1 ring-emerald-200/20" :
                    "bg-slate-100 dark:bg-slate-500/10 text-slate-600 dark:text-slate-200 ring-1 ring-slate-200 dark:ring-slate-200/20"
                  }`}>
                    {item.disease_info.severity_level === "high" ? <AlertTriangle size={18} /> :
                     item.disease_info.severity_level === "moderate" ? <Info size={18} /> :
                     item.disease_info.severity_level === "low" ? <CheckCircle2 size={18} /> :
                     <Activity size={18} />}
                  </div>
                  <div>
                    <h3 className="font-heading text-lg text-slate-900 dark:text-white">{item.class_name}</h3>
                    <Badge variant={
                      item.disease_info.severity_level === "high" ? "danger" :
                      item.disease_info.severity_level === "moderate" ? "warning" :
                      item.disease_info.severity_level === "low" ? "success" : "default"
                    }>
                      {item.disease_info.severity_level}
                    </Badge>
                  </div>
                </div>
              </div>
              <div className="mt-4 space-y-3 text-sm leading-7 text-slate-600 dark:text-slate-300">
                <p>{item.disease_info.description}</p>
                <div className="rounded-xl border border-slate-200 dark:border-white/10 bg-slate-100 dark:bg-white/[0.03] p-3">
                  <p className="text-xs font-semibold uppercase tracking-wider text-slate-500 dark:text-slate-400 mb-1">Recommendation</p>
                  <p className="text-sm text-slate-700 dark:text-slate-200">{item.disease_info.recommendation}</p>
                </div>
              </div>
            </GlassCard>
          </motion.div>
        ))}
      </div>
    </section>
  );
}
