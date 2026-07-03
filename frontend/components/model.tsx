"use client";

import { motion } from "framer-motion";
import { ChevronRight, Cpu, Sparkles, Upload } from "lucide-react";
import { featureCards, pipelineSteps } from "@/lib/data";
import { GlassCard, SectionHeading } from "@/components/ui";

export function ModelArchitecture() {
  return (
    <section className="space-y-8">
      <SectionHeading
        eyebrow="Model architecture"
        title="ResNet18 or EfficientNet-style inference wrapped in a clean pipeline"
        description="The backend keeps the original preprocessing path, loads the checkpoint when available, and falls back gracefully when the model file is missing."
      />

      <div className="grid gap-4 lg:grid-cols-4">
        {pipelineSteps.map((step, index) => (
          <motion.div key={step.title} initial={{ opacity: 0, y: 14 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: index * 0.08 }}>
            <GlassCard className="h-full transition-all duration-200 hover:border-slate-300 dark:hover:border-white/20 hover:bg-slate-100 dark:hover:bg-white/[0.06] group">
              <div className="flex h-11 w-11 items-center justify-center rounded-2xl bg-slate-100 dark:bg-white/[0.05] text-accent-300 ring-1 ring-slate-200 dark:ring-white/10 transition-all duration-200 group-hover:bg-accent-400/10 group-hover:ring-accent-300/20">
                {index === 0 ? <Upload size={20} /> : index === 1 ? <Sparkles size={20} /> : index === 2 ? <Cpu size={20} /> : <ChevronRight size={20} />}
              </div>
              <div className="mt-5 flex items-center gap-2">
                <span className="text-xs font-semibold text-accent-300">0{index + 1}</span>
                <h3 className="font-heading text-xl text-slate-900 dark:text-white">{step.title}</h3>
              </div>
              <p className="mt-2 text-sm leading-7 text-slate-600 dark:text-slate-300">{step.detail}</p>
            </GlassCard>
          </motion.div>
        ))}
      </div>

      <div className="grid gap-4 md:grid-cols-3">
        {featureCards.map((card) => (
          <GlassCard key={card.title} className="transition-all duration-200 hover:border-slate-300 dark:hover:border-white/20 hover:bg-slate-100 dark:hover:bg-white/[0.06]">
            <h4 className="font-semibold text-slate-900 dark:text-white">{card.title}</h4>
            <p className="mt-2 text-sm leading-7 text-slate-600 dark:text-slate-300">{card.detail}</p>
          </GlassCard>
        ))}
      </div>
    </section>
  );
}
