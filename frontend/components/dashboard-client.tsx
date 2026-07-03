"use client";

import { motion } from "framer-motion";
import { useState } from "react";
import { Brain, Cpu, ShieldCheck, Zap } from "lucide-react";
import { GlassCard } from "@/components/ui";
import { Navbar } from "@/components/navbar";
import { StatCard } from "@/components/primitives";
import { ResultsPanel } from "@/components/results";
import { UploadAnalyze } from "@/components/upload";
import type { PredictResponse } from "@/lib/types";

export function DashboardClient() {
  const [result, setResult] = useState<PredictResponse | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);

  return (
    <div className="relative min-h-screen bg-white dark:bg-slate-950 text-slate-900 dark:text-white">
      <Navbar />

      <main className="relative pt-24 pb-16">
        <div className="mx-auto flex w-full max-w-7xl flex-col gap-10 px-4 sm:px-6 lg:px-8">
          {/* Hero Section */}
          <section className="relative overflow-hidden rounded-[2rem] border border-white/10 bg-gradient-to-br from-white/[0.06] via-white/[0.02] to-transparent p-8 sm:p-12 lg:p-16">
            <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top_right,rgba(24,189,224,0.12),transparent_50%),radial-gradient(ellipse_at_bottom_left,rgba(99,102,241,0.08),transparent_40%)]" />

            <div className="relative grid gap-10 lg:grid-cols-[1.2fr,0.8fr] lg:items-center">
              <motion.div
                initial={{ opacity: 0, y: 24 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
                className="space-y-6"
              >
                <div className="inline-flex items-center gap-2 rounded-full border border-accent-300/20 bg-accent-400/10 px-4 py-1.5 text-xs font-semibold uppercase tracking-[0.2em] text-accent-200">
                  <span className="relative flex h-2 w-2">
                    <span className="absolute inset-0 animate-ping rounded-full bg-accent-400 opacity-75" />
                    <span className="relative rounded-full bg-accent-400 h-2 w-2" />
                  </span>
                  AI-Powered Analysis
                </div>

                <h1 className="font-heading text-4xl font-bold tracking-tight text-slate-900 dark:text-white sm:text-5xl lg:text-6xl">
                  Intelligent skin
                  <br />
                  <span className="bg-gradient-to-r from-accent-300 via-cyan-300 to-accent-400 bg-clip-text text-transparent">
                    disease detection
                  </span>
                </h1>

                <p className="max-w-xl text-base leading-7 text-slate-600 dark:text-slate-300 sm:text-lg">
                  Advanced deep learning meets medical imaging. Upload a photo or use your webcam for instant,
                  educational AI analysis across 9 skin conditions.
                </p>

                <div className="flex flex-wrap gap-3">
                  <a
                    href="#analyze"
                    className="inline-flex items-center justify-center rounded-full bg-gradient-to-r from-accent-500 to-cyan-400 px-6 py-3 text-sm font-semibold text-slate-950 shadow-[0_12px_40px_rgba(24,189,224,0.3)] transition-all duration-200 hover:shadow-[0_16px_48px_rgba(24,189,224,0.4)] hover:scale-[1.02] active:scale-[0.98]"
                  >
                    Start Analysis
                  </a>
                  <a
                    href="/conditions"
                    className="inline-flex items-center justify-center rounded-full border border-slate-200 dark:border-white/12 bg-slate-100 dark:bg-white/[0.06] px-6 py-3 text-sm font-semibold text-slate-900 dark:text-white transition-all duration-200 hover:border-slate-300 dark:hover:border-white/20 hover:bg-slate-200 dark:hover:bg-white/[0.1] hover:scale-[1.02] active:scale-[0.98]"
                  >
                    View Library
                  </a>
                </div>
              </motion.div>

              <motion.div
                initial={{ opacity: 0, scale: 0.95 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ duration: 0.7, delay: 0.15, ease: [0.22, 1, 0.36, 1] }}
                className="hidden lg:block"
              >
                <div className="relative">
                  <div className="absolute -inset-4 rounded-[2rem] bg-gradient-to-r from-accent-400/20 to-cyan-400/20 blur-2xl" />
                  <GlassCard className="relative space-y-5 p-6">
                    <div className="flex items-center gap-3">
                      <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-accent-500 to-cyan-400 text-slate-950">
                        <Brain className="h-5 w-5" />
                      </div>
                      <div>
                          <p className="text-sm font-semibold text-slate-900 dark:text-white">Neural Engine</p>
                          <p className="text-xs text-slate-500 dark:text-slate-400">ResNet18 / EfficientNet</p>
                      </div>
                    </div>

                    <div className="space-y-3">
                      {[
                        { label: "Preprocessing", value: 94, color: "from-accent-400 to-cyan-400" },
                        { label: "Inference", value: 99, color: "from-emerald-400 to-cyan-400" },
                        { label: "Confidence", value: 96, color: "from-violet-400 to-accent-400" },
                      ].map((item) => (
                        <div key={item.label} className="space-y-1.5">
                          <div className="flex items-center justify-between text-xs">
                              <span className="text-slate-500 dark:text-slate-400">{item.label}</span>
                              <span className="font-semibold text-slate-900 dark:text-white">{item.value}%</span>
                          </div>
                          <div className="h-1.5 overflow-hidden rounded-full bg-slate-200 dark:bg-white/[0.08]">
                            <motion.div
                              initial={{ scaleX: 0 }}
                              animate={{ scaleX: 1 }}
                              transition={{ duration: 1.2, delay: 0.5, ease: [0.22, 1, 0.36, 1] }}
                              className={`h-full rounded-full bg-gradient-to-r ${item.color} origin-left`}
                              style={{ scale: item.value / 100 }}
                            />
                          </div>
                        </div>
                      ))}
                    </div>
                  </GlassCard>
                </div>
              </motion.div>
            </div>
          </section>

          {/* Stats Bar */}
          <section className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
            <StatCard label="Conditions" value={9} icon={<Cpu size={20} />} delay={0} />
            <StatCard label="Accuracy" value={95} suffix="%" icon={<Zap size={20} />} delay={0.08} />
            <StatCard label="Inference" value={200} suffix="ms" icon={<Brain size={20} />} delay={0.16} />
            <StatCard label="Safe Mode" value={100} suffix="%" icon={<ShieldCheck size={20} />} delay={0.24} />
          </section>

          {/* Main Analyze Section */}
          <section id="analyze" className="grid gap-6 lg:grid-cols-[1.02fr,0.98fr]">
            <UploadAnalyze onResult={(nextResult, nextPreview) => { setResult(nextResult); setPreviewUrl(nextPreview); }} />
            <ResultsPanel result={result} previewUrl={previewUrl} onReset={() => { setResult(null); setPreviewUrl(null); }} />
          </section>
        </div>
      </main>
    </div>
  );
}
