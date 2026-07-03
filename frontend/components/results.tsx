"use client";

import { motion, AnimatePresence } from "framer-motion";
import { RefreshCcw, ShieldAlert, TrendingUp, AlertTriangle, Info, BarChart3 } from "lucide-react";
import { GlassCard, PrimaryButton, Badge } from "@/components/ui";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";
import type { PredictResponse } from "@/lib/types";
import { cn } from "@/lib/utils";

const chartColors = ["#18bde0", "#6366f1", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6", "#ec4899", "#14b8a6", "#f97316"];

export function ResultsPanel({ result, previewUrl, onReset }: { result: PredictResponse | null; previewUrl: string | null; onReset: () => void }) {
  return (
    <div className="space-y-4">
      <AnimatePresence mode="wait">
        {!result ? (
          <motion.div key="empty" initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -18 }} transition={{ duration: 0.35 }}>
            <GlassCard className="flex min-h-[420px] flex-col items-center justify-center text-center p-8">
              <motion.div initial={{ scale: 0.8, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ delay: 0.1, type: "spring", stiffness: 200 }} className="mb-6 inline-flex h-16 w-16 items-center justify-center rounded-2xl bg-slate-100 dark:bg-white/[0.05] ring-1 ring-slate-200 dark:ring-white/10">
                <ShieldAlert className="text-accent-300" size={28} />
              </motion.div>
              <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
                <h3 className="font-heading text-2xl text-slate-900 dark:text-white">Results area</h3>
                <p className="mt-3 max-w-sm text-sm leading-7 text-slate-600 dark:text-slate-300">Your analysis result will show here after you press analyze.</p>
              </motion.div>
            </GlassCard>
          </motion.div>
        ) : (
          <motion.div key="result" initial={{ opacity: 0, y: 18 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -18 }} transition={{ duration: 0.35 }} className="space-y-4">
            {result && (
              <>
                <GlassCard className="space-y-5">
                  <div className="flex items-start justify-between gap-3">
                    <div>
                       <p className="text-xs uppercase tracking-[0.3em] text-slate-500 dark:text-slate-400">Analysis result</p>
                       <motion.h3 initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.1 }} className="mt-3 font-heading text-3xl text-slate-900 dark:text-white md:text-4xl">
                        {result.predicted_class}
                      </motion.h3>
                    </div>
                    <motion.div initial={{ scale: 0.8, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} transition={{ delay: 0.2, type: "spring" }}>
                      <Badge variant={result.disease_info.severity_level === "high" ? "danger" : result.disease_info.severity_level === "moderate" ? "warning" : result.disease_info.severity_level === "low" ? "success" : "default"}>
                        {result.disease_info.severity}
                      </Badge>
                    </motion.div>
                  </div>
                  <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.2 }}>
                    <ConfidenceGauge value={result.confidence} />
                  </motion.div>
                   <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.3 }} className="rounded-2xl border border-slate-200 dark:border-white/10 bg-slate-100 dark:bg-white/[0.03] p-4">
                     <div className="flex items-start gap-3">
                       <Info className="mt-0.5 h-4 w-4 shrink-0 text-accent-300" />
                       <div className="space-y-2 text-sm leading-7 text-slate-600 dark:text-slate-300">
                         <p><span className="font-semibold text-slate-900 dark:text-white">Description:</span> {result.disease_info.description}</p>
                         <p><span className="font-semibold text-slate-900 dark:text-white">Recommendation:</span> {result.disease_info.recommendation}</p>
                       </div>
                     </div>
                   </motion.div>
                   <motion.div initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.4 }} className="rounded-2xl border border-slate-200 dark:border-white/10 bg-slate-100 dark:bg-white/[0.02] p-4">
                     <details className="group">
                       <summary className="flex cursor-pointer items-center justify-between font-medium text-slate-900 dark:text-white">
                        <span className="flex items-center gap-2">
                          <BarChart3 size={16} className="text-accent-300" />
                          Probability Distribution
                        </span>
                        <ChevronDown className="h-4 w-4 text-slate-400" />
                      </summary>
                      <div className="mt-4 h-64 w-full">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={result.probabilities} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                            <XAxis dataKey="class_name" tick={{ fontSize: 10, fill: "#94a3b8" }} axisLine={{ stroke: "rgba(255,255,255,0.08)" }} tickLine={false} interval={0} angle={-30} textAnchor="end" height={60} />
                            <YAxis tick={{ fontSize: 10, fill: "#94a3b8" }} axisLine={{ stroke: "rgba(255,255,255,0.08)" }} tickLine={false} tickFormatter={(value: number) => `${(value * 100).toFixed(0)}%`} />
                            <Tooltip contentStyle={{ backgroundColor: "rgba(15,23,42,0.95)", border: "1px solid rgba(255,255,255,0.1)", borderRadius: "12px", backdropFilter: "blur(12px)" }} labelStyle={{ color: "#e2e8f0", fontSize: "12px" }} formatter={(value: number) => [`${(value * 100).toFixed(1)}%`, "Probability"]} />
                            <Bar dataKey="probability" radius={[4, 4, 0, 0]}>
                              {result.probabilities.map((entry, index) => (
                                <Cell key={`cell-${index}`} fill={chartColors[index % chartColors.length]} />
                              ))}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </details>
                  </motion.div>
                   <details className="rounded-2xl border border-slate-200 dark:border-white/10 bg-slate-100 dark:bg-white/[0.02] p-4 text-sm text-slate-600 dark:text-slate-300">
                     <summary className="cursor-pointer select-none font-medium text-slate-900 dark:text-white">View analysis details</summary>
                     <div className="mt-4 grid gap-3 sm:grid-cols-2">
                       <div className="rounded-xl border border-slate-200 dark:border-white/10 bg-slate-100 dark:bg-white/[0.03] p-3">
                         <p className="text-xs uppercase tracking-[0.25em] text-slate-500 dark:text-slate-400">Confidence</p>
                         <p className="mt-2 text-lg font-semibold text-slate-900 dark:text-white">{Math.round(result.confidence)}%</p>
                       </div>
                       <div className="rounded-xl border border-slate-200 dark:border-white/10 bg-slate-100 dark:bg-white/[0.03] p-3">
                         <p className="text-xs uppercase tracking-[0.25em] text-slate-500 dark:text-slate-400">Model mode</p>
                         <p className="mt-2 text-lg font-semibold text-slate-900 dark:text-white capitalize">{result.model_mode}</p>
                       </div>
                       <div className="rounded-xl border border-slate-200 dark:border-white/10 bg-slate-100 dark:bg-white/[0.03] p-3 sm:col-span-2">
                         <p className="text-xs uppercase tracking-[0.25em] text-slate-500 dark:text-slate-400">Architecture</p>
                         <p className="mt-2 text-lg font-semibold text-slate-900 dark:text-white">{result.model_architecture}</p>
                       </div>
                     </div>
                   </details>
                  <div className="rounded-xl border border-amber-300/20 bg-amber-400/10 p-3 text-sm leading-7 text-amber-100 flex items-start gap-2">
                    <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0" />
                    Educational use only. This result is not a medical diagnosis.
                  </div>
                  <div className="flex flex-wrap gap-3">
                    <PrimaryButton onClick={onReset} type="button">
                      <RefreshCcw className="mr-2 h-4 w-4" /> Analyze another image
                    </PrimaryButton>
                  </div>
                </GlassCard>
              </>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function ChevronDown(props: React.SVGProps<SVGSVGElement>) {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round" {...props}>
      <path d="m6 9 6 6 6-6" />
    </svg>
  );
}

function ConfidenceGauge({ value }: { value: number }) {
  const clamped = Math.max(0, Math.min(100, value));
  const circumference = 2 * Math.PI * 54;
  const offset = circumference - (clamped / 100) * circumference;

  return (
    <div className="flex items-center gap-6">
      <div className="relative h-32 w-32 shrink-0">
        <svg className="h-full w-full -rotate-90" viewBox="0 0 120 120">
          <circle cx="60" cy="60" r="54" fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="8" />
          <motion.circle cx="60" cy="60" r="54" fill="none" stroke="url(#gaugeGradient)" strokeWidth="8" strokeLinecap="round" strokeDasharray={circumference} initial={{ strokeDashoffset: circumference }} animate={{ strokeDashoffset: offset }} transition={{ duration: 1.2, delay: 0.3, ease: [0.22, 1, 0.36, 1] }} />
          <defs>
            <linearGradient id="gaugeGradient" x1="0%" y1="0%" x2="100%" y2="0%">
              <stop offset="0%" stopColor="#18bde0" />
              <stop offset="100%" stopColor="#6366f1" />
            </linearGradient>
          </defs>
        </svg>
        <div className="absolute inset-0 flex items-center justify-center">
          <div className="text-center">
             <div className="text-3xl font-semibold text-slate-900 dark:text-white">{Math.round(clamped)}%</div>
             <div className="text-[10px] uppercase tracking-[0.25em] text-slate-500 dark:text-slate-400">confidence</div>
          </div>
        </div>
      </div>
      <div className="space-y-2">
        <div className="flex items-center gap-2 text-sm text-slate-600 dark:text-slate-300">
          <TrendingUp className="h-4 w-4 text-emerald-400" />
          High confidence model
        </div>
        <p className="text-xs leading-6 text-slate-500 dark:text-slate-400">Higher confidence means the model is more certain about the top prediction.</p>
      </div>
    </div>
  );
}
