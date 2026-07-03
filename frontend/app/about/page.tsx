import { ModelArchitecture } from "@/components/model";
import { Footer } from "@/components/footer";
import { AnimatedBackground } from "@/components/background";
import { GlassCard, SectionHeading } from "@/components/ui";

export default function AboutPage() {
  return (
    <div className="relative min-h-screen overflow-hidden bg-white dark:bg-slate-950 px-4 py-6 text-slate-900 dark:text-white sm:px-6 lg:px-8">
      <AnimatedBackground />
      <div className="relative mx-auto max-w-6xl space-y-10">
        <section className="space-y-5">
          <SectionHeading eyebrow="About the model" title="How the classifier is wired" description="The architecture keeps the original preprocessing logic, exposes a clean API, and presents the ML pipeline as a visually guided flow rather than a text dump." />
          <GlassCard>
            <ModelArchitecture />
          </GlassCard>
        </section>
        <div className="flex flex-wrap gap-3">
            <a href="/" className="inline-flex items-center justify-center rounded-full bg-gradient-to-r from-accent-500 to-cyan-400 px-5 py-3 text-sm font-semibold text-slate-950 shadow-[0_12px_40px_rgba(24,189,224,0.28)]">
              Back to app
            </a>
            <a href="/conditions" className="inline-flex items-center justify-center rounded-full border border-slate-200 dark:border-white/12 bg-slate-100 dark:bg-white/5 px-5 py-3 text-sm font-semibold text-slate-900 dark:text-white hover:bg-slate-200 dark:hover:bg-white/[0.08]">
              Open library
            </a>
        </div>
        <Footer />
      </div>
    </div>
  );
}
