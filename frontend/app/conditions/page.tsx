import { AnimatedBackground } from "@/components/background";
import { DiseaseLibrary } from "@/components/library";
import { Footer } from "@/components/footer";
import { classFallback } from "@/lib/data";

export default function ConditionsPage() {
  return (
    <div className="relative min-h-screen overflow-hidden bg-white dark:bg-slate-950 px-4 py-6 text-slate-900 dark:text-white sm:px-6 lg:px-8">
      <AnimatedBackground />
      <div className="relative mx-auto max-w-6xl space-y-10">
        <DiseaseLibrary classes={classFallback as never} />
        <Footer />
      </div>
    </div>
  );
}
