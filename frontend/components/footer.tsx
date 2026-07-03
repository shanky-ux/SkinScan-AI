import { Github, Globe, Mail } from "lucide-react";

export function Footer() {
  return (
    <footer className="border-t border-slate-200 dark:border-white/10 pt-8 text-sm text-slate-500 dark:text-slate-400">
      <div className="grid gap-6 md:grid-cols-[1.4fr,0.6fr]">
        <div>
          <p className="font-semibold text-slate-900 dark:text-white">Ravi Shankar</p>
          <p className="mt-2 max-w-xl leading-7 text-slate-600 dark:text-slate-300">B.Tech CSE AIML. SkinScan-AI is an educational AI-assisted classifier for skin condition triage, built as a portfolio-grade medical-tech demo.</p>
        </div>
        <div className="grid gap-3 justify-start md:justify-end">
          <a className="inline-flex items-center gap-2 text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white" href="https://ravish4nkar.vercel.app" target="_blank" rel="noreferrer"><Globe size={16} /> Portfolio</a>
          <a className="inline-flex items-center gap-2 text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white" href="https://github.com/shanky-ux" target="_blank" rel="noreferrer"><Github size={16} /> GitHub</a>
          <a className="inline-flex items-center gap-2 text-slate-600 dark:text-slate-300 hover:text-slate-900 dark:hover:text-white" href="mailto:ravi.shankar@example.com"><Mail size={16} /> Contact</a>
        </div>
      </div>
    </footer>
  );
}