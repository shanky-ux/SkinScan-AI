"use client";

import { motion } from "framer-motion";
import { Activity, Menu, MoonStar, SunMedium, X } from "lucide-react";
import { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import { useTheme } from "@/components/theme-provider";

const navLinks = [
  { href: "/", label: "Analyzer" },
  { href: "/conditions", label: "Conditions" },
  { href: "/about", label: "About" },
];

export function Navbar() {
  const [scrolled, setScrolled] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const { theme, toggle } = useTheme();

  useEffect(() => {
    const handleScroll = () => setScrolled(window.scrollY > 20);
    window.addEventListener("scroll", handleScroll, { passive: true });
    return () => window.removeEventListener("scroll", handleScroll);
  }, []);

  return (
    <motion.nav
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.6, ease: [0.22, 1, 0.36, 1] }}
      className={cn(
        "fixed top-0 left-0 right-0 z-50 transition-all duration-500",
        scrolled
          ? "border-b border-slate-200 dark:border-white/10 bg-white/80 dark:bg-slate-950/80 shadow-[0_1px_0_rgba(0,0,0,0.06)] backdrop-blur-2xl"
          : "bg-transparent"
      )}
    >
      <div className="mx-auto flex w-full max-w-7xl items-center justify-between px-4 py-4 sm:px-6 lg:px-8">
        <a href="/" className="flex items-center gap-2.5 group">
          <div className="relative flex h-9 w-9 items-center justify-center rounded-xl bg-gradient-to-br from-accent-500 to-cyan-400 shadow-[0_8px_24px_rgba(24,189,224,0.3)] transition-transform duration-300 group-hover:scale-105">
            <Activity className="h-5 w-5 text-slate-950" strokeWidth={2.5} />
            <div className="absolute inset-0 rounded-xl bg-gradient-to-br from-accent-400 to-cyan-300 opacity-0 blur transition-opacity duration-300 group-hover:opacity-40" />
          </div>
            <span className="font-heading text-lg font-semibold text-slate-900 dark:text-white tracking-tight">
              SkinScan<span className="text-accent-300">AI</span>
            </span>
        </a>

        <div className="hidden items-center gap-1 md:flex">
          {navLinks.map((link) => (
            <a
              key={link.href}
              href={link.href}
              className="relative rounded-full px-4 py-2 text-sm font-medium text-slate-600 dark:text-slate-300 transition-colors duration-200 hover:text-slate-900 dark:hover:text-white"
            >
              {link.label}
            </a>
          ))}
        </div>

          <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={toggle}
            className="inline-flex h-10 w-10 items-center justify-center rounded-full border border-slate-200 dark:border-white/10 bg-slate-100 dark:bg-white/[0.05] text-slate-900 dark:text-white transition-all duration-200 hover:border-slate-300 dark:hover:border-white/20 hover:bg-slate-200 dark:hover:bg-white/[0.12]"
            aria-label="Toggle theme"
          >
            {theme === "dark" ? <SunMedium size={16} /> : <MoonStar size={16} />}
          </button>

          <button
            type="button"
            onClick={() => setMobileOpen((v) => !v)}
            className="inline-flex h-10 w-10 items-center justify-center rounded-full border border-slate-200 dark:border-white/10 bg-slate-100 dark:bg-white/[0.05] text-slate-900 dark:text-white transition-all duration-200 hover:border-slate-300 dark:hover:border-white/20 hover:bg-slate-200 dark:hover:bg-white/[0.12] md:hidden"
            aria-label="Toggle menu"
          >
            {mobileOpen ? <X size={16} /> : <Menu size={16} />}
          </button>
        </div>
      </div>

      {mobileOpen && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="border-b border-slate-200 dark:border-white/10 bg-white/95 dark:bg-slate-950/95 px-4 pb-4 backdrop-blur-2xl md:hidden"
          >
            <div className="flex flex-col gap-1">
              {navLinks.map((link) => (
                <a
                  key={link.href}
                  href={link.href}
                  onClick={() => setMobileOpen(false)}
                  className="rounded-xl px-4 py-3 text-sm font-medium text-slate-600 dark:text-slate-300 transition-colors duration-200 hover:bg-slate-100 dark:hover:bg-white/[0.06] hover:text-slate-900 dark:hover:text-white"
                >
                  {link.label}
                </a>
              ))}
            </div>
          </motion.div>
      )}
    </motion.nav>
  );
}
