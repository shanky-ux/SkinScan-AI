import type { Config } from "tailwindcss";

const config: Config = {
  darkMode: "class",
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}", "./lib/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "hsl(var(--bg))",
        panel: "hsl(var(--panel))",
        line: "hsl(var(--line))",
        text: "hsl(var(--text))",
        muted: "hsl(var(--muted))",
        accent: {
          50: "#eefcff",
          100: "#d8fbff",
          200: "#b9f4ff",
          300: "#7fe9ff",
          400: "#43d6f8",
          500: "#18bde0",
          600: "#1497ba",
          700: "#156f8f",
        },
      },
      boxShadow: {
        glow: "0 0 0 1px rgba(255,255,255,0.06), 0 18px 50px rgba(0,0,0,0.45)",
        "glow-sm": "0 0 0 1px rgba(255,255,255,0.04), 0 8px 24px rgba(0,0,0,0.35)",
        "glow-accent": "0 0 0 1px rgba(24,189,224,0.2), 0 12px 40px rgba(24,189,224,0.25)",
      },
      backgroundImage: {
        mesh: "radial-gradient(circle at top left, rgba(24,189,224,0.18), transparent 32%), radial-gradient(circle at 80% 20%, rgba(121,93,255,0.14), transparent 28%), radial-gradient(circle at 50% 100%, rgba(16,185,129,0.12), transparent 24%)",
      },
      keyframes: {
        float: {
          "0%, 100%": { transform: "translate3d(0,0,0)" },
          "50%": { transform: "translate3d(0,-14px,0)" },
        },
        pulseGlow: {
          "0%, 100%": { opacity: "0.45" },
          "50%": { opacity: "0.85" },
        },
        shimmer: {
          "0%": { backgroundPosition: "-200% 0" },
          "100%": { backgroundPosition: "200% 0" },
        },
        scan: {
          "0%": { top: "0%" },
          "50%": { top: "100%" },
          "100%": { top: "0%" },
        },
      },
      animation: {
        float: "float 10s ease-in-out infinite",
        pulseGlow: "pulseGlow 5s ease-in-out infinite",
        shimmer: "shimmer 2s infinite linear",
        scan: "scan 2s ease-in-out infinite",
      },
      borderRadius: {
        '3xl': '1.75rem',
      },
    },
  },
  plugins: [],
};

export default config;
