/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#09090b",
        surface: "#18181b",
        "surface-light": "#27272a",
        border: "#3f3f46",
        muted: "#71717a",
        foreground: "#fafafa",
        accent: "#6366f1",
        "accent-glow": "#818cf8",
      },
      keyframes: {
        "orb-idle": {
          "0%, 100%": { transform: "scale(1)", opacity: "0.6" },
          "50%": { transform: "scale(1.03)", opacity: "0.8" },
        },
        "orb-listening": {
          "0%, 100%": { transform: "scale(1)" },
          "25%": { transform: "scale(1.08) rotate(1deg)" },
          "50%": { transform: "scale(0.95)" },
          "75%": { transform: "scale(1.05) rotate(-1deg)" },
        },
        "orb-speaking": {
          "0%, 100%": { transform: "scale(1)", filter: "brightness(1)" },
          "20%": { transform: "scale(1.12)", filter: "brightness(1.2)" },
          "40%": { transform: "scale(0.92)", filter: "brightness(0.9)" },
          "60%": { transform: "scale(1.08)", filter: "brightness(1.15)" },
          "80%": { transform: "scale(0.96)", filter: "brightness(1)" },
        },
        "ring-pulse": {
          "0%": { transform: "scale(1)", opacity: "0.4" },
          "100%": { transform: "scale(1.8)", opacity: "0" },
        },
        "ring-pulse-delay": {
          "0%": { transform: "scale(1)", opacity: "0.3" },
          "100%": { transform: "scale(2.2)", opacity: "0" },
        },
        "fade-in": {
          from: { opacity: "0", transform: "translateY(8px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
        "slide-in-right": {
          from: { transform: "translateX(100%)" },
          to: { transform: "translateX(0)" },
        },
      },
      animation: {
        "orb-idle": "orb-idle 4s ease-in-out infinite",
        "orb-listening": "orb-listening 1.2s ease-in-out infinite",
        "orb-speaking": "orb-speaking 0.8s ease-in-out infinite",
        "orb-thinking": "orb-idle 2s ease-in-out infinite",
        "ring-pulse": "ring-pulse 2s ease-out infinite",
        "ring-pulse-delay": "ring-pulse-delay 2s ease-out infinite 0.6s",
        "fade-in": "fade-in 0.3s ease-out",
        "slide-in-right": "slide-in-right 0.25s ease-out",
      },
    },
  },
  plugins: [],
};
