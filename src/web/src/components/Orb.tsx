type OrbState = "idle" | "listening" | "speaking" | "thinking";

interface OrbProps {
  state: OrbState;
  connected: boolean;
}

export function Orb({ state, connected }: OrbProps) {
  const isActive = connected && state !== "idle";

  const orbAnimation = (() => {
    if (!connected) return "opacity-30";
    switch (state) {
      case "listening":
        return "animate-orb-listening";
      case "speaking":
        return "animate-orb-speaking";
      case "thinking":
        return "animate-orb-thinking opacity-70";
      default:
        return "animate-orb-idle";
    }
  })();

  const gradientStyle = (() => {
    if (!connected) {
      return "radial-gradient(circle at 40% 40%, #3f3f46, #18181b)";
    }
    switch (state) {
      case "listening":
        return "radial-gradient(circle at 35% 35%, #818cf8, #6366f1, #4338ca)";
      case "speaking":
        return "radial-gradient(circle at 35% 35%, #a78bfa, #7c3aed, #6366f1)";
      case "thinking":
        return "radial-gradient(circle at 35% 35%, #6366f1, #4338ca, #312e81)";
      default:
        return "radial-gradient(circle at 35% 35%, #6366f1, #4338ca, #1e1b4b)";
    }
  })();

  const glowColor = (() => {
    switch (state) {
      case "listening":
        return "rgba(99, 102, 241, 0.4)";
      case "speaking":
        return "rgba(139, 92, 246, 0.5)";
      case "thinking":
        return "rgba(99, 102, 241, 0.25)";
      default:
        return "rgba(99, 102, 241, 0.15)";
    }
  })();

  const ringColor =
    state === "speaking"
      ? "rgba(139,92,246,0.3)"
      : "rgba(99,102,241,0.3)";

  const ringColorFaint =
    state === "speaking"
      ? "rgba(139,92,246,0.2)"
      : "rgba(99,102,241,0.2)";

  return (
    <div className="relative flex items-center justify-center w-52 h-52">
      {/* Pulse rings */}
      {isActive && (
        <>
          <div
            className="absolute w-40 h-40 rounded-full animate-ring-pulse"
            style={{ border: `1.5px solid ${ringColor}` }}
          />
          <div
            className="absolute w-40 h-40 rounded-full animate-ring-pulse-delay"
            style={{ border: `1px solid ${ringColorFaint}` }}
          />
        </>
      )}

      {/* Glow */}
      <div
        className="absolute w-48 h-48 rounded-full blur-2xl transition-all duration-700"
        style={{ background: glowColor }}
      />

      {/* Orb */}
      <div
        className={`relative w-40 h-40 rounded-full transition-all duration-500 ${orbAnimation}`}
        style={{
          background: gradientStyle,
          boxShadow: `
            inset 0 -8px 20px rgba(0,0,0,0.3),
            inset 0 4px 12px rgba(255,255,255,0.08),
            0 0 40px ${glowColor}
          `,
        }}
      >
        {/* Glass highlight */}
        <div
          className="absolute top-3 left-6 w-16 h-8 rounded-full opacity-20"
          style={{
            background:
              "linear-gradient(180deg, rgba(255,255,255,0.4), transparent)",
          }}
        />
      </div>
    </div>
  );
}
