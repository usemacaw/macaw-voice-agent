import { useEffect, useRef, useState } from "react";
import { useRealtimeSession } from "./hooks/useRealtimeSession";
import { Orb } from "./components/Orb";
import { TranscriptPanel } from "./components/TranscriptPanel";
import { MetricsPanel } from "./components/MetricsPanel";
import { Mic, MicOff, MessageSquareText, X, Activity } from "lucide-react";

type OrbState = "idle" | "listening" | "speaking" | "thinking";

function App() {
  const {
    status,
    messages,
    responseMetrics,
    isUserSpeaking,
    isAssistantSpeaking,
    connect,
    disconnect,
  } = useRealtimeSession();

  const [showTranscript, setShowTranscript] = useState(false);
  const [showMetrics, setShowMetrics] = useState(false);
  const lastAssistantText = useRef("");

  // Determine orb state
  let orbState: OrbState = "idle";
  if (status === "connected") {
    if (isAssistantSpeaking) orbState = "speaking";
    else if (isUserSpeaking) orbState = "listening";
    else {
      const lastMsg = messages[messages.length - 1];
      if (lastMsg && lastMsg.role === "assistant" && !lastMsg.isFinal && !lastMsg.text) {
        orbState = "thinking";
      }
    }
  }

  // Track latest assistant text for subtitle
  useEffect(() => {
    const assistantMsgs = messages.filter((m) => m.role === "assistant" && m.text);
    if (assistantMsgs.length > 0) {
      lastAssistantText.current = assistantMsgs[assistantMsgs.length - 1].text;
    }
  }, [messages]);

  const currentSubtitle = (() => {
    if (status !== "connected") return null;
    const lastMsg = messages[messages.length - 1];
    if (lastMsg?.role === "assistant" && lastMsg.text) {
      return lastMsg.text;
    }
    if (lastMsg?.role === "user" && lastMsg.text && isUserSpeaking) {
      return lastMsg.text;
    }
    return null;
  })();

  const statusLabel =
    status === "connected"
      ? orbState === "listening"
        ? "Ouvindo..."
        : orbState === "speaking"
          ? "Respondendo..."
          : orbState === "thinking"
            ? "Pensando..."
            : "Pronto para ouvir"
      : status === "connecting"
        ? "Conectando..."
        : "Desconectado";

  return (
    <div className="relative h-full flex flex-col items-center justify-center overflow-hidden">
      {/* Background gradient */}
      <div
        className="absolute inset-0"
        style={{
          background:
            "radial-gradient(ellipse at center, rgba(99,102,241,0.08) 0%, transparent 70%)",
        }}
      />

      {/* Top bar */}
      <div className="absolute top-0 left-0 right-0 flex items-center justify-between px-6 py-4 z-10">
        <div className="flex items-center gap-3">
          <div
            className={`w-2 h-2 rounded-full transition-colors ${
              status === "connected"
                ? "bg-emerald-400"
                : status === "connecting"
                  ? "bg-amber-400"
                  : "bg-zinc-600"
            }`}
            style={
              status === "connected"
                ? { boxShadow: "0 0 8px rgba(52,211,153,0.6)" }
                : undefined
            }
          />
          <span className="text-sm text-muted">{statusLabel}</span>
        </div>

        <div className="flex items-center gap-1">
          <button
            onClick={() => setShowMetrics(!showMetrics)}
            className="flex items-center gap-1.5 px-2.5 py-1.5 rounded-lg text-xs
                       text-muted hover:text-foreground hover:bg-surface-light
                       transition-colors cursor-pointer"
            title="Métricas"
          >
            <Activity size={14} />
            {responseMetrics.length > 0 && (
              <span className="text-[10px] font-mono text-accent">
                {responseMetrics.length}
              </span>
            )}
          </button>

          {messages.length > 0 && (
            <button
              onClick={() => setShowTranscript(!showTranscript)}
              className="flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm
                         text-muted hover:text-foreground hover:bg-surface-light
                         transition-colors cursor-pointer"
            >
              {showTranscript ? <X size={16} /> : <MessageSquareText size={16} />}
              {showTranscript ? "Fechar" : "Transcrição"}
            </button>
          )}
        </div>
      </div>

      {/* Central orb */}
      <div className="flex flex-col items-center gap-8 z-10">
        <Orb state={orbState} connected={status === "connected"} />

        {/* Subtitle */}
        <div className="h-16 flex items-start justify-center px-8 max-w-lg">
          {currentSubtitle ? (
            <p className="text-center text-sm text-muted leading-relaxed animate-fade-in">
              {currentSubtitle.length > 120
                ? "..." + currentSubtitle.slice(-120)
                : currentSubtitle}
            </p>
          ) : status === "connected" ? (
            <p className="text-center text-sm text-muted/50">
              Diga algo para iniciar a conversa
            </p>
          ) : null}
        </div>
      </div>

      {/* Bottom controls */}
      <div className="absolute bottom-0 left-0 right-0 flex justify-center pb-10 z-10">
        {status === "disconnected" ? (
          <button
            onClick={connect}
            className="flex items-center gap-3 px-8 py-4 rounded-full
                       bg-accent hover:bg-accent-glow text-white font-medium
                       text-base transition-all hover:scale-105
                       cursor-pointer"
            style={{
              boxShadow: "0 0 30px rgba(99,102,241,0.3)",
            }}
          >
            <Mic size={20} />
            Iniciar conversa
          </button>
        ) : status === "connecting" ? (
          <button
            disabled
            className="flex items-center gap-3 px-8 py-4 rounded-full
                       bg-surface-light text-muted font-medium text-base
                       cursor-wait"
          >
            <div className="w-5 h-5 border-2 border-muted border-t-transparent rounded-full animate-spin" />
            Conectando...
          </button>
        ) : (
          <button
            onClick={disconnect}
            className="flex items-center gap-3 px-6 py-3 rounded-full
                       bg-surface border border-border text-muted
                       hover:text-red-400 hover:border-red-400/50
                       font-medium text-sm transition-all cursor-pointer"
          >
            <MicOff size={18} />
            Encerrar
          </button>
        )}
      </div>

      {/* Transcript panel */}
      {showTranscript && (
        <TranscriptPanel
          messages={messages}
          onClose={() => setShowTranscript(false)}
        />
      )}

      {/* Metrics panel */}
      {showMetrics && (
        <MetricsPanel
          metrics={responseMetrics}
          onClose={() => setShowMetrics(false)}
        />
      )}
    </div>
  );
}

export default App;
