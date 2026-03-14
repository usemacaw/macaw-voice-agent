import { useEffect, useRef } from "react";
import { X } from "lucide-react";
import type { Message } from "../types";

interface TranscriptPanelProps {
  messages: Message[];
  onClose: () => void;
}

export function TranscriptPanel({ messages, onClose }: TranscriptPanelProps) {
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <>
      {/* Backdrop */}
      <div
        className="absolute inset-0 bg-black/40 z-20 backdrop-blur-sm"
        onClick={onClose}
      />

      {/* Panel */}
      <div
        className="absolute right-0 top-0 bottom-0 w-full max-w-md z-30
                   bg-surface/95 backdrop-blur-lg border-l border-border
                   flex flex-col animate-[slide-in-right_0.25s_ease-out]"
      >
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-border">
          <h2 className="text-sm font-semibold text-foreground">Transcrição</h2>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-muted hover:text-foreground
                       hover:bg-surface-light transition-colors cursor-pointer"
          >
            <X size={18} />
          </button>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4">
          {messages.length === 0 ? (
            <p className="text-sm text-muted text-center mt-8">
              Nenhuma mensagem ainda
            </p>
          ) : (
            messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}
                           animate-[fade-in_0.2s_ease-out]`}
              >
                <div
                  className={`max-w-[85%] px-4 py-2.5 rounded-2xl text-sm leading-relaxed
                    ${
                      msg.role === "user"
                        ? "bg-accent/20 text-foreground rounded-br-md"
                        : "bg-surface-light text-foreground/90 rounded-bl-md"
                    }`}
                >
                  {msg.text || (
                    <span className="text-muted italic text-xs">
                      {msg.role === "user" ? "Ouvindo..." : "Pensando..."}
                    </span>
                  )}
                </div>
              </div>
            ))
          )}
          <div ref={bottomRef} />
        </div>
      </div>
    </>
  );
}
