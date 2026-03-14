import { useEffect, useRef } from "react";
import { useRealtimeSession } from "./hooks/useRealtimeSession";
import "./App.css";

function StatusDot({ status }: { status: string }) {
  const color =
    status === "connected"
      ? "#4caf50"
      : status === "connecting"
        ? "#ff9800"
        : "#f44336";
  return <span className="status-dot" style={{ backgroundColor: color }} />;
}

function App() {
  const {
    status,
    messages,
    isUserSpeaking,
    isAssistantSpeaking,
    connect,
    disconnect,
  } = useRealtimeSession();

  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  return (
    <div className="app">
      <header className="header">
        <h1>Theo Voice Agent</h1>
        <div className="status">
          <StatusDot status={status} />
          <span>{status}</span>
        </div>
      </header>

      <main className="chat">
        {messages.length === 0 && status !== "connected" && (
          <div className="empty-state">
            <div className="empty-icon">🎙</div>
            <p>Conecte para iniciar uma conversa por voz</p>
          </div>
        )}

        {messages.length === 0 && status === "connected" && (
          <div className="empty-state">
            <div className="empty-icon listening">🎙</div>
            <p>Ouvindo... fale algo!</p>
          </div>
        )}

        {messages.map((msg) => (
          <div key={msg.id} className={`message ${msg.role}`}>
            <div className="bubble">
              {msg.text || (
                <span className="placeholder">
                  {msg.role === "user" ? "Ouvindo..." : "Pensando..."}
                </span>
              )}
            </div>
          </div>
        ))}

        <div ref={messagesEndRef} />
      </main>

      <footer className="controls">
        <div className="indicators">
          {isUserSpeaking && (
            <span className="indicator user-speaking">
              <span className="pulse" /> Falando...
            </span>
          )}
          {isAssistantSpeaking && (
            <span className="indicator assistant-speaking">
              <span className="pulse" /> Respondendo...
            </span>
          )}
        </div>

        {status === "disconnected" ? (
          <button className="btn connect" onClick={connect}>
            Conectar
          </button>
        ) : status === "connecting" ? (
          <button className="btn connecting" disabled>
            Conectando...
          </button>
        ) : (
          <button className="btn disconnect" onClick={disconnect}>
            Desconectar
          </button>
        )}
      </footer>
    </div>
  );
}

export default App;
