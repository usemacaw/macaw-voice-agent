import { X } from "lucide-react";
import { useEffect, useRef } from "react";
import type { ResponseMetrics } from "../types";

interface MetricsPanelProps {
  metrics: ResponseMetrics[];
  onClose: () => void;
}

function fmtMs(ms: number | undefined): string {
  if (ms == null) return "\u2014";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(2)}s`;
}

function fmtNum(n: number | undefined): string {
  if (n == null) return "\u2014";
  return n.toLocaleString();
}

// ---- Visual bar ----
function MetricBar({
  label,
  value,
  max,
  color,
}: {
  label: string;
  value: number | undefined;
  max: number;
  color: string;
}) {
  if (value == null || value === 0) return null;
  const pct = Math.min((value / max) * 100, 100);
  return (
    <div className="flex items-center gap-2 text-xs">
      <span className="w-28 text-muted shrink-0 text-right truncate">{label}</span>
      <div className="flex-1 h-2.5 bg-surface rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, background: color }}
        />
      </div>
      <span className="w-14 text-foreground shrink-0 font-mono text-[11px]">{fmtMs(value)}</span>
    </div>
  );
}

// ---- Detail row ----
function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between text-[11px]">
      <span className="text-muted">{label}</span>
      <span className="text-foreground font-mono">{value}</span>
    </div>
  );
}

// ---- Per-response card ----
function ResponseCard({ entry, index }: { entry: ResponseMetrics; index: number }) {
  const maxBar = Math.max(entry.total_ms || 3000, 3000);
  const time = new Date(entry.timestamp).toLocaleTimeString();
  const hasTools = entry.tools_used && entry.tools_used.length > 0;

  return (
    <div className="bg-surface border border-border rounded-lg p-3.5 space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-xs font-medium text-foreground">
            Turn #{entry.turn ?? index + 1}
          </span>
          <span className="text-[10px] text-muted">{time}</span>
        </div>
        <div className="flex items-center gap-2">
          {hasTools && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-purple-500/20 text-purple-300">
              {entry.tools_used!.join(", ")}
            </span>
          )}
          {entry.slo_met != null && (
            <span className={`text-[10px] px-1.5 py-0.5 rounded font-mono ${
              entry.slo_met
                ? "bg-emerald-500/15 text-emerald-300"
                : "bg-red-500/15 text-red-300"
            }`}>
              SLO {entry.slo_met ? "OK" : "MISS"}
            </span>
          )}
          {entry.e2e_ms != null && (
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-red-500/15 text-red-300 font-mono">
              E2E {fmtMs(entry.e2e_ms)}
            </span>
          )}
        </div>
      </div>

      {/* Pipeline section: VAD + Turn Detection */}
      {(entry.vad_silence_wait_ms || entry.smart_turn_inference_ms) && (
        <div className="space-y-1">
          <span className="text-[10px] text-muted font-medium uppercase tracking-wider">VAD / Turn</span>
          <MetricBar label="VAD Silence" value={entry.vad_silence_wait_ms} max={maxBar} color="#f97316" />
          <MetricBar label="Smart Turn" value={entry.smart_turn_inference_ms} max={maxBar} color="#fb923c" />
        </div>
      )}

      {/* Pipeline section: ASR */}
      <div className="space-y-1">
        <span className="text-[10px] text-muted font-medium uppercase tracking-wider">ASR</span>
        <MetricBar label="ASR" value={entry.asr_ms} max={maxBar} color="#f59e0b" />
      </div>

      {/* Pipeline section: LLM */}
      <div className="space-y-1">
        <span className="text-[10px] text-muted font-medium uppercase tracking-wider">LLM</span>
        <MetricBar label="TTFT" value={entry.llm_ttft_ms} max={maxBar} color="#6366f1" />
        <MetricBar label="1a Frase" value={entry.llm_first_sentence_ms} max={maxBar} color="#a78bfa" />
        <MetricBar label="Total" value={entry.llm_total_ms} max={maxBar} color="#818cf8" />
      </div>

      {/* Pipeline section: TTS */}
      <div className="space-y-1">
        <span className="text-[10px] text-muted font-medium uppercase tracking-wider">TTS</span>
        <MetricBar label="1o Chunk" value={entry.tts_first_chunk_ms} max={maxBar} color="#c084fc" />
        <MetricBar label="Synth" value={entry.tts_synth_ms} max={maxBar} color="#8b5cf6" />
        <MetricBar label="Queue Wait" value={entry.tts_wait_ms} max={maxBar} color="#7c3aed" />
      </div>

      {/* Pipeline section: Delivery */}
      <div className="space-y-1">
        <span className="text-[10px] text-muted font-medium uppercase tracking-wider">Delivery</span>
        <MetricBar label="Encode+Send" value={entry.encode_send_ms} max={maxBar} color="#14b8a6" />
        <MetricBar label="1o Audio" value={entry.pipeline_first_audio_ms} max={maxBar} color="#10b981" />
        <MetricBar label="E2E" value={entry.e2e_ms} max={maxBar} color="#ef4444" />
        <MetricBar label="Total" value={entry.total_ms} max={maxBar} color="#3b82f6" />
      </div>

      {/* Tool timings */}
      {entry.tool_timings && entry.tool_timings.length > 0 && (
        <div className="space-y-0.5 pt-1 border-t border-border/40">
          <span className="text-[10px] text-muted font-medium">Ferramentas</span>
          {entry.tool_timings.map((t, i) => (
            <div key={i} className="flex items-center justify-between text-[11px]">
              <div className="flex items-center gap-1.5">
                <span className={`w-1.5 h-1.5 rounded-full ${t.ok ? "bg-emerald-400" : "bg-red-400"}`} />
                <span className="text-foreground">{t.name}</span>
              </div>
              <span className="font-mono text-muted">{fmtMs(t.exec_ms)}</span>
            </div>
          ))}
        </div>
      )}

      {/* Detail grid */}
      <div className="grid grid-cols-2 gap-x-6 gap-y-0.5 pt-1 border-t border-border/40">
        {entry.speech_ms != null && <DetailRow label="Fala" value={fmtMs(entry.speech_ms)} />}
        {entry.speech_rms != null && <DetailRow label="RMS (vol.)" value={fmtNum(Math.round(entry.speech_rms))} />}
        {entry.asr_mode != null && <DetailRow label="ASR modo" value={entry.asr_mode} />}
        {entry.asr_partial_count != null && entry.asr_partial_count > 0 && <DetailRow label="ASR partials" value={fmtNum(entry.asr_partial_count)} />}
        {entry.smart_turn_waits != null && entry.smart_turn_waits > 0 && <DetailRow label="Smart Turn waits" value={fmtNum(entry.smart_turn_waits)} />}
        {entry.input_chars != null && <DetailRow label="Input chars" value={fmtNum(entry.input_chars)} />}
        {entry.output_chars != null && <DetailRow label="Output chars" value={fmtNum(entry.output_chars)} />}
        {entry.sentences != null && entry.sentences > 0 && <DetailRow label="Frases" value={fmtNum(entry.sentences)} />}
        {entry.audio_chunks != null && entry.audio_chunks > 0 && <DetailRow label="Audio chunks" value={fmtNum(entry.audio_chunks)} />}
        {entry.tool_rounds != null && entry.tool_rounds > 0 && <DetailRow label="Tool rounds" value={fmtNum(entry.tool_rounds)} />}
        {entry.backpressure_level != null && entry.backpressure_level > 0 && (
          <DetailRow label="Backpressure" value={`L${entry.backpressure_level}`} />
        )}
        {entry.events_dropped != null && entry.events_dropped > 0 && (
          <DetailRow label="Events dropped" value={fmtNum(entry.events_dropped)} />
        )}
        {entry.session_duration_s != null && <DetailRow label="Sessao" value={`${entry.session_duration_s.toFixed(0)}s`} />}
        {entry.barge_in_count != null && entry.barge_in_count > 0 && <DetailRow label="Interrupcoes" value={fmtNum(entry.barge_in_count)} />}
      </div>
    </div>
  );
}

// ---- Summary / Averages ----
function SummaryCard({ metrics }: { metrics: ResponseMetrics[] }) {
  if (metrics.length === 0) return null;

  const avg = (key: keyof ResponseMetrics) => {
    const values = metrics
      .map((m) => m[key])
      .filter((v): v is number => typeof v === "number" && v > 0);
    if (values.length === 0) return undefined;
    return values.reduce((a, b) => a + b, 0) / values.length;
  };

  const min = (key: keyof ResponseMetrics) => {
    const values = metrics
      .map((m) => m[key])
      .filter((v): v is number => typeof v === "number" && v > 0);
    if (values.length === 0) return undefined;
    return Math.min(...values);
  };

  const max = (key: keyof ResponseMetrics) => {
    const values = metrics
      .map((m) => m[key])
      .filter((v): v is number => typeof v === "number" && v > 0);
    if (values.length === 0) return undefined;
    return Math.max(...values);
  };

  const last = metrics[metrics.length - 1];
  const totalToolCalls = metrics.reduce(
    (sum, m) => sum + (m.tool_timings?.length ?? 0),
    0
  );
  const totalBargeIns = last?.barge_in_count ?? 0;

  return (
    <div className="bg-surface-light border border-border rounded-lg p-4 space-y-3">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-foreground">
          Resumo \u2014 {metrics.length} respostas
        </span>
        {last?.session_duration_s != null && (
          <span className="text-[10px] text-muted font-mono">
            Sessao: {last.session_duration_s.toFixed(0)}s
          </span>
        )}
      </div>

      {/* Key KPIs: E2E */}
      <div className="grid grid-cols-3 gap-2">
        <KpiCell label="E2E (avg)" value={fmtMs(avg("e2e_ms"))} color="text-red-300" />
        <KpiCell label="E2E (min)" value={fmtMs(min("e2e_ms"))} color="text-emerald-300" />
        <KpiCell label="E2E (max)" value={fmtMs(max("e2e_ms"))} color="text-amber-300" />
      </div>

      {/* Full pipeline stage averages */}
      <div className="grid grid-cols-4 gap-2">
        <KpiCell label="VAD Silence" value={fmtMs(avg("vad_silence_wait_ms"))} />
        <KpiCell label="Smart Turn" value={fmtMs(avg("smart_turn_inference_ms"))} />
        <KpiCell label="ASR" value={fmtMs(avg("asr_ms"))} />
        <KpiCell label="LLM TTFT" value={fmtMs(avg("llm_ttft_ms"))} />
        <KpiCell label="1a Frase" value={fmtMs(avg("llm_first_sentence_ms"))} />
        <KpiCell label="TTS 1o Chunk" value={fmtMs(avg("tts_first_chunk_ms"))} />
        <KpiCell label="1o Audio" value={fmtMs(avg("pipeline_first_audio_ms"))} />
        <KpiCell label="Total" value={fmtMs(avg("total_ms"))} />
      </div>

      {/* Counters */}
      <div className="flex gap-4 text-[10px] text-muted pt-1 border-t border-border/40">
        <span>Turns: {metrics.length}</span>
        {totalBargeIns > 0 && <span>Barge-ins: {totalBargeIns}</span>}
        {totalToolCalls > 0 && <span>Tool calls: {totalToolCalls}</span>}
      </div>
    </div>
  );
}

function KpiCell({
  label,
  value,
  color,
}: {
  label: string;
  value: string;
  color?: string;
}) {
  return (
    <div className="text-center">
      <div className="text-[10px] text-muted">{label}</div>
      <div className={`text-sm font-mono ${color || "text-foreground"}`}>{value}</div>
    </div>
  );
}

// ---- Main panel ----
export function MetricsPanel({ metrics, onClose }: MetricsPanelProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [metrics]);

  return (
    <div className="absolute inset-0 z-30 flex justify-end">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/40" onClick={onClose} />

      {/* Panel */}
      <div className="relative w-full max-w-md h-full bg-bg/95 backdrop-blur-xl border-l border-border animate-slide-in-right flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between px-5 py-4 border-b border-border">
          <div>
            <h2 className="text-sm font-medium text-foreground">Observabilidade</h2>
            <p className="text-[10px] text-muted mt-0.5">Pipeline Voice-to-Voice</p>
          </div>
          <button
            onClick={onClose}
            className="p-1 rounded-md text-muted hover:text-foreground hover:bg-surface-light transition-colors cursor-pointer"
          >
            <X size={16} />
          </button>
        </div>

        {/* Content */}
        <div ref={scrollRef} className="flex-1 overflow-y-auto p-4 space-y-3">
          {metrics.length === 0 ? (
            <div className="text-center py-12 space-y-2">
              <p className="text-sm text-muted">Nenhuma metrica ainda.</p>
              <p className="text-xs text-muted/60">
                As metricas aparecem apos cada resposta do agente.
              </p>
            </div>
          ) : (
            <>
              <SummaryCard metrics={metrics} />
              {metrics.map((entry, i) => (
                <ResponseCard key={entry.response_id} entry={entry} index={i} />
              ))}
            </>
          )}
        </div>
      </div>
    </div>
  );
}
