import { useCallback, useRef, useState } from "react";
import { AudioCapture } from "../audio/capture";
import { AudioPlayback } from "../audio/playback";
import type { Message, ResponseMetrics, SessionStatus, ToolTiming } from "../types";

const WS_URL =
  import.meta.env.VITE_WS_URL || "ws://localhost:8765/v1/realtime";

function arrayBufferToBase64(buffer: ArrayBuffer): string {
  const bytes = new Uint8Array(buffer);
  let binary = "";
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

// Metrics tracker
class SessionMetrics {
  chunksSent = 0;
  bytesSent = 0;
  eventsReceived = 0;
  audioDeltas = 0;
  audioBytesReceived = 0;
  private lastLog = Date.now();

  logChunk(bytes: number) {
    this.chunksSent++;
    this.bytesSent += bytes;
    const now = Date.now();
    if (now - this.lastLog >= 5000) {
      const audioDurMs = this.bytesSent / (24000 * 2) * 1000;
      console.log(
        `[OVA] METRICS: sent ${this.chunksSent} chunks (${this.bytesSent} bytes, ${audioDurMs.toFixed(0)}ms audio), ` +
        `received ${this.eventsReceived} events, ${this.audioDeltas} audio deltas (${this.audioBytesReceived} bytes)`
      );
      this.lastLog = now;
    }
  }

  logEvent(type: string) {
    this.eventsReceived++;
    // Log all events except audio deltas (too noisy)
    if (type !== "response.audio.delta" && type !== "input_audio_buffer.append") {
      console.log(`[OVA] ← ${type}`);
    }
  }
}

export function useRealtimeSession() {
  const [status, setStatus] = useState<SessionStatus>("disconnected");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isUserSpeaking, setIsUserSpeaking] = useState(false);
  const [isAssistantSpeaking, setIsAssistantSpeaking] = useState(false);
  const [responseMetrics, setResponseMetrics] = useState<ResponseMetrics[]>([]);

  const wsRef = useRef<WebSocket | null>(null);
  const captureRef = useRef<AudioCapture | null>(null);
  const playbackRef = useRef<AudioPlayback | null>(null);
  const metricsRef = useRef<SessionMetrics>(new SessionMetrics());

  // Track current assistant message id for streaming updates
  const currentAssistantIdRef = useRef<string | null>(null);
  // Track pending user speech item id
  const pendingUserItemRef = useRef<string | null>(null);

  // Batched transcript delta updates — accumulate deltas and flush via rAF
  const pendingTranscriptRef = useRef<string>("");
  const rafIdRef = useRef<number | null>(null);

  const send = useCallback((event: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(event));
    }
  }, []);

  const addMessage = useCallback((msg: Message) => {
    setMessages((prev) => [...prev, msg]);
  }, []);

  const updateMessage = useCallback((id: string, update: Partial<Message>) => {
    setMessages((prev) =>
      prev.map((m) => (m.id === id ? { ...m, ...update } : m))
    );
  }, []);

  const handleServerEvent = useCallback(
    (data: Record<string, unknown>) => {
      const type = data.type as string;
      const metrics = metricsRef.current;
      metrics.logEvent(type);

      switch (type) {
        case "session.created":
          console.log("[OVA] Session created, configuring audio + VAD...");
          // Configure session for audio with server VAD
          send({
            type: "session.update",
            session: {
              modalities: ["text", "audio"],
              input_audio_format: "pcm16",
              output_audio_format: "pcm16",
              turn_detection: {
                type: "server_vad",
                silence_duration_ms: 250,
                create_response: true,
                interrupt_response: true,
              },
            },
          });
          break;

        case "session.updated":
          console.log("[OVA] Session configured, ready to receive audio");
          setStatus("connected");
          break;

        case "input_audio_buffer.speech_started": {
          console.log(`[OVA] Speech started (item=${(data.item_id as string)?.slice(0, 12)})`);
          setIsUserSpeaking(true);
          // Interrupt assistant playback (barge-in)
          playbackRef.current?.clear();
          setIsAssistantSpeaking(false);

          const itemId =
            (data.item_id as string) || `user_${Date.now()}`;
          pendingUserItemRef.current = itemId;
          addMessage({
            id: itemId,
            role: "user",
            text: "",
            isFinal: false,
          });
          break;
        }

        case "input_audio_buffer.speech_stopped":
          console.log("[OVA] Speech stopped, waiting for transcription...");
          setIsUserSpeaking(false);
          break;

        case "conversation.item.input_audio_transcription.completed": {
          const itemId = data.item_id as string;
          const transcript = data.transcript as string;
          console.log(`[OVA] Transcription: "${transcript}"`);
          if (itemId && transcript) {
            updateMessage(itemId, { text: transcript, isFinal: true });
          }
          break;
        }

        case "response.created":
          console.log("[OVA] Response started...");
          break;

        case "response.output_item.added": {
          const item = data.item as Record<string, unknown> | undefined;
          const itemId = item?.id as string;
          if (itemId && item?.role === "assistant") {
            currentAssistantIdRef.current = itemId;
            addMessage({
              id: itemId,
              role: "assistant",
              text: "",
              isFinal: false,
            });
            setIsAssistantSpeaking(true);
          }
          break;
        }

        case "response.audio.delta": {
          const delta = data.delta as string;
          if (delta) {
            metrics.audioDeltas++;
            const bytes = atob(delta).length;
            metrics.audioBytesReceived += bytes;
            playbackRef.current?.enqueue(delta);
          }
          break;
        }

        case "response.audio_transcript.delta": {
          const id = currentAssistantIdRef.current;
          const textDelta = data.delta as string;
          if (id && textDelta) {
            // Batch transcript deltas: accumulate and flush once per animation frame
            // to avoid 100+ React re-renders per response
            pendingTranscriptRef.current += textDelta;
            if (rafIdRef.current === null) {
              rafIdRef.current = requestAnimationFrame(() => {
                const pending = pendingTranscriptRef.current;
                pendingTranscriptRef.current = "";
                rafIdRef.current = null;
                if (pending) {
                  setMessages((prev) =>
                    prev.map((m) =>
                      m.id === currentAssistantIdRef.current
                        ? { ...m, text: m.text + pending }
                        : m
                    )
                  );
                }
              });
            }
          }
          break;
        }

        case "response.audio.done": {
          const durMs = metrics.audioBytesReceived / (24000 * 2) * 1000;
          console.log(`[OVA] Audio done: ${metrics.audioDeltas} chunks, ${metrics.audioBytesReceived} bytes (${durMs.toFixed(0)}ms)`);
          setIsAssistantSpeaking(false);
          break;
        }

        case "response.done": {
          const resp = data.response as Record<string, unknown> | undefined;
          console.log(`[OVA] Response done: status=${resp?.status}`);
          const id = currentAssistantIdRef.current;
          if (id) {
            // Flush any pending transcript before marking final
            // (race condition: transcript_delta may arrive just before response.done)
            if (pendingTranscriptRef.current) {
              const pending = pendingTranscriptRef.current;
              pendingTranscriptRef.current = "";
              if (rafIdRef.current !== null) {
                cancelAnimationFrame(rafIdRef.current);
                rafIdRef.current = null;
              }
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === id ? { ...m, text: m.text + pending } : m
                )
              );
            }
            updateMessage(id, { isFinal: true });
            currentAssistantIdRef.current = null;
          }
          // Reset audio metrics for next response
          metrics.audioDeltas = 0;
          metrics.audioBytesReceived = 0;
          break;
        }

        case "macaw.metrics": {
          const m = data.metrics as Record<string, unknown>;
          if (m) {
            const entry: ResponseMetrics = {
              response_id: (data.response_id as string) || "",
              timestamp: Date.now(),
              // Session
              turn: m.turn as number | undefined,
              session_duration_s: m.session_duration_s as number | undefined,
              barge_in_count: m.barge_in_count as number | undefined,
              // VAD / Speech
              speech_ms: m.speech_ms as number | undefined,
              speech_rms: m.speech_rms as number | undefined,
              vad_silence_wait_ms: m.vad_silence_wait_ms as number | undefined,
              smart_turn_inference_ms: m.smart_turn_inference_ms as number | undefined,
              smart_turn_waits: m.smart_turn_waits as number | undefined,
              // ASR
              asr_ms: m.asr_ms as number | undefined,
              asr_mode: m.asr_mode as string | undefined,
              asr_partial_count: m.asr_partial_count as number | undefined,
              input_chars: m.input_chars as number | undefined,
              // LLM
              llm_ttft_ms: m.llm_ttft_ms as number | undefined,
              llm_total_ms: m.llm_total_ms as number | undefined,
              llm_first_sentence_ms: m.llm_first_sentence_ms as number | undefined,
              // TTS
              tts_synth_ms: m.tts_synth_ms as number | undefined,
              tts_wait_ms: m.tts_wait_ms as number | undefined,
              tts_first_chunk_ms: m.tts_first_chunk_ms as number | undefined,
              // Encode + Send
              encode_send_ms: m.encode_send_ms as number | undefined,
              // Pipeline
              e2e_ms: m.e2e_ms as number | undefined,
              pipeline_first_audio_ms: m.pipeline_first_audio_ms as number | undefined,
              sentences: m.sentences as number | undefined,
              audio_chunks: m.audio_chunks as number | undefined,
              output_chars: m.output_chars as number | undefined,
              // Tools
              tool_rounds: m.tool_rounds as number | undefined,
              tools_used: m.tools_used as string[] | undefined,
              tool_timings: m.tool_timings as ToolTiming[] | undefined,
              // Backpressure
              backpressure_level: m.backpressure_level as number | undefined,
              events_dropped: m.events_dropped as number | undefined,
              // SLO
              slo_met: m.slo_met as boolean | undefined,
              // Total
              total_ms: m.total_ms as number | undefined,
            };
            console.log("[OVA] Response metrics:", entry);
            setResponseMetrics((prev) => [...prev, entry]);
          }
          break;
        }

        case "error": {
          console.error("[OVA] Server error:", data);
          break;
        }
      }
    },
    [send, addMessage, updateMessage]
  );

  const connect = useCallback(async () => {
    if (wsRef.current) return;

    setStatus("connecting");
    setMessages([]);
    setResponseMetrics([]);
    metricsRef.current = new SessionMetrics();

    console.log(`[OVA] Connecting to ${WS_URL}...`);

    try {
      // Start audio first (needs user gesture)
      const capture = new AudioCapture();
      const playback = new AudioPlayback();
      console.log("[OVA] Starting audio playback context...");
      await playback.start();
      console.log("[OVA] Audio playback ready");

      // Open WebSocket
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = async () => {
        console.log("[OVA] WebSocket connected, starting mic capture...");
        // Start mic after WS is open
        await capture.start((pcm16) => {
          if (ws.readyState === WebSocket.OPEN) {
            metricsRef.current.logChunk(pcm16.byteLength);
            ws.send(
              JSON.stringify({
                type: "input_audio_buffer.append",
                audio: arrayBufferToBase64(pcm16),
              })
            );
          }
        });
        captureRef.current = capture;
        playbackRef.current = playback;
        console.log("[OVA] Mic capture started, sending audio chunks");
      };

      ws.onmessage = (e) => {
        try {
          const data = JSON.parse(e.data);
          handleServerEvent(data);
        } catch {
          console.error("[OVA] Failed to parse:", e.data);
        }
      };

      ws.onclose = (e) => {
        console.log(`[OVA] WebSocket closed: code=${e.code}, reason=${e.reason}`);
        capture.stop();
        playback.stop();
        captureRef.current = null;
        playbackRef.current = null;
        wsRef.current = null;
        setStatus("disconnected");
        setIsUserSpeaking(false);
        setIsAssistantSpeaking(false);
      };

      ws.onerror = (err) => {
        console.error("[OVA] WebSocket error:", err);
      };
    } catch (err) {
      console.error("[OVA] Connection error:", err);
      setStatus("disconnected");
    }
  }, [handleServerEvent]);

  const disconnect = useCallback(() => {
    console.log("[OVA] Disconnecting...");
    if (rafIdRef.current !== null) {
      cancelAnimationFrame(rafIdRef.current);
      rafIdRef.current = null;
    }
    pendingTranscriptRef.current = "";
    captureRef.current?.stop();
    playbackRef.current?.stop();
    captureRef.current = null;
    playbackRef.current = null;

    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }

    setStatus("disconnected");
    setIsUserSpeaking(false);
    setIsAssistantSpeaking(false);
  }, []);

  return {
    status,
    messages,
    responseMetrics,
    isUserSpeaking,
    isAssistantSpeaking,
    connect,
    disconnect,
  };
}
