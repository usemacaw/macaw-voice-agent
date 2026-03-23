/** Conversation message shown in UI. */
export interface Message {
  id: string;
  role: "user" | "assistant";
  text: string;
  isFinal: boolean;
}

/** Connection status. */
export type SessionStatus = "disconnected" | "connecting" | "connected";

/** Per-tool execution timing. */
export interface ToolTiming {
  name: string;
  exec_ms: number;
  ok: boolean;
}

/** Per-response observability metrics from the server. */
export interface ResponseMetrics {
  response_id: string;
  timestamp: number;
  // Session
  turn?: number;
  session_duration_s?: number;
  barge_in_count?: number;
  // VAD / Speech
  speech_ms?: number;
  speech_rms?: number;
  vad_silence_wait_ms?: number;
  smart_turn_inference_ms?: number;
  smart_turn_waits?: number;
  // ASR
  asr_ms?: number;
  asr_mode?: string;
  asr_partial_count?: number;
  input_chars?: number;
  // LLM
  llm_ttft_ms?: number;
  llm_total_ms?: number;
  llm_first_sentence_ms?: number;
  // TTS
  tts_synth_ms?: number;
  tts_wait_ms?: number;
  tts_first_chunk_ms?: number;
  // Encode + Send
  encode_send_ms?: number;
  // Pipeline
  e2e_ms?: number;
  pipeline_first_audio_ms?: number;
  sentences?: number;
  audio_chunks?: number;
  output_chars?: number;
  // Tools
  tool_rounds?: number;
  tools_used?: string[];
  tool_timings?: ToolTiming[];
  // Backpressure
  backpressure_level?: number;
  events_dropped?: number;
  // SLO
  slo_met?: boolean;
  // Total
  total_ms?: number;
}
