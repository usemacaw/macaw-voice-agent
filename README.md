# Macaw Voice Agent

Drop-in replacement for the [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime). WebSocket server that orchestrates a full **ASR → LLM → TTS** pipeline with pluggable providers, server-side tool calling, and real-time observability.

```
Browser (mic) ──PCM16 24kHz──▶ WebSocket :8765
                                    │
                              ┌─────▼──────┐
                              │  Silero VAD │  (speech detection)
                              └─────┬──────┘
                                    │
                              ┌─────▼──────┐
                              │   ASR       │  Whisper / Qwen3-ASR (gRPC :50060)
                              └─────┬──────┘
                                    │
                              ┌─────▼──────┐
                              │   LLM       │  Claude / GPT / vLLM (HTTP :8000)
                              │  + Tools    │  web_search, recall_memory, custom
                              └─────┬──────┘
                                    │
                              ┌─────▼──────┐
                              │   TTS       │  Kokoro / Qwen3-TTS (gRPC :50070)
                              └─────┬──────┘
                                    │
Browser (speaker) ◀──PCM16 24kHz───┘
```

## Features

- **OpenAI Realtime API compatible** — works with any client that speaks the protocol
- **Pluggable providers** — swap ASR, LLM, and TTS independently via env vars
- **Sentence-level streaming** — LLM streams text → split into sentences → TTS synthesizes in parallel → audio plays as first sentence is ready
- **Server-side tool calling** — async execution with filler audio ("Vou pesquisar..."), timeout, and multi-round support
- **Server-side VAD** — Silero ML model (ONNX), 32ms frames, configurable aggressiveness
- **Real-time observability** — per-response metrics (ASR, LLM TTFT, TTS, E2E latency, tool timing) via `macaw.metrics` WebSocket event
- **Production-ready** — rate limiting, idle timeout, origin validation, health checks, graceful shutdown
- **GPU microservices** — containerized STT/TTS with CUDA support for Vast.ai or any GPU host

## Quick Start

### 1. API Server

```bash
cd src/api
pip install -e ".[dev,vad]"
cp .env.example .env
# Set your ANTHROPIC_API_KEY (or OPENAI_API_KEY)
python main.py
```

### 2. Web Client

```bash
cd src/web
npm install
npm run dev
# Open http://localhost:5173
```

### 3. With STT/TTS Microservices (Docker + GPU)

```bash
cd src
docker compose -f docker-compose.gpu.yml up -d
```

## Project Structure

```
src/
├── api/                    # Main WebSocket server (Python)
│   ├── main.py             # Entry point
│   ├── config.py           # Env var loading + validation
│   ├── server/
│   │   ├── ws_server.py    # WebSocket lifecycle, health endpoint
│   │   └── session.py      # RealtimeSession state machine (per-connection)
│   ├── pipeline/
│   │   ├── sentence_pipeline.py  # LLM→TTS streaming pipeline
│   │   └── conversation.py       # Message history windowing
│   ├── providers/
│   │   ├── asr.py          # ASR ABC + factory
│   │   ├── llm.py          # LLM ABC + sentence splitting
│   │   ├── tts.py          # TTS ABC + factory
│   │   ├── asr_remote.py   # gRPC ASR client
│   │   ├── llm_anthropic.py    # Claude provider
│   │   ├── llm_openai.py       # GPT provider
│   │   ├── llm_vllm.py         # vLLM provider
│   │   ├── tts_remote.py       # gRPC TTS client
│   │   └── registry.py         # Generic ProviderRegistry[T]
│   ├── tools/
│   │   ├── registry.py     # ToolRegistry (register, execute, fork)
│   │   ├── handlers.py     # Mock tools (banking demo)
│   │   ├── web_search.py   # DuckDuckGo (zero API keys)
│   │   └── recall_memory.py    # Conversation memory search
│   ├── audio/
│   │   ├── codec.py        # PCM16/G.711, resampling, base64
│   │   ├── vad.py          # Silero VAD processor
│   │   └── smart_turn.py   # Turn detection helpers
│   ├── protocol/
│   │   ├── events.py       # OpenAI Realtime API event builders
│   │   ├── models.py       # SessionConfig, TurnDetection
│   │   └── event_emitter.py    # Backpressure-aware WebSocket sender
│   └── tests/              # 109+ tests (pytest)
│
├── stt/                    # STT microservice (gRPC :50060)
│   ├── server.py           # gRPC server (Transcribe, TranscribeStream)
│   └── providers/          # Whisper, Qwen3-ASR, Mock
│
├── tts/                    # TTS microservice (gRPC :50070)
│   ├── server.py           # gRPC server (Synthesize, SynthesizeStream)
│   └── providers/          # Kokoro-ONNX, FasterQwen3TTS, Qwen3-TTS, Mock
│
├── llm/                    # vLLM container config
│   └── Dockerfile.qwen     # Qwen2.5-7B-AWQ with tool calling
│
├── web/                    # React + TypeScript web client (Vite)
│   └── src/
│       ├── App.tsx          # Main UI (Orb + Transcript + Metrics)
│       ├── hooks/useRealtimeSession.ts  # WebSocket + audio lifecycle
│       ├── audio/capture.ts    # AudioWorklet mic capture (24kHz)
│       ├── audio/playback.ts   # AudioWorklet speaker output
│       └── components/
│           ├── Orb.tsx             # Animated orb (idle/listening/speaking/thinking)
│           ├── TranscriptPanel.tsx  # Chat history slide-in
│           └── MetricsPanel.tsx    # Real-time observability dashboard
│
├── common/                 # Shared modules (config, audio_utils, executor)
├── shared/                 # Generated gRPC stubs
└── docker-compose.gpu.yml  # Multi-container orchestration
```

## Configuration

All configuration via environment variables. See [`src/api/.env.example`](src/api/.env.example) for the full reference.

### Key Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WS_PORT` | `8765` | WebSocket server port |
| `REALTIME_API_KEY` | *(empty)* | API key for auth (empty = no auth) |
| `ASR_PROVIDER` | `remote` | `remote` (gRPC), `whisper`, `qwen` |
| `TTS_PROVIDER` | `remote` | `remote` (gRPC), `kokoro`, `edge` |
| `LLM_PROVIDER` | `anthropic` | `anthropic`, `openai`, `vllm` |
| `ANTHROPIC_API_KEY` | — | Required if `LLM_PROVIDER=anthropic` |
| `LLM_MODEL` | `claude-sonnet-4-20250514` | Model ID for LLM provider |
| `LLM_SYSTEM_PROMPT` | *(built-in)* | System prompt for the voice agent |
| `VAD_AGGRESSIVENESS` | `2` | 0-3, higher = more aggressive silence detection |
| `VAD_SILENCE_MS` | `200` | Milliseconds of silence to trigger end-of-speech |
| `TOOL_ENABLE_MOCK` | `false` | Enable mock banking tools (demo) |
| `TOOL_ENABLE_WEB_SEARCH` | `false` | Enable DuckDuckGo web search tool |
| `LOG_LEVEL` | `INFO` | Python logging level |

### Provider Combinations

| Setup | ASR | LLM | TTS | GPU Required |
|-------|-----|-----|-----|-------------|
| **Cloud-only** | remote (gRPC) | `anthropic` or `openai` | remote (gRPC) | No (API server) |
| **Self-hosted** | remote (Whisper) | `vllm` (Qwen) | remote (Kokoro) | Yes (all services) |
| **Hybrid** | remote (Whisper) | `anthropic` | remote (Kokoro) | Yes (STT/TTS only) |

## Architecture

### Key Design Decisions

- **LLM is stateless** — full conversation history passed every call. No server-side state in the LLM. History managed by `RealtimeSession`
- **Sentence-level pipelining** — LLM text → sentence split → TTS queue (prefetch 4 ahead) → audio chunks. First audio plays before LLM finishes generating
- **Server-side VAD** — Silero ML model runs on the server, not in the browser. 32ms frames at 8kHz internal rate
- **Dual audio rates** — API speaks 24kHz PCM16 (client standard), internal processing at 8kHz (telephony standard). Codec layer handles resampling
- **Provider auto-discovery** — `ProviderRegistry[T]` with lazy module import. Add a new provider file → it's automatically available
- **Windowed context** — only last 8 conversation items sent to LLM. Reduces latency and token usage

### Tool Calling Flow

```
User speaks → ASR → LLM (with tool schemas)
                      │
                      ├─ tool_call: web_search("dólar cotação")
                      │     │
                      │     ├─ Filler TTS: "Vou pesquisar sobre dólar, aguarde."
                      │     └─ Execute tool → result JSON
                      │
                      ├─ LLM re-called with tool result
                      │
                      └─ Final text → SentencePipeline → TTS → Audio
```

### Observability Metrics

Every response emits a `macaw.metrics` WebSocket event with:

| Metric | Description |
|--------|-------------|
| `asr_ms` | Speech-to-text latency |
| `llm_ttft_ms` | LLM time to first token |
| `llm_total_ms` | Total LLM generation time |
| `llm_first_sentence_ms` | Time until first complete sentence |
| `tts_synth_ms` | TTS synthesis time |
| `pipeline_first_audio_ms` | End-to-end: speech stopped → first audio chunk |
| `e2e_ms` | Full round-trip: speech stopped → response audio playing |
| `tool_timings[]` | Per-tool execution time and status |

## Testing

```bash
cd src/api
pip install -e ".[dev,vad]"

# All tests
pytest -v

# Single file
pytest tests/test_session.py -v

# With coverage
pytest --cov=. --cov-report=term-missing
```

Test suite includes 109+ tests covering:
- Session lifecycle, VAD, barge-in, response cancellation
- Audio codec (PCM16, G.711, resampling)
- Protocol event builders
- Conversation history windowing
- Tool registry, mock handlers, timeout, error handling
- Web search and conversation memory
- OpenAI SDK compatibility

## Health Check

```bash
curl http://localhost:8765/health
```

Returns provider status and overall health:
```json
{
  "status": "ok",
  "providers": {
    "asr": "connected",
    "tts": "connected",
    "llm": "connected"
  }
}
```

Returns HTTP 503 when any provider is degraded.

## License

Proprietary. All rights reserved.
