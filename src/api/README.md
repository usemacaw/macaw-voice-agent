# OpenVoiceAPI

Drop-in replacement for the [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) via WebSocket, with pluggable ASR, LLM, and TTS providers.

## Architecture

```
WebSocket Client (browser, SDK, etc.)
       │
       │  JSON events + base64 audio (PCM16 24kHz)
       ▼
┌──────────────────────────────────────────────────┐
│            OpenVoiceAPI  (ws://host:8765)         │
│                                                    │
│  ┌────────────┐  ┌──────────┐  ┌───────────────┐ │
│  │  Protocol   │  │  Server   │  │    Audio      │ │
│  │  (events,   │  │ (session, │  │ (codec, VAD,  │ │
│  │   models)   │  │  ws)      │  │  resampling)  │ │
│  └──────┬──────┘  └─────┬────┘  └───────┬───────┘ │
│         │               │               │          │
│         └───────┬───────┘               │          │
│                 ▼                        │          │
│  ┌──────────────────────────────┐       │          │
│  │     RealtimeSession          │◄──────┘          │
│  │  (state machine per conn)    │                  │
│  └──────────┬───────────────────┘                  │
│             │                                       │
│             ▼                                       │
│  ┌──────────────────────────────┐                  │
│  │     SentencePipeline         │                  │
│  │  LLM → sentence_queue → TTS │                  │
│  │       → audio_queue → yield  │                  │
│  └──────┬──────────┬────────────┘                  │
│         │          │                                │
│    ┌────▼──┐  ┌────▼──┐                            │
│    │  ASR  │  │  TTS  │   ← Pluggable providers    │
│    │  LLM  │  │       │                            │
│    └───────┘  └───────┘                            │
└──────────────────────────────────────────────────┘
         │               │
    gRPC (optional)  gRPC (optional)
         │               │
    ┌────▼──┐       ┌────▼──┐
    │  STT  │       │  TTS  │
    │ Server│       │ Server│
    │ :50060│       │ :50070│
    └───────┘       └───────┘
```

## Quick Start

```bash
# 1. Install
pip install -e ".[dev,vad]"

# 2. Configure
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY, adjust providers

# 3. Run
python main.py

# 4. Connect
# Any OpenAI Realtime API client → ws://localhost:8765/v1/realtime
```

## Providers

### ASR (Speech-to-Text)

| Provider | `ASR_PROVIDER=` | Requirements | Streaming |
|----------|-----------------|--------------|-----------|
| Remote gRPC | `remote` | Running stt-server | Yes |
| Faster-Whisper | `whisper` | `pip install .[whisper]` + GPU | No |
| Qwen3-ASR | `qwen` | qwen-asr + GPU | No |

### LLM

| Provider | `LLM_PROVIDER=` | Requirements |
|----------|-----------------|--------------|
| Anthropic Claude | `anthropic` | `ANTHROPIC_API_KEY` |
| OpenAI GPT | `openai` | `OPENAI_API_KEY` |

### TTS (Text-to-Speech)

| Provider | `TTS_PROVIDER=` | Requirements | Streaming |
|----------|-----------------|--------------|-----------|
| Remote gRPC | `remote` | Running tts-server | Yes |
| Kokoro-ONNX | `kokoro` | `pip install .[kokoro]` | Yes |
| Edge TTS | `edge` | `pip install .[edge]` + ffmpeg | No |
| Qwen3-TTS | `qwen` | qwen-tts + GPU | No |

## Configuration

All configuration is via environment variables. See [`.env.example`](.env.example) for the full list.

**Key variables:**

| Variable | Default | Description |
|----------|---------|-------------|
| `WS_PORT` | `8765` | WebSocket server port |
| `REALTIME_API_KEY` | *(empty)* | Bearer token for auth. Empty = no auth |
| `ASR_PROVIDER` | `remote` | ASR backend |
| `TTS_PROVIDER` | `remote` | TTS backend |
| `LLM_PROVIDER` | `anthropic` | LLM backend |
| `LLM_MODEL` | `claude-sonnet-4-20250514` | Model ID |
| `LLM_SYSTEM_PROMPT` | *(generic)* | System prompt for the assistant |
| `VAD_SILENCE_MS` | `200` | Silence (ms) to detect end of speech |
| `PIPELINE_TTS_PREFETCH_SIZE` | `4` | Sentences to prefetch TTS |

## Protocol Compatibility

Implements the [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) WebSocket protocol:

**Supported client events:**
- `session.update` — configure modalities, system prompt, tools, audio format, VAD
- `input_audio_buffer.append` / `.commit` / `.clear` — send audio
- `conversation.item.create` / `.truncate` / `.delete` — manage conversation
- `response.create` / `.cancel` — trigger/cancel assistant response

**Supported audio formats:** `pcm16` (24kHz), `g711_ulaw`, `g711_alaw`

**Server-side VAD:** Silero VAD (ML-based) with configurable threshold, silence detection, prefix padding, and barge-in support.

## Endpoints

| Path | Method | Description |
|------|--------|-------------|
| `/v1/realtime` | WebSocket | Realtime API (main endpoint) |
| `/health` | GET | Health check: `{ status, active_sessions, max_connections }` |

## Tests

```bash
pip install -e ".[dev,vad]"
pytest -v
```

109 tests covering: protocol events, audio codec, session lifecycle, VAD, pipeline, provider failures, backpressure, SDK compatibility.

## Project Structure

```
open-voice-api/
├── main.py                    # Entry point
├── config.py                  # Env var loading + validation
├── audio/
│   ├── codec.py               # PCM, G.711, base64, resampling (24kHz↔8kHz)
│   ├── vad.py                 # Silero VAD (8kHz, 32ms chunks)
│   └── utils.py               # PCM↔float32, scipy resampling
├── protocol/
│   ├── events.py              # 28+ event builders (OpenAI format)
│   ├── models.py              # SessionConfig, ConversationItem, TurnDetection
│   └── event_emitter.py       # Serial WebSocket sender with backpressure
├── providers/
│   ├── registry.py            # Generic ProviderRegistry[T]
│   ├── asr.py / asr_*.py      # ASR providers (remote, whisper, qwen)
│   ├── llm.py / llm_*.py      # LLM providers (anthropic, openai)
│   └── tts.py / tts_*.py      # TTS providers (remote, kokoro, edge, qwen)
├── pipeline/
│   ├── sentence_pipeline.py   # LLM→TTS streaming pipeline
│   └── conversation.py        # ConversationItem→OpenAI message format
├── server/
│   ├── ws_server.py           # WebSocket server, auth, /health
│   └── session.py             # RealtimeSession state machine
└── tests/                     # 109 tests
```
