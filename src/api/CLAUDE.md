# OpenVoiceAPI

Drop-in replacement for the OpenAI Realtime API. WebSocket server that orchestrates ASR→LLM→TTS pipeline with pluggable providers.

## Commands

```bash
# Run server
python main.py

# Run tests
pip install -e ".[dev,vad]" && pytest -v

# Run single test file
pytest tests/test_session.py -v
```

## Architecture

- **Entry point:** `main.py` — creates providers (factory), connects with retry, starts WebSocket server
- **Session per connection:** `server/session.py` — `RealtimeSession` is the state machine. Handles audio buffer, conversation items, response lifecycle. Delegates audio input to `server/audio_input.py` (VAD, ASR, RMS, barge-in) and response execution to `server/response_runner.py` (LLM→tools→TTS→audio)
- **Config:** `config.py` — env vars loaded into dicts + frozen dataclass policies (`VadPolicy`, `LLMPolicy`, `PipelinePolicy`, etc.) for type-safe access
- **Pipeline:** `pipeline/sentence_pipeline.py` — LLM streams sentences → queue → TTS worker synthesizes in parallel → audio chunks yielded to consumer
- **Providers:** ABC + `ProviderRegistry[T]` with lazy auto-discovery. Registration happens at module bottom (`register_*_provider`)
- **Protocol:** Events match OpenAI Realtime API 1:1. `protocol/events.py` builds JSON dicts, `event_emitter.py` sends them with backpressure (SlowClientError for structural events)
- **Audio:** All internal processing at 8kHz PCM16. API speaks 24kHz. Codec handles resampling + G.711 + base64

## Key Design Decisions

- **LLM is stateless:** receives full message list every call. Conversation history owned by RealtimeSession
- **Sentence-level pipelining:** LLM→TTS runs in parallel (producer-consumer with asyncio queues). Prefetches 4 TTS sentences ahead
- **Server-side VAD:** Silero ML model (ONNX), not webrtcvad. 32ms chunks at 8kHz. Callbacks for speech_started/stopped
- **SlowClientError:** Droppable events (audio.delta) silently dropped on timeout. Structural events (response.done) raise error → connection terminated
- **Handler dispatch:** String-based `_HANDLER_MAP` with `getattr(self, name)` to avoid circular refs in __init__

## Server-Side Tool Calling

- **Tool Registry:** `tools/registry.py` — register async handlers with schemas, timeout, filler phrases
- **Mock Handlers:** `tools/handlers.py` — demo tools (lookup_customer, get_balance, get_card_info, etc.)
- **Enable:** `TOOL_ENABLE_MOCK=true` in `.env` to activate mock tools
- **Flow:** LLM emits tool_call → filler TTS sent → tool executed server-side → result added to conversation → LLM re-called with result
- **Dual mode:** If ToolRegistry has handlers → server-side execution. If no handlers → falls back to client-side (OpenAI Realtime API compat)
- **Config:** `TOOL_TIMEOUT` (10s), `TOOL_MAX_ROUNDS` (5), `TOOL_DEFAULT_FILLER`

## Config

All env vars. See `.env.example`. Key: `ASR_PROVIDER`, `TTS_PROVIDER`, `LLM_PROVIDER`, `ANTHROPIC_API_KEY`, `LLM_SYSTEM_PROMPT`, `TOOL_ENABLE_MOCK`.

## Audio Formats

- API (client↔server): PCM16 24kHz, G.711 mu-law 8kHz, G.711 A-law 8kHz
- Internal (providers): PCM16 8kHz mono
- Constants: `API_SAMPLE_RATE=24000`, `INTERNAL_SAMPLE_RATE=8000`, `SAMPLE_WIDTH=2`

## Test Patterns

- `FakeWebSocket` with `asyncio.Event` drain (no sleeps)
- `FakeASR/FakeLLM/FakeTTS` with configurable behavior
- `delay_after` parameter for timing-sensitive tests (e.g., cancellation)
- All tests deterministic, no flaky sleeps
