# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed
- Replaced untyped `dict[str, object]` metrics with `ResponseMetrics` dataclass (`protocol/metrics.py`) — eliminates silent typos, provides IDE autocomplete, and documents all metric fields with types
- Replaced `**kwargs` forwarding in `ResponseRunner._run_audio_response()` with explicit typed signature — typos now caught at type-check time instead of runtime
- Introduced `ToolResponseContext` parameter object to replace 17 keyword-only params in `run_with_tools()` — improves testability and readability
- Fixed encapsulation violation: `session.py` no longer accesses private `_asr_stream_id`; uses new `has_active_asr_stream` property instead
- Moved inline `from config import STREAMING` in hot path (`audio_input.py:feed_asr_chunk`) to module-level import
- LLM timing metrics now accessed via `get_last_timing()` snapshot instead of mutable class attributes — eliminates race condition when multiple sessions share a single LLMProvider instance
- Unified `response.done` lifecycle ownership: `ResponseRunner` is now the single emitter of `response.done` and `macaw.metrics` for ALL response paths (text, audio, tools) — eliminates ambiguity about which module owns the lifecycle event

### Added
- 221 new tests covering previously untested modules: `test_response_strategy.py` (strategy selection, SLO targets, tool merging), `test_context_builder.py` (message building, windowing, tool pairs), `test_admission.py` (semaphore concurrency, timeout, metrics), `test_filler.py` (dynamic phrases, TTS integration, Portuguese accents), `test_response_metrics.py` (typed metrics, to_dict filtering, merge_prior), `test_tool_engine.py` (tool parsing, server-side execution, filler coordination)

### Changed (prior)
- Decomposed `ResponseRunner` god object (809→374 LOC): extracted `server/response/audio_response.py`, `server/response/text_response.py`, `server/response/tool_response.py` — each response path is now an independent, testable module
- Extracted `HealthTracker` class into `common/grpc_server.py`: eliminates copy-pasted health tracking logic duplicated across STT, TTS, and LLM servers (~60 LOC x 3)
- Extracted `configure_logging()` into `common/grpc_server.py`: eliminates duplicated `_configure_logging()` in each microservice
- Migrated legacy config dicts (`AUDIO_CONFIG`, `STT_CONFIG`, etc.) to auto-generated via `dataclasses.asdict()` — eliminates manual dict/dataclass parallel maintenance

### Added
- 35 new tests: `test_event_emitter.py` (13 tests — cancellation fencing, backpressure escalation, throttling), `test_health_tracker.py` (7 tests — error counting, degradation, recovery), `test_conversation_store.py` (15 tests — CRUD, eviction, memory indexing)

### Changed (prior)
- Unified config API: all consumers migrated from legacy dict access (`CONFIG["key"]`) to frozen dataclass attributes (`POLICY.key`). Dict intermediaries made private (`_*_CONFIG`). New policies: `ASRPolicy`, `TTSPolicy`, `LogPolicy`
- Decomposed `ResponseRunner._run_with_tools()` god method (240→145 lines): extracted `_stream_llm_with_inline_tts()`, `_capture_llm_timing()`, `_emit_fallback_response()`, eliminated 65-line inline closure
- Removed dead provider references from registries: `whisper_stt`, `qwen_tts`, `kokoro_tts`, `faster_tts` entries pointing to deleted files
- Refactored all providers to use gRPC remote only — API server is now a pure orchestrator
- LLM provider: new gRPC microservice (`src/llm/server.py`, port 50080) with proto `theo.llm.LLMService`
- LLM client: `llm_remote.py` replaces `llm_anthropic.py`, `llm_openai.py`, `llm_vllm.py`
- Config defaults: `LLM_PROVIDER=remote`, `LLM_REMOTE_TARGET=localhost:50080`

### Removed
- Direct API providers: `llm_anthropic.py`, `llm_openai.py`, `llm_vllm.py`, `_openai_stream.py`
- Local ASR providers: `asr_whisper.py`, `asr_qwen.py`
- Local TTS providers: `tts_kokoro.py`, `tts_qwen.py`, `tts_edge.py`
- API keys from API server config (`ANTHROPIC_API_KEY`, `OPENAI_API_KEY` — moved to LLM server)

### Added
- `audio/text_cleaning.py`: single source of truth for emoji stripping and Qwen3 thinking block removal — eliminates DRY violation between `response_runner.py` and `sentence_pipeline.py`
- `IncrementalSplitter` in `pipeline/sentence_splitter.py`: reusable incremental sentence splitting for inline TTS during tool-calling — eliminates duplicated regex logic in `response_runner._run_with_tools()`
- `server/audio_emitter.py`: unified TTS→encode→emit class with `emit_from_text()` and `emit_from_queue()` — eliminates 3 duplicated audio emission paths in `response_runner.py`
- 29 new unit tests: `test_text_cleaning.py` (11), `test_incremental_splitter.py` (12), `test_audio_emitter.py` (6) covering text cleaning, sentence splitting, and audio emission
- Cancellation fences in `EventEmitter`: events from stale `response_id` are silently dropped, preventing ghost audio/text during barge-in
- Turn detection metrics: `vad_silence_wait_ms`, `smart_turn_inference_ms`, `smart_turn_waits` emitted in `macaw.metrics` per response
- TTS first-class metrics: `tts_first_chunk_ms`, `tts_queue_max_depth`, `tts_calls` emitted in `macaw.metrics` per response
- Per-provider admission control: `MAX_CONCURRENT_ASR`, `MAX_CONCURRENT_TTS`, `MAX_CONCURRENT_LLM` semaphores prevent resource exhaustion under load (`providers/admission.py`)
- Progressive backpressure in `EventEmitter`: 4-level degradation (normal → throttle transcripts → drop transcripts → terminate) instead of binary drop/kill
- Backpressure metrics: `backpressure_level` and `events_dropped` emitted in `macaw.metrics` per response
- System metrics endpoint `/metrics`: CPU, memory, event loop lag, active sessions, cancel rate, admission control stats (`server/system_metrics.py`)
- 11 cancellation race condition tests: validates fence mechanism, rapid barge-in, double cancel, cancel-then-new-response (`tests/test_cancellation_races.py`)
- Token-budget windowing with pinning: replaces fixed 8-item window with `LLM_MAX_CONTEXT_TOKENS` budget, pins first user message, preserves tool call pairs (`pipeline/conversation.py`)
- Latency SLO tracking: `slo_met`, `slo_target_ms` emitted in `macaw.metrics`; warns on breach. Configurable via `SLO_FIRST_AUDIO_MS` (1500ms) and `SLO_FIRST_AUDIO_TOOL_MS` (5000ms)
- 9 token-budget windowing tests: validates budget limits, fallback cap, first-user pinning, tool pair cohesion (`tests/test_conversation_window.py`)
- Server-side tool calling: `ToolRegistry` with async execution, timeout, and error handling (`tools/registry.py`)
- Mock tool handlers for demo: `lookup_customer`, `get_account_balance`, `get_card_info`, `get_recent_transactions`, `create_support_ticket`, `transfer_to_human` (`tools/handlers.py`)
- Filler TTS during tool execution: synthesizes phrases like "Vou consultar seu saldo." while tools run
- Tool execution loop: LLM can call tools multiple times in sequence with results fed back automatically
- Tool configuration via env vars: `TOOL_ENABLE_MOCK`, `TOOL_TIMEOUT`, `TOOL_MAX_ROUNDS`, `TOOL_DEFAULT_FILLER`
- Unified web search tool via DuckDuckGo — zero API keys: single `web_search` with `type` param for general/news (`tools/web_search.py`)
- Conversation memory with `recall_memory` tool: LLM can search older conversation history via keyword matching (`tools/recall_memory.py`)
- Streaming TTS pipeline for final tool round: `SentencePipeline` (LLM→TTS pipelined) so audio starts as first sentence completes instead of waiting for all text
- `_strip_think()` safety filter: strips `<think>...</think>` reasoning blocks from LLM output for thinking models like Qwen3
- Windowed conversation history: LLM receives only the last 8 items instead of full history, reducing context size and latency
- Orphan tool call cleanup: incomplete function_call/function_call_output pairs are removed from LLM context
- `ToolRegistry.fork()` method for per-session tool customization without affecting shared registry
- Tool error short-circuit: if a tool returns an error, the next LLM call is made without tools to force a text response
- 65 unit tests for tool registry, mock handlers, web search, recall memory, and conversation windowing
- Self-contained STT/TTS microservices: migrated providers, common modules, and gRPC stubs from theo-ai-voice-agent
- STT providers: Whisper (Faster-Whisper/CTranslate2), Qwen3-ASR (batch + streaming), Mock
- TTS providers: Kokoro-ONNX (streaming), FasterQwen3TTS (CUDA graphs), Qwen3-TTS, Mock
- Common modules: `src/common/` with config, audio_utils, executor shared by STT/TTS
- GPU Dockerfiles: `Dockerfile.whisper` (STT) and `Dockerfile.kokoro-gpu` (TTS) for Vast.ai deployment
- Docker Compose for single-GPU host running both STT + TTS containers (`docker-compose.gpu.yml`)
- Session idle timeout: connections are automatically closed after 10 minutes of inactivity
- Rate limiting: max 200 events/second per WebSocket connection with sliding window
- Conversation memory management: FIFO eviction when items exceed 200 limit
- Input validation for conversation items: blocks system role injection, enforces size limits
- Session config validation: size limits for instructions (50k chars), tools (128 max), and tool schemas (200k chars total)
- WebSocket origin validation via `WS_ALLOWED_ORIGINS` env var (CORS-like protection)
- Health endpoint now reports per-provider status and returns 503 when degraded
- Function calling / tool use support for both OpenAI and Anthropic LLM providers
- `LLMStreamEvent` dataclass and `generate_stream_with_tools()` method on LLM provider ABC
- `LOG_LEVEL` env var validation against valid Python logging levels
- AudioContext `resume()` on start for Chrome/Safari autoplay policy compliance
- `recv()` method on FakeWebSocket test helper for idle timeout compatibility

### Changed
- Extracted `AudioInputHandler` from `RealtimeSession`: VAD, ASR, RMS, and speech detection logic moved to `server/audio_input.py` (~270 lines), reducing `session.py` from 769 to ~500 lines
- Added frozen dataclass config policies (`VadPolicy`, `PipelinePolicy`, `LLMPolicy`, `ConnectionPolicy`, `ToolPolicy`) to `config.py` for type-safe access with IDE autocomplete; legacy dicts kept for backward compatibility
- Migrated 4 priority consumers (`session.py`, `audio_input.py`, `response_runner.py`, `sentence_pipeline.py`) from dict-based config to policy dataclasses
- Extracted `ResponseRunner` from `RealtimeSession`: response execution (LLM→tools→TTS→audio events) moved to `server/response_runner.py`, reducing `session.py` from 1,775 to 769 lines
- Fixed pre-existing test bugs: added missing `warmup()` to FakeASR/FakeTTS and `last_ttft_ms`/`last_stream_total_ms` to FakeLLM
- Switched default LLM to Qwen3-8B-AWQ: better tool calling accuracy at lower latency than Qwen2.5-14B
- vLLM provider disables thinking mode (`enable_thinking: False`) to prevent `<think>` blocks in voice responses
- Filler audio no longer stored in conversation history: prevents LLM from mimicking filler phrases instead of calling tools
- Filler phrase for web_search changed from "Vou pesquisar isso para voce." to "Um momento." to avoid model confusion
- DuckDuckGo search uses `region="br-pt"` for Portuguese language results
- System prompt updated to explicitly forbid markdown formatting and reinforce tool calling for factual queries
- System prompt updated to prevent false promises: agent now refuses actions it cannot perform instead of promising and never delivering
- `_run_response_with_tools` in session.py rewritten: supports both server-side execution (with ToolRegistry) and client-side execution (OpenAI Realtime API compat)
- `LLM_MAX_TOKENS` increased from 30 to 150 to support tool calling responses
- `WebSocketServer` and `RealtimeSession` accept optional `tool_registry` parameter
- STT/TTS servers now use `python -m stt.server` / `python -m tts.server` module execution
- STT/TTS Dockerfiles use `src/` as build context with `PYTHONPATH=/app` (no more multi-repo PYTHONPATH)
- Auto-discovery module paths updated to new package structure (`stt.providers.*`, `tts.providers.*`)
- Session loop now uses explicit `recv()` with timeout instead of `async for` iteration
- VAD frame buffer uses in-place `del` instead of slice-and-copy for better memory efficiency
- RMS computation uses numpy vectorized operations instead of Python loop
- Capture processor (AudioWorklet) uses Float32Array ring buffer instead of plain Array (O(1) vs O(n) operations, prevents stack overflow on spread operator)
- Provider disconnect in `main.py` now wraps each provider individually to prevent cascade failures
- Pipeline cleanup in `sentence_pipeline.py` replaced `asyncio.shield` with direct cancel-then-await
- `_start_response` cancels previous response task outside `_state_lock` to prevent deadlock
- TTS remote provider now raises `RuntimeError` on gRPC/timeout errors instead of silently ending stream

### Fixed
- Tool response audio: eliminated redundant 3rd LLM call — `_emit_tool_response_audio` now synthesizes collected text directly through TTS instead of calling LLM again via SentencePipeline
- `_handle_response_cancel` could raise `NameError` due to undefined `task` variable outside lock block
- ASR stream ID cleanup in `_transcribe_audio` now always runs under `_state_lock`
- `_handle_conversation_item_retrieve` and `_handle_conversation_item_truncate` now use `_state_lock` for thread-safe reads
- `_handle_response_cancel` and `_handle_output_audio_buffer_clear` now use `_state_lock`
- Audio buffer overflow now returns error event instead of silently dropping data
