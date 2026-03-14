# Arquitetura Técnica — Macaw Voice Agent

> Agente de voz voice-to-voice em tempo real.
> Drop-in replacement para a OpenAI Realtime API com providers plugáveis de ASR, LLM e TTS.

---

## Visão Geral do Sistema

```
┌──────────────────────────────────────────────────────────────────────────┐
│                          BROWSER (React + Vite)                         │
│                                                                         │
│  ┌─────────┐   ┌─────────────────┐   ┌────────────┐   ┌─────────────┐ │
│  │  Orb UI  │   │ TranscriptPanel │   │MetricsPanel│   │AudioWorklet │ │
│  └─────────┘   └─────────────────┘   └────────────┘   └──────┬──────┘ │
│                                                               │         │
│                useRealtimeSession (WebSocket hook)             │         │
└───────────────────────────────────┬───────────────────────────┘         │
                                    │ WebSocket (PCM16 24kHz base64)      │
                                    ▼                                     │
┌──────────────────────────────────────────────────────────────────────────┐
│                        API SERVER (Python asyncio)                      │
│                                                                         │
│  ┌────────────────┐                                                     │
│  │ WebSocketServer │ auth, health, rate limit, connection pooling        │
│  └───────┬────────┘                                                     │
│          │ per-connection                                                │
│  ┌───────▼────────────────────────────────────────────────────────────┐ │
│  │                     RealtimeSession                                │ │
│  │                                                                    │ │
│  │  ┌────────────────┐  ┌───────────────────┐  ┌───────────────┐  │ │
│  │  │AudioInputHandler│  │ResponseOrchestrator│  │ConversationStore│  │ │
│  │  │ VAD + ASR       │  │ (ResponseRunner)  │  │ items + memory │  │ │
│  │  └───────┬────────┘  └──────┬────────────┘  └───────────────┘  │ │
│  │          │                   │                                    │ │
│  │  ┌───────▼────────┐  ┌──────▼────────────────────────────────┐  │ │
│  │  │  Silero VAD    │  │  Intelligence Layer                    │  │ │
│  │  │  Smart Turn    │  │  ContextBuilder + ToolEngine + Strategy│  │ │
│  │  └────────────────┘  └──────┬────────────────────────────────┘  │ │
│  │                              │                                    │ │
│  │                       ┌──────▼───────────────────────────────┐   │ │
│  │                       │         SentencePipeline              │   │ │
│  │                       │  LLM → sentence_queue → TTS → audio  │   │ │
│  │                       └──────────────────────────────────────┘   │ │
│  └────────────────────────────────────────────────────────────────────┘ │
│                                                                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐                    │
│  │ ASRProvider  │  │ LLMProvider  │  │ TTSProvider  │   Provider ABCs   │
│  │   (ABC)      │  │   (ABC)      │  │   (ABC)      │                   │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘                    │
│         │                │                 │                            │
└─────────┼────────────────┼─────────────────┼────────────────────────────┘
          │                │                 │
          ▼                ▼                 ▼
   ┌──────────┐     ┌───────────┐     ┌──────────┐
   │ STT gRPC │     │ Claude    │     │ TTS gRPC │
   │ :50060   │     │ GPT      │     │ :50070   │
   │          │     │ vLLM     │     │          │
   │ Whisper  │     └───────────┘     │ Kokoro   │
   │ Qwen ASR │                       │ Qwen TTS │
   └──────────┘                       └──────────┘
```

---

## Pipeline Voice-to-Voice

O fluxo completo de um turno de voz, do microfone ao alto-falante:

```
Mic (24kHz PCM16)
 │
 ▼ base64 via WebSocket
 │
 ▼ decode_audio_from_client()
 │  PCM16 24kHz → resample → PCM16 8kHz
 │
 ▼ AudioInputHandler.feed_audio()
 │  Buffer chunks de 32ms (256 samples @ 8kHz)
 │
 ▼ Silero VAD (ONNX, single-thread)
 │  Probabilidade de speech por chunk
 │  Onset: ≥3 chunks consecutivos acima do threshold
 │  Offset: silence_duration_ms de silêncio
 │
 ▼ Smart Turn (ONNX, opcional)
 │  Análise prosódica do áudio acumulado
 │  Decide se o turno está completo ou se o usuário pausou para pensar
 │
 ▼ ASR Provider (gRPC → STT Server)
 │  Whisper large-v3-turbo ou Qwen3-ASR
 │  PCM 8kHz → texto transcrito
 │
 ▼ RealtimeSession._on_user_speech_complete()
 │  Cria ConversationItem(role="user", transcript=...)
 │  Inicia ResponseRunner
 │
 ▼ ResponseRunner.run()
 │  ┌─────────────────────────────────────────────────┐
 │  │ COM TOOLS:                                       │
 │  │  for round in range(max_rounds):                 │
 │  │    LLM.generate_stream_with_tools()              │
 │  │    ├─ text_delta → acumula texto                 │
 │  │    └─ tool_call → acumula tool calls             │
 │  │                                                   │
 │  │    Se tool calls detectados:                      │
 │  │      1. Envia filler TTS ("Vou pesquisar...")     │
 │  │      2. Executa tools server-side com timeout     │
 │  │      3. Adiciona resultados ao histórico          │
 │  │      4. Reconstrói messages e repete o loop       │
 │  │                                                   │
 │  │    Se apenas texto:                               │
 │  │      Sintetiza via TTS e emite áudio              │
 │  │                                                   │
 │  │ SEM TOOLS:                                        │
 │  │  SentencePipeline (producer-consumer)             │
 │  │    LLM stream → split sentences → TTS parallel    │
 │  └─────────────────────────────────────────────────┘
 │
 ▼ SentencePipeline.process_streaming()
 │  ┌────────────────────────────────────────────┐
 │  │ Producer (Task 1):                          │
 │  │   LLM.generate_stream() → buffer texto     │
 │  │   → split em frases (eager first: 20 chars) │
 │  │   → enqueue no sentence_queue (max 6)       │
 │  │                                              │
 │  │ TTS Worker (Task 2):                         │
 │  │   dequeue sentence                           │
 │  │   → TTS.synthesize_stream()                  │
 │  │   → enqueue no audio_queue (max 4-200)       │
 │  │                                              │
 │  │ Consumer (main loop):                        │
 │  │   dequeue audio_chunk                        │
 │  │   → chunk em pedaços de 100ms                │
 │  │   → yield (sentence, chunk, is_new)          │
 │  └────────────────────────────────────────────┘
 │
 ▼ encode_audio_for_client()
 │  PCM16 8kHz → resample → PCM16 24kHz → base64
 │
 ▼ EventEmitter.emit(response.audio.delta)
 │  JSON via WebSocket
 │
 ▼ AudioWorklet (browser)
 │  Decode base64 → PCM16 → Speaker
```

### Latências Típicas por Estágio

| Estágio | Métrica | Valor típico |
|---------|---------|-------------|
| VAD chunk processing | — | < 1ms |
| Smart Turn inference | — | ~12ms |
| ASR (Whisper batch) | `asr_ms` | 200-500ms |
| LLM time-to-first-token | `llm_ttft_ms` | 300-800ms |
| LLM total streaming | `llm_total_ms` | 1000-3000ms |
| First sentence ready | `llm_first_sentence_ms` | 400-1200ms |
| TTS synthesis | `tts_synth_ms` | 100-500ms |
| First audio to client | `pipeline_first_audio_ms` | 600-1500ms |
| E2E (speech stop → first audio) | `e2e_ms` | 800-2000ms |

---

## Estrutura de Diretórios

```
src/
├── api/                           # Servidor WebSocket principal (Python 3.11+)
│   ├── main.py                    # Entry point: cria providers, conecta, inicia server
│   ├── config.py                  # Env vars → frozen dataclasses + dicts legados
│   ├── pyproject.toml             # Deps, pytest config
│   │
│   ├── server/
│   │   ├── ws_server.py           # WebSocket lifecycle, health, auth, CORS
│   │   ├── session.py             # RealtimeSession — fachada fina de protocolo
│   │   ├── conversation_store.py  # ConversationStore — items + memory + lock
│   │   ├── response_runner.py     # ResponseOrchestrator — delega para strategies
│   │   ├── audio_input.py         # VAD + ASR input processing
│   │   └── filler.py              # Filler phrases contextuais durante tool calling
│   │
│   ├── intelligence/              # Camada de decisão (extraída do ResponseRunner)
│   │   ├── context_builder.py     # Autoridade única sobre contexto LLM
│   │   ├── tool_engine.py         # Execução de tools server-side
│   │   └── response_strategy.py   # Seleção de estratégia (ResponseMode/ResponsePlan)
│   │
│   ├── turns/                     # Modelo explícito de turno de voz
│   │   └── turn_pipeline.py       # VoiceTurn, TurnStage, TurnMetrics
│   │
│   ├── pipeline/
│   │   ├── sentence_pipeline.py   # Producer-consumer LLM→TTS streaming
│   │   ├── sentence_splitter.py   # Split texto em frases para TTS
│   │   └── conversation.py        # Histórico → messages OpenAI format, windowing
│   │
│   ├── providers/
│   │   ├── registry.py            # ProviderRegistry[T] — auto-discovery genérico
│   │   ├── capabilities.py        # TTSCapabilities, ASRCapabilities, LLMCapabilities
│   │   ├── asr.py                 # ASRProvider ABC
│   │   ├── llm.py                 # LLMProvider ABC + LLMStreamEvent
│   │   ├── tts.py                 # TTSProvider ABC
│   │   ├── _openai_stream.py      # Parser compartilhado OpenAI streaming (tool calls)
│   │   ├── asr_remote.py          # gRPC client → STT server
│   │   ├── tts_remote.py          # gRPC client → TTS server
│   │   ├── llm_anthropic.py       # Claude (converte OpenAI→Anthropic format)
│   │   ├── llm_openai.py          # GPT
│   │   └── llm_vllm.py            # vLLM (OpenAI-compatible, thinking desabilitado)
│   │
│   ├── tools/
│   │   ├── registry.py            # ToolRegistry — register, execute, fork, timeout
│   │   ├── handlers.py            # Mock tools bancários (demo)
│   │   ├── web_search.py          # DuckDuckGo (zero API keys)
│   │   └── recall_memory.py       # Busca keyword no histórico da conversa
│   │
│   ├── audio/
│   │   ├── codec.py               # PCM16/G.711, resampling 24kHz↔8kHz, base64
│   │   ├── vad.py                 # Silero VAD (ONNX) + Smart Turn integration
│   │   └── smart_turn.py          # Pipecat Smart Turn v3.2 (end-of-turn semântico)
│   │
│   ├── protocol/
│   │   ├── contract.py            # Invariantes formalizados (delivery, ordering, limites)
│   │   ├── events.py              # Builders dos eventos OpenAI Realtime API + macaw.metrics
│   │   ├── models.py              # SessionConfig, TurnDetection, ConversationItem
│   │   └── event_emitter.py       # WebSocket sender com backpressure
│   │
│   └── tests/                     # 16 arquivos, 174+ testes (pytest-asyncio)
│
├── stt/                           # Microserviço STT (gRPC :50060)
│   ├── server.py                  # STTServicer + STTServer (usa GrpcMicroservice)
│   ├── providers/
│   │   ├── base.py                # STTProvider ABC + factory + MockSTT
│   │   ├── whisper_stt.py         # Faster-Whisper (ONNX/CUDA)
│   │   └── qwen_stt.py            # Qwen3-ASR
│   └── tests/
│
├── tts/                           # Microserviço TTS (gRPC :50070)
│   ├── server.py                  # TTSServicer + TTSServer (usa GrpcMicroservice)
│   ├── providers/
│   │   ├── base.py                # TTSProvider ABC + factory + MockTTS
│   │   ├── kokoro_tts.py          # Kokoro-ONNX (~82M params)
│   │   └── qwen_tts.py            # Qwen3-TTS
│   └── tests/
│
├── common/                        # Módulos compartilhados entre API e microserviços
│   ├── config.py                  # AudioConfig, STTConfig, TTSConfig (frozen dataclasses)
│   ├── audio_utils.py             # PCM↔float32, resample (scipy anti-aliasing)
│   └── grpc_server.py             # GrpcMicroservice — health, reflection, shutdown
│
├── shared/                        # Stubs gRPC gerados
│   └── grpc_gen/
│       ├── stt_service_pb2.py
│       ├── stt_service_pb2_grpc.py
│       ├── tts_service_pb2.py
│       └── tts_service_pb2_grpc.py
│
├── web/                           # Frontend React + TypeScript + Vite + Tailwind v3
│   └── src/
│       ├── App.tsx                # UI principal (Orb + Transcript + Metrics)
│       ├── hooks/useRealtimeSession.ts
│       ├── audio/capture.ts       # AudioWorklet mic (24kHz PCM16)
│       ├── audio/playback.ts      # AudioWorklet speaker
│       └── components/
│           ├── Orb.tsx            # Orb animado (idle, listening, thinking, speaking)
│           ├── TranscriptPanel.tsx
│           └── MetricsPanel.tsx
│
├── llm/                           # vLLM container
│   └── Dockerfile.qwen
│
└── docker-compose.gpu.yml         # STT (Whisper) + TTS (Kokoro) com GPU
```

---

## Componentes em Detalhe

### 1. WebSocket Server

**Arquivo:** `src/api/server/ws_server.py`

Aceita conexões WebSocket em `/v1/realtime`, implementando o protocolo OpenAI Realtime API.

**Pré-flight (antes do upgrade WebSocket):**

| Check | Detalhes |
|-------|----------|
| Health | `GET /health` → JSON com status dos providers (200/503) |
| Path | Deve ser `/v1/realtime` (configurável) |
| Origin | Validado contra `WS_ALLOWED_ORIGINS` (se definido) |
| Connections | Rejeita se `active_sessions >= max_connections` |
| Auth | Bearer token ou `?api_key=` query param, comparação constant-time |

**Health check response:**
```json
{
  "status": "ok",
  "active_sessions": 2,
  "max_connections": 10,
  "providers": {
    "asr": true,
    "asr_healthy": true,
    "llm": true,
    "tts": true,
    "tts_healthy": true
  }
}
```

A verificação de saúde dos providers usa o método `health_check()` do ABC, sem acessar internals dos providers.

---

### 2. RealtimeSession (Protocol Facade)

**Arquivo:** `src/api/server/session.py`

Fachada fina de protocolo — uma instância por conexão WebSocket. Delega para:
- `ConversationStore`: items, memory, locking
- `AudioInputHandler`: VAD, ASR, speech detection
- `ResponseRunner`: LLM, tools, TTS, audio streaming

**Estado interno:**

| Campo | Tipo | Descrição |
|-------|------|-----------|
| `_store` | `ConversationStore` | Items + memory + lock (extraído) |
| `_audio_buffer` | `bytearray` | Buffer para modo manual commit (max 10 min) |
| `_config` | `SessionConfig` | Configuração mutável (atualizada via `session.update`) |
| `_response_task` | `asyncio.Task` | Task ativa de geração de resposta |
| `_tool_registry` | `ToolRegistry` | Fork per-session (isolamento) |

**ConversationStore** (`src/api/server/conversation_store.py`):

Extraído da session para isolar estado da conversa do protocolo.

| Método | Descrição |
|--------|-----------|
| `append(item)` | Append + feed memory (sob lock) |
| `delete(item_id)` | Remove por ID (sob lock) |
| `find(item_id)` | Busca por ID |
| `last_id()` | ID do último item |
| `items` | `deque[ConversationItem]` com maxlen=200 |
| `memory` | `ConversationMemory` para recall_memory |
| `lock` | `asyncio.Lock` para serialização |

**Dispatch de eventos (string-based):**

```python
_HANDLER_MAP = {
    "session.update":                "_handle_session_update",
    "input_audio_buffer.append":     "_handle_input_audio_buffer_append",
    "input_audio_buffer.commit":     "_handle_input_audio_buffer_commit",
    "input_audio_buffer.clear":      "_handle_input_audio_buffer_clear",
    "conversation.item.create":      "_handle_conversation_item_create",
    "conversation.item.delete":      "_handle_conversation_item_delete",
    "conversation.item.retrieve":    "_handle_conversation_item_retrieve",
    "conversation.item.truncate":    "_handle_conversation_item_truncate",
    "response.create":               "_handle_response_create",
    "response.cancel":               "_handle_response_cancel",
    "output_audio_buffer.clear":     "_handle_output_audio_buffer_clear",
}
```

Dispatch usa `getattr(self, handler_name)(data)` — evita imports circulares e permite extensão sem modificar o dispatch.

**Proteções:**

| Proteção | Valor | Comportamento |
|----------|-------|---------------|
| Rate limit | 200 events/s | Emite `error(rate_limit_exceeded)`, ignora evento |
| Idle timeout | 600s (10 min) | Desconecta sessão |
| Buffer overflow | 9.6 MB | Emite `error(buffer_too_large)`, trunca |
| Items overflow | 200 items | Auto-evicção FIFO via `deque(maxlen=200)` |

---

### 3. Audio Input (VAD + ASR)

**Arquivo:** `src/api/server/audio_input.py`

Responsabilidade única: detectar fala via VAD e transcrever via ASR.

**Callbacks (desacoplado da session via dataclass):**

```python
@dataclass
class AudioInputCallbacks:
    cancel_active_response: Callable    # Barge-in
    append_user_item_and_respond: Callable  # Fala completa → resposta
    emit: Callable                      # Enviar eventos WebSocket
```

**VAD — 2 estágios:**

```
                    ┌──────────────────────────────┐
                    │     Silero VAD (ONNX)        │
                    │  32ms chunks, 8kHz, threshold │
                    │  Onset: 3 chunks consecutivos │
                    │  Offset: silence_duration_ms  │
                    └──────────────┬───────────────┘
                                   │ (silêncio detectado)
                                   ▼
                    ┌──────────────────────────────┐
                    │   Smart Turn v3.2 (ONNX)     │
                    │  Análise prosódica            │
                    │  Max 4 tentativas             │
                    │  threshold=0.5                │
                    │  Se incompleto: espera mais   │
                    │  Se completo: emite callback  │
                    └──────────────────────────────┘
```

**Fallback de transcrição:**

1. **Streaming** (se `ASR_REMOTE_STREAMING=true`): chunks enviados em tempo real via gRPC bidirectional
2. **Batch** (padrão): áudio acumulado enviado inteiro após VAD detectar fim do turno
3. **Batch fallback**: se streaming falhar, usa batch como fallback

---

### 4. Response Orchestrator

**Arquivo:** `src/api/server/response_runner.py` (classe `ResponseRunner`, conceitualmente um orchestrator)

Criado por resposta (sem estado entre respostas). Delega decisões para módulos especializados:
- `ResponsePlan` (de `intelligence/response_strategy.py`) — decide **como** responder
- `ContextBuilder` (de `intelligence/context_builder.py`) — monta **contexto** do LLM
- `ToolExecutionEngine` (de `intelligence/tool_engine.py`) — executa **tools** server-side

**Strategy Selection (`intelligence/response_strategy.py`):**

```python
ResponseMode = Enum:
    TEXT_ONLY           # Texto sem TTS
    AUDIO_STREAMING     # LLM→SentencePipeline→TTS
    TOOL_CALLING        # Tools + texto
    TOOL_CALLING_AUDIO  # Tools + TTS

ResponsePlan = dataclass:
    mode: ResponseMode
    has_audio: bool
    has_tools: bool
    server_side_tools: bool
    tools: list[dict]       # Merged schemas
    max_rounds: int

# Toda a lógica de decisão concentrada em:
plan = select_strategy(config, tool_registry)
```

**ContextBuilder (`intelligence/context_builder.py`):**

Autoridade única sobre construção de contexto LLM:

```python
builder = ContextBuilder(config)
messages, system, temperature, max_tokens = builder.build_for_response(items, has_tools)
messages = builder.rebuild_after_tool_round(items)
tools = builder.merge_tool_schemas(config_tools, server_schemas)
```

**ToolExecutionEngine (`intelligence/tool_engine.py`):**

Execução de tools extraída do runner:

```
ToolExecutionEngine
├── execute_server_side(tool_calls, has_audio, ...)
│   ├── Send filler audio (via filler.py)
│   ├── Execute cada tool com timeout
│   ├── Criar function_call + function_call_output items
│   └── Retornar ToolRoundResult(all_tools_ok, output_index_delta)
└── emit_tool_calls_for_client(tool_calls, ...)  # Fallback client-side
```

**Filler phrases (PT-BR, em `server/filler.py`):**

| Tool | Exemplos |
|------|----------|
| `web_search` | "Vou pesquisar sobre {query}...", "Deixa eu buscar sobre {query}..." |
| `recall_memory` | "Deixa eu verificar...", "Vou checar, aguarde." |
| Genérico | "Um momento, por favor.", "Aguarde um instante." |

---

### 5. Sentence Pipeline (Producer-Consumer)

**Arquivo:** `src/api/pipeline/sentence_pipeline.py`

Minimiza latência do primeiro áudio com pipelining de 3 estágios:

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                  │
│  ┌──────────────┐    sentence_queue    ┌────────────┐           │
│  │   Producer    │ ──────(max 6)────── │ TTS Worker │           │
│  │              │                      │            │           │
│  │ LLM stream   │                      │ synthesize │           │
│  │ → buffer     │                      │ _stream()  │           │
│  │ → split      │    audio_queue       │            │           │
│  │   sentences  │ ◄────(max 4-200)──── │            │           │
│  └──────────────┘                      └────────────┘           │
│                                                                  │
│  Consumer (main loop):                                           │
│    dequeue → chunk 100ms → yield → emit response.audio.delta    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**Sentence splitting (eager-first):**

O primeiro "sentence" é emitido agressivamente em breaks de cláusula (`,;:—`) após 20 chars para minimizar o tempo até o primeiro áudio. Frases subsequentes usam pontuação final (`.!?`).

Frases longas (>150 chars) são splitadas recursivamente em break points naturais.

**Queue sizing:**

| Queue | Tamanho padrão | Ajuste |
|-------|---------------|--------|
| `sentence_queue` | 6 | `PIPELINE_SENTENCE_QUEUE_SIZE` |
| `audio_queue` (batch TTS) | 4 | `PIPELINE_TTS_PREFETCH_SIZE` |
| `audio_queue` (streaming TTS) | 200 | prefetch × 50 |

---

### 6. Provider System (Auto-Discovery)

**Arquivo:** `src/api/providers/registry.py`

Pattern: Generic Registry com lazy import.

```python
# Definição do registry (no módulo ABC)
_registry = ProviderRegistry[ASRProvider]("ASR", {
    "remote":  "providers.asr_remote",
    "whisper": "providers.asr_whisper",
    "qwen":    "providers.asr_qwen",
})

# Registro (side-effect no import do módulo concreto)
register_asr_provider("remote", RemoteASR)

# Criação (lazy import + instanciação)
asr = create_asr_provider("remote")
# → import providers.asr_remote (se não carregado)
# → RemoteASR()
```

**Providers disponíveis:**

| Tipo | Nome | Classe | Protocolo | Modelo |
|------|------|--------|-----------|--------|
| ASR | `remote` | `RemoteASR` | gRPC → STT server | Whisper/Qwen |
| ASR | `whisper` | `WhisperASR` | Local | Faster-Whisper |
| ASR | `qwen` | `QwenASR` | Local | Qwen3-ASR |
| LLM | `anthropic` | `AnthropicLLM` | HTTP | Claude |
| LLM | `openai` | `OpenAILLM` | HTTP | GPT |
| LLM | `vllm` | `VLLMProvider` | HTTP (OpenAI-compat) | Qwen2.5/outros |
| TTS | `remote` | `RemoteTTS` | gRPC → TTS server | Kokoro/Qwen |
| TTS | `kokoro` | `KokoroTTS` | Local (ONNX) | Kokoro 82M |
| TTS | `edge` | `EdgeTTS` | HTTP | Microsoft Edge TTS |
| TTS | `qwen` | `QwenTTS` | Local | Qwen3-TTS |

**ABCs e contratos:**

```
ASRProvider
├── transcribe(audio: bytes) → str              # Obrigatório
├── start_stream(stream_id) → None              # Opcional (streaming)
├── feed_chunk(audio, stream_id) → str          # Opcional
├── finish_stream(stream_id) → str              # Opcional
├── supports_streaming → bool                   # Default: False
├── connect() / disconnect() / warmup()         # Lifecycle
└── health_check() → bool                       # Default: True

LLMProvider
├── generate_stream(messages, system, ...) → AsyncGen[str]        # Obrigatório
├── generate_stream_with_tools(...) → AsyncGen[LLMStreamEvent]    # Default: wraps generate_stream
├── last_ttft_ms: float                         # Set por implementação
├── last_stream_total_ms: float                 # Set por implementação
└── connect() / disconnect()                    # Lifecycle

TTSProvider
├── synthesize(text: str) → bytes               # Obrigatório
├── synthesize_stream(text: str) → AsyncGen[bytes]  # Default: chunked batch
├── supports_streaming → bool                   # Default: False
├── connect() / disconnect() / warmup()         # Lifecycle
└── health_check() → bool                       # Default: True
```

**OpenAI/vLLM tool stream — parser compartilhado:**

`providers/_openai_stream.py` contém `parse_openai_tool_stream()` usado por ambos `OpenAILLM` e `VLLMProvider` para parsing de tool call deltas. Evita duplicação do protocolo OpenAI streaming (~40 linhas).

**Provider Capabilities (`providers/capabilities.py`):**

Substitui flags soltas (`supports_streaming`, `hasattr(...)`) por objetos tipados:

```python
@dataclass(frozen=True)
class TTSCapabilities:
    streaming: bool = False

@dataclass(frozen=True)
class ASRCapabilities:
    streaming: bool = False

@dataclass(frozen=True)
class LLMCapabilities:
    tool_calling: bool = True
    thinking_mode: bool = False

# Uso:
caps = TTSCapabilities.from_provider(tts)
if caps.streaming: ...
```

---

### 7. Tool System

**Arquivo:** `src/api/tools/registry.py`

**Conceitos:**

- **ToolDef**: definição (name, handler, schema, filler_phrase)
- **ToolRegistry**: container de ToolDefs com execute + fork
- **fork()**: cópia shallow para customização per-session
- **Dual mode**: se registry tem handlers → server-side. Senão → client-side (compat OpenAI)

**Execução:**

```
ToolRegistry.execute(name, arguments_json)
├── Parse JSON args
├── asyncio.wait_for(handler(**args), timeout=10s)
├── On success: return JSON result
├── On timeout: return {"error": "timeout", "message": "..."}
└── On exception: return {"error": "execution_failed", "message": "..."}
```

**Tools disponíveis:**

| Tool | Módulo | Descrição |
|------|--------|-----------|
| `lookup_customer` | `handlers.py` | Busca cliente por telefone/CPF (mock) |
| `get_account_balance` | `handlers.py` | Consulta saldo (mock) |
| `get_card_info` | `handlers.py` | Informações do cartão (mock) |
| `get_recent_transactions` | `handlers.py` | Últimas transações (mock) |
| `create_support_ticket` | `handlers.py` | Criar ticket de suporte (mock) |
| `transfer_to_human` | `handlers.py` | Transferir para atendente (mock) |
| `web_search` | `web_search.py` | Busca DuckDuckGo + fetch de conteúdo |
| `recall_memory` | `recall_memory.py` | Busca keyword no histórico da conversa |

---

### 8. Protocol Layer

**Modelos de dados (`protocol/models.py`):**

```python
@dataclass
class SessionConfig:
    modalities: list[str]           # ["text", "audio"]
    instructions: str               # System prompt (max 50K chars)
    voice: str                      # "alloy", etc.
    input_audio_format: str         # "pcm16" | "g711_ulaw" | "g711_alaw"
    output_audio_format: str
    turn_detection: TurnDetection   # VAD config
    tools: list[dict]               # OpenAI function schemas (max 128)
    temperature: float              # 0.0–2.0
    max_response_output_tokens: int | str  # int ou "inf"

@dataclass
class TurnDetection:
    type: str = "server_vad"
    threshold: float = 0.5
    prefix_padding_ms: int = 300
    silence_duration_ms: int = 200
    create_response: bool = True
    interrupt_response: bool = True     # Barge-in

@dataclass
class ConversationItem:
    id: str
    type: str       # "message" | "function_call" | "function_call_output"
    role: str       # "user" | "assistant" | "system" | None
    content: list[ContentPart]
    status: str     # "completed" | "in_progress" | "incomplete"
    call_id: str    # Para function_call items
    name: str       # Tool name
    arguments: str  # JSON string
    output: str     # Tool result
```

**Eventos emitidos (`protocol/events.py`):**

| Categoria | Eventos |
|-----------|---------|
| Session | `session.created`, `session.updated` |
| Conversation | `conversation.created`, `conversation.item.created`, `.deleted`, `.retrieved` |
| Audio Input | `input_audio_buffer.speech_started`, `.speech_stopped`, `.committed`, `.cleared` |
| Transcription | `conversation.item.input_audio_transcription.completed` |
| Response | `response.created`, `response.done` |
| Response Output | `response.output_item.added`, `response.output_item.done` |
| Response Content | `response.content_part.added`, `response.content_part.done` |
| Audio Output | `response.audio.delta`, `response.audio.done` |
| Text Output | `response.text.delta`, `response.text.done` |
| Transcript | `response.audio_transcript.delta`, `response.audio_transcript.done` |
| Function Call | `response.function_call_arguments.done` |
| Error | `error` |
| Observability | `macaw.metrics` |

**Protocol Contract (`protocol/contract.py`):**

Invariantes formalizados do protocolo — anteriormente implícitos no código:

```
Protocol Ordering Guarantees:

1. SESSION: session.created always precedes session.updated
2. RESPONSE: response.created → output_item.added → content → output_item.done → response.done
3. AUDIO: audio.delta events are ordered but droppable; audio.done is structural
4. TOOLS: function_call_arguments.done precedes execution; rounds complete before final response
5. INPUT: speech_started precedes speech_stopped; barge-in cancels before starting new
6. METRICS: macaw.metrics always emitted AFTER response.done
```

Constantes centralizadas: `MAX_EVENTS_PER_SECOND`, `SESSION_IDLE_TIMEOUT_S`, `MAX_CONVERSATION_ITEMS`, `DROPPABLE_EVENTS`.

**EventEmitter — backpressure (`protocol/event_emitter.py`):**

```
Evento recebido para envio
   │
   ▼ asyncio.wait_for(ws.send(json), timeout=5s)
   │
   ├─ Sucesso: evento entregue
   │
   └─ Timeout:
      ├─ Se droppable (audio.delta, text.delta, transcript.delta):
      │  → log warning, drop silenciosamente
      │
      └─ Se structural (response.done, session.created, error):
         → raise SlowClientError → sessão terminada
```

---

### 9. Audio Codec

**Arquivo:** `src/api/audio/codec.py`

| Constante | Valor | Uso |
|-----------|-------|-----|
| `API_SAMPLE_RATE` | 24000 Hz | Client ↔ WebSocket |
| `INTERNAL_SAMPLE_RATE` | 8000 Hz | Providers, gRPC, VAD |
| `SAMPLE_WIDTH` | 2 bytes | PCM 16-bit |

**G.711 — lookup tables pré-computadas:**

4 tabelas construídas no import (numpy vectorizado, ~5ms):
- `_ULAW_DECODE_TABLE`: 256 entries (byte → PCM16)
- `_ULAW_ENCODE_TABLE`: 65536 entries (PCM16 → byte)
- `_ALAW_DECODE_TABLE`: 256 entries
- `_ALAW_ENCODE_TABLE`: 65536 entries

Performance: O(1) per sample, zero alocação por chamada.

**Resampling:** usa `scipy.signal.resample_poly` (filtro polifásico com anti-aliasing) via `common/audio_utils.py`.

---

### 10. gRPC Microservices

**Infraestrutura compartilhada (`common/grpc_server.py`):**

```python
class GrpcMicroservice:
    """Base para STT/TTS servers."""

    # gRPC options padrão:
    # - keepalive: ping 30s, timeout 10s
    # - max message: 10MB send/receive
    # - permit pings without calls

    async def start(add_servicers, service_names, provider)
    async def stop(grace=5.0)     # NOT_SERVING → wait → disconnect
    async def run_until_signal()  # Block até SIGINT/SIGTERM
```

**Health check dinâmico:**

Ambos os servicers (STT e TTS) rastreiam erros consecutivos:

```python
_consecutive_errors = 0
_MAX_ERRORS_BEFORE_UNHEALTHY = 5

# No try/except de cada RPC:
# Sucesso → _record_success() → reset counter, set SERVING
# Erro → _record_error() → increment counter, if ≥5: set NOT_SERVING
```

**STT Server (:50060):**

| RPC | Input | Output | Modo |
|-----|-------|--------|------|
| `Transcribe` | `audio_data` (bytes), `language` | `text`, `processing_ms` | Batch |
| `TranscribeStream` | stream `AudioChunk` | stream `TranscribeResult` | Bidirectional |

**TTS Server (:50070):**

| RPC | Input | Output | Modo |
|-----|-------|--------|------|
| `Synthesize` | `text`, `output_config` | `audio_data`, `duration_ms`, `processing_ms` | Batch |
| `SynthesizeStream` | `text` | stream `AudioChunk` | Server streaming |

---

### 11. Observabilidade

Cada resposta emite `macaw.metrics` via WebSocket com timing detalhado:

```json
{
  "type": "macaw.metrics",
  "response_id": "resp_abc123",
  "metrics": {
    "turn": 3,
    "session_duration_s": 45.2,
    "barge_in_count": 1,

    "asr_ms": 320.5,
    "speech_ms": 1500.0,
    "input_chars": 42,

    "llm_ttft_ms": 450.2,
    "llm_total_ms": 1200.8,
    "llm_first_sentence_ms": 680.3,

    "tts_synth_ms": 280.1,
    "tts_wait_ms": 15.3,

    "pipeline_first_audio_ms": 950.6,
    "pipeline_total_ms": 2100.4,
    "e2e_ms": 1270.8,
    "total_ms": 2100.4,

    "sentences": 3,
    "audio_chunks": 24,
    "output_chars": 156,

    "tools_used": ["web_search"],
    "tool_rounds": 2,
    "tool_timings": [
      {"name": "web_search", "exec_ms": 1200.5, "ok": true}
    ]
  }
}
```

O frontend exibe essas métricas no `MetricsPanel` com barras visuais e sumário (avg/min/max por sessão).

O `TurnMetrics` em `turns/turn_pipeline.py` modela as mesmas métricas como dataclass tipada com `to_dict()` para emissão.

---

### 12. Turn Pipeline (Modelo Explícito de Turno)

**Arquivo:** `src/api/turns/turn_pipeline.py`

Modela um turno de voz como conceito de primeira classe:

```python
class TurnStage(Enum):
    CREATED          # Turno criado
    INPUT_DETECTED   # Speech detectado pelo VAD
    TRANSCRIBED      # ASR concluído
    CONTEXT_BUILT    # Contexto LLM montado
    LLM_STREAMING    # LLM gerando resposta
    TOOL_EXECUTING   # Tools sendo executados
    TTS_SYNTHESIZING # TTS sintetizando áudio
    DELIVERING       # Áudio sendo enviado ao cliente
    COMPLETED        # Turno finalizado
    CANCELLED        # Cancelado (barge-in)
    FAILED           # Erro durante execução

class VoiceTurn:
    """Um turno = user fala → sistema entende → sistema responde → user ouve."""
    turn_id: str
    metrics: TurnMetrics

    def advance(stage)          # Progressão do turno
    def record_input(metrics)   # Métricas do ASR
    def record_e2e(timestamp)   # Latência E2E
    def finalize() -> dict      # Métricas finais
```

Esse modelo torna o fluxo end-to-end rastreável e mensurável por estágio.

---

### 13. Conversation History e Windowing

**Arquivo:** `src/api/pipeline/conversation.py` (low-level conversion)
**Arquivo:** `src/api/intelligence/context_builder.py` (high-level policy)

O histórico é gerenciado pelo `ConversationStore` como uma `deque` de `ConversationItem`s. O `ContextBuilder` é a autoridade única sobre como converter esses items em messages para o LLM.

**Estratégia de windowing:**

| Cenário | Window | Motivo |
|---------|--------|--------|
| Com tools | 8 items | Reduz tokens, mantém contexto de tool call/result |
| Sem tools | Todos os items | Contexto completo para respostas naturais |

**Proteção contra orphans:**

```
Antes do windowing:
  [..., assistant(tool_call=A), tool(call_id=A, result=R), user("ok"), ...]

Se o window cortar no meio:
  1. Orphan tool result (call_id=A sem tool_call anterior) → puxa a call
  2. Orphan tool call (tool_call=A sem result depois) → puxa o result
  3. _clean_orphan_tool_messages() → remove pares incompletos restantes
```

---

### 14. Frontend

**Stack:** React 18 + TypeScript + Vite + Tailwind CSS v3

**Componentes:**

| Componente | Responsabilidade |
|------------|-----------------|
| `App.tsx` | Layout principal, state management |
| `Orb.tsx` | Orb animado com 4 estados visuais (idle/listening/thinking/speaking) |
| `TranscriptPanel.tsx` | Chat bubbles com slide-in animation |
| `MetricsPanel.tsx` | Dashboard de latência com barras visuais |

**Hook central: `useRealtimeSession`**

```
Gerencia:
├── WebSocket connection lifecycle
├── Audio capture (AudioWorklet → 24kHz PCM16 → base64 → WS)
├── Audio playback (WS → base64 → PCM16 → AudioWorklet → Speaker)
├── Session state (connected, speaking, transcript)
├── Event dispatch (session.update, input_audio_buffer.append, response.create)
└── Metrics accumulation (avg, min, max por sessão)
```

---

### 15. Deploy

**Docker Compose (GPU):**

```yaml
services:
  stt:
    build: stt/Dockerfile.whisper
    port: 50060
    env:
      STT_PROVIDER: whisper
      WHISPER_MODEL: large-v3-turbo
      WHISPER_DEVICE: cuda
      WHISPER_COMPUTE_TYPE: int8
    gpu: all
    healthcheck: 120s startup

  tts:
    build: tts/Dockerfile.kokoro-gpu
    port: 50070
    env:
      TTS_PROVIDER: kokoro
      KOKORO_VOICE: pf_dora
      ONNX_PROVIDER: CUDAExecutionProvider
    gpu: all
    healthcheck: 60s startup
```

**Topologia de deploy:**

```
┌─────────────────────────────────────────────────┐
│                  Vast.ai (A100)                  │
│                                                   │
│  ┌──────────────┐  ┌──────────────┐              │
│  │  STT Server  │  │  TTS Server  │  GPU         │
│  │  :50060      │  │  :50070      │  containers   │
│  └──────┬───────┘  └──────┬───────┘              │
│         │                  │                      │
│  ┌──────┴──────────────────┴───────┐             │
│  │        API Server               │             │
│  │        :8765 (WebSocket)        │  CPU         │
│  └──────────────┬──────────────────┘             │
│                 │                                 │
│  ┌──────────────┴──────────────────┐             │
│  │    vLLM Server (opcional)       │             │
│  │    :8000 (Qwen2.5-7B)          │  GPU         │
│  └─────────────────────────────────┘             │
│                                                   │
└──────────────────────┬──────────────────────────┘
                       │ SSH tunnel
                       ▼
              ┌─────────────────┐
              │  Browser/Client │
              │  (React app)    │
              └─────────────────┘
```

---

## Configuração Completa

### Variáveis de Ambiente

| Variável | Default | Descrição |
|----------|---------|-----------|
| **WebSocket** | | |
| `WS_HOST` | `0.0.0.0` | Listen address |
| `WS_PORT` | `8765` | Port |
| `WS_PATH` | `/v1/realtime` | Endpoint path |
| `REALTIME_API_KEY` | *(vazio)* | Auth token (vazio = sem auth) |
| `MAX_CONNECTIONS` | `10` | Max sessões simultâneas |
| `WS_ALLOWED_ORIGINS` | *(vazio)* | CORS origins (comma-separated) |
| **ASR** | | |
| `ASR_PROVIDER` | `remote` | `remote` · `whisper` · `qwen` |
| `ASR_REMOTE_TARGET` | `localhost:50060` | gRPC server address |
| `ASR_LANGUAGE` | `pt` | Idioma |
| `ASR_REMOTE_TIMEOUT` | `30.0` | Timeout (s) |
| `ASR_REMOTE_STREAMING` | `false` | Habilitar streaming ASR |
| **TTS** | | |
| `TTS_PROVIDER` | `remote` | `remote` · `kokoro` · `qwen` · `edge` |
| `TTS_REMOTE_TARGET` | `localhost:50070` | gRPC server address |
| `TTS_LANGUAGE` | `pt` | Idioma |
| `TTS_VOICE` | `alloy` | Voz |
| `TTS_REMOTE_TIMEOUT` | `60.0` | Timeout (s) |
| **LLM** | | |
| `LLM_PROVIDER` | `anthropic` | `anthropic` · `openai` · `vllm` |
| `LLM_MODEL` | `claude-sonnet-4-20250514` | Modelo |
| `LLM_MAX_TOKENS` | `1024` | Max output tokens |
| `LLM_TEMPERATURE` | `0.8` | Temperatura |
| `LLM_TIMEOUT` | `30.0` | Timeout (s) |
| `LLM_SYSTEM_PROMPT` | `"You are a helpful assistant."` | System prompt |
| `ANTHROPIC_API_KEY` | — | Obrigatório se anthropic |
| `OPENAI_API_KEY` | — | Obrigatório se openai |
| `VLLM_BASE_URL` | — | Obrigatório se vllm |
| **VAD** | | |
| `VAD_SILENCE_MS` | `500` | Silêncio antes de finalizar turno (ms) |
| `VAD_PREFIX_PADDING_MS` | `300` | Áudio antes do onset a incluir (ms) |
| `VAD_MIN_SPEECH_MS` | `250` | Duração mínima de fala (ms) |
| `VAD_MIN_SPEECH_RMS` | `500` | RMS mínimo para filtrar ruído |
| `VAD_AGGRESSIVENESS` | `3` | 0=permissivo, 3=estrito |
| **Pipeline** | | |
| `PIPELINE_SENTENCE_QUEUE_SIZE` | `6` | Queue de frases |
| `PIPELINE_TTS_PREFETCH_SIZE` | `4` | Prefetch de áudio |
| `PIPELINE_MAX_SENTENCE_CHARS` | `150` | Max chars por frase |
| `PIPELINE_TTS_TIMEOUT` | `15.0` | Timeout TTS (s) |
| `PIPELINE_SENTENCE_TIMEOUT` | `15.0` | Timeout sentence queue (s) |
| **Tools** | | |
| `TOOL_ENABLE_MOCK` | `false` | Habilitar tools bancários demo |
| `TOOL_ENABLE_WEB_SEARCH` | `false` | Habilitar DuckDuckGo |
| `TOOL_TIMEOUT` | `10.0` | Timeout execução de tool (s) |
| `TOOL_MAX_ROUNDS` | `5` | Max rounds de tool calling |
| `TOOL_DEFAULT_FILLER` | `"Um momento, por favor."` | Filler padrão |
| **Logging** | | |
| `LOG_LEVEL` | `INFO` | `DEBUG` · `INFO` · `WARNING` · `ERROR` |

---

## Decisões de Design

| Decisão | Motivo | Trade-off |
|---------|--------|-----------|
| **LLM stateless** | Cada chamada recebe messages completas. Simplifica provider, permite troca de modelo sem perder contexto | Mais tokens por chamada |
| **Sentence pipelining** | Primeiro áudio toca antes do LLM terminar de gerar. Reduz latência percebida em 40-60% | Complexidade do producer-consumer |
| **Server-side VAD** | Silero ML é mais preciso que WebRTC VAD do browser. Smart Turn reduz respostas prematuras | Dependência de PyTorch no server |
| **Dual audio rate** | 24kHz para qualidade no client, 8kHz internamente (padrão telefonia, menos dados via gRPC) | Resampling em cada direção |
| **Provider auto-discovery** | Adicionar provider = criar arquivo + registrar. Sem tocar no core | Indireção via registry |
| **Filler não vai no histórico** | Se armazenado, LLM imita frases como "Vou pesquisar..." em vez de chamar tools | Filler desaparece do contexto |
| **Windowed context** | 8 items para tool calling reduz tokens sem perder contexto relevante | Pode perder contexto antigo |
| **gRPC microservices** | GPU isolation, modelo carregado 1x, independente de connections | Network hop extra (~1-2ms) |
| **Backpressure droppable** | Áudio dropado = glitch. Evento estrutural dropado = estado divergente = erro fatal | Possível glitch audível em clients lentos |
| **deque com maxlen** | Auto-evicção O(1) vs `list.pop(0)` O(n) | Sem log de evicção |
| **Session como fachada** | Session delega para ConversationStore, ResponseRunner, AudioInput. Reduz carga cognitiva | Mais arquivos para navegar |
| **ContextBuilder isolado** | Autoridade única sobre contexto LLM. Evita bugs de prompting espalhados | Uma indireção a mais |
| **ToolEngine extraído** | Tool execution isolado permite testar e evoluir independente da resposta | Coordenação entre runner e engine |
| **ResponseStrategy** | `select_strategy()` concentra toda decisão de "como responder" num único ponto | Mais um módulo para entender |
| **Protocol contract** | Invariantes formalizados em código. Previne degradação do contrato | Manutenção do doc + código |
| **Turn como conceito** | Turno explícito permite observabilidade por estágio e narrativa arquitetural | Abstração ainda não integrada end-to-end |

---

## Testes

**Estrutura:** 174+ testes (160+ API + 14 STT/TTS)

**Padrões:**
- **Fakes:** `FakeWebSocket`, `FakeASR`, `FakeLLM`, `FakeTTS` configuráveis
- **Sem sleeps:** testes determinísticos, `delay_after` para timing-sensitive
- **pytest-asyncio:** `asyncio_mode = "auto"`
- **Fixtures:** em `tests/conftest.py` — session pronta com fakes injetados

**Cobertura por módulo:**

| Módulo | Testes | Foco |
|--------|--------|------|
| `test_session.py` | 30+ | State machine, dispatch, barge-in |
| `test_response_runner.py` | 20+ | Tool calling, text/audio response |
| `test_audio_input.py` | 15+ | VAD callbacks, ASR streaming |
| `test_sentence_pipeline.py` | 10+ | Producer-consumer, timeouts |
| `test_tools.py` | 15+ | Registry, execute, mock handlers |
| `test_vad.py` | 10+ | Speech detection, silence, errors |
| `test_audio_codec.py` | 16 | PCM, G.711, resampling roundtrips |
| `test_web_search.py` | 10+ | DuckDuckGo, HTML extraction |
| `test_ws_server.py` | 3 | WebSocket integration |
| `test_tts_server.py` | 7 | gRPC servicer + lifecycle |
| `test_stt_server.py` | 7 | gRPC servicer + lifecycle |
