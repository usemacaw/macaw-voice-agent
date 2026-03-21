# Macaw Voice Agent

Agente de voz voice-to-voice em tempo real. Drop-in replacement para a OpenAI Realtime API com providers plugáveis de ASR, LLM e TTS.

## Comandos

```bash
# --- API Server (src/api/) ---
cd src/api
pip install -e ".[dev,vad]"       # Instalar deps + dev + VAD
python main.py                     # Iniciar servidor WebSocket :8765

# --- Testes (src/api/) ---
pytest -v                          # Todos os 109+ testes
pytest tests/test_session.py -v    # Arquivo específico
pytest -k "test_tool" -v           # Por padrão de nome

# --- Frontend (src/web/) ---
cd src/web
npm install                        # Instalar deps
npm run dev                        # Dev server → http://localhost:5173
npm run build                      # Build produção → dist/
npm run lint                       # ESLint

# --- STT Server (src/stt/) ---
cd src && PYTHONPATH=. python3 -m stt.server    # gRPC :50060
cd src && PYTHONPATH=. pytest stt/tests/ -v     # Testes

# --- TTS Server (src/tts/) ---
cd src && PYTHONPATH=. python3 -m tts.server    # gRPC :50070
cd src && PYTHONPATH=. pytest tts/tests/ -v     # Testes

# --- LLM Server (src/llm/) ---
cd src && PYTHONPATH=. python3 -m llm.server    # gRPC :50080

# --- Docker (GPU) ---
cd src && docker compose -f docker-compose.gpu.yml up -d   # STT + TTS + LLM
docker build -f src/stt/Dockerfile.whisper -t stt-whisper src/
docker build -f src/tts/Dockerfile.kokoro-gpu -t tts-kokoro src/
docker build -f src/llm/Dockerfile.qwen -t llm-qwen src/

# --- Health check ---
curl http://localhost:8765/health
```

## Estrutura do Projeto

```
src/
├── api/                    # Servidor WebSocket principal (Python 3.11+)
│   ├── main.py             # Entry point — cria providers, conecta, inicia WS server
│   ├── config.py           # Env vars + frozen dataclass policies (VadPolicy, LLMPolicy, etc.)
│   ├── .env                # Configuração ativa (NÃO committar)
│   ├── .env.example        # Template de configuração
│   ├── server/
│   │   ├── ws_server.py    # WebSocket lifecycle, health endpoint, rate limit
│   │   ├── session.py      # RealtimeSession — protocol facade per connection
│   │   ├── response_runner.py  # Response orchestrator (selects strategy, delegates)
│   │   ├── response/          # Decomposed response paths
│   │   │   ├── audio_response.py  # LLM → SentencePipeline → audio events
│   │   │   ├── text_response.py   # LLM → text deltas
│   │   │   └── tool_response.py   # LLM → tools → inline TTS → loop
│   │   ├── audio_input.py  # VAD, ASR, RMS, speech detection
│   │   ├── audio_emitter.py # TTS → encode → emit (unified)
│   │   ├── conversation_store.py # Conversation items + memory
│   │   ├── filler.py        # Filler phrases for tool waiting
│   │   └── system_metrics.py # System health metrics
│   ├── pipeline/
│   │   ├── sentence_pipeline.py  # LLM→TTS streaming (producer-consumer)
│   │   └── conversation.py       # Histórico → messages, windowing (últimos 8 items)
│   ├── providers/
│   │   ├── registry.py     # ProviderRegistry[T] — auto-discovery por módulo
│   │   ├── asr.py          # ABC: transcribe(), start_stream(), feed_chunk(), finish_stream()
│   │   ├── llm.py          # ABC: generate_stream(), generate_stream_with_tools()
│   │   ├── tts.py          # ABC: synthesize(), synthesize_stream()
│   │   ├── asr_remote.py   # gRPC client para STT server (:50060)
│   │   ├── tts_remote.py   # gRPC client para TTS server (:50070)
│   │   └── llm_remote.py   # gRPC client para LLM server (:50080)
│   ├── tools/
│   │   ├── registry.py     # ToolRegistry — register, execute, fork, timeout
│   │   ├── handlers.py     # Mock tools bancários (demo)
│   │   ├── web_search.py   # DuckDuckGo (zero API keys)
│   │   └── recall_memory.py    # Busca keyword no histórico da conversa
│   ├── audio/
│   │   ├── codec.py        # PCM16/G.711, resampling 24kHz↔8kHz, base64
│   │   ├── vad.py          # Silero VAD (ONNX), 32ms frames @ 8kHz
│   │   └── smart_turn.py   # Turn detection
│   ├── protocol/
│   │   ├── events.py       # Builders dos eventos OpenAI Realtime API + macaw.metrics
│   │   ├── models.py       # SessionConfig, TurnDetection (dataclasses)
│   │   └── event_emitter.py    # WebSocket sender com backpressure
│   └── tests/              # 14 arquivos, 109+ testes (pytest-asyncio)
│
├── stt/                    # Microserviço STT (gRPC :50060)
│   ├── server.py           # STTServicer + STTServer
│   ├── providers/          # qwen_stt, mock
│   └── tests/
│
├── tts/                    # Microserviço TTS (gRPC :50070)
│   ├── server.py           # TTSServicer + TTSServer
│   ├── providers/          # macaw_streaming_tts, mock
│   └── tests/
│
├── llm/                    # Microserviço LLM (gRPC :50080)
│   ├── server.py           # LLMServicer + LLMServer
│   └── providers/          # vllm_provider, mock
├── web/                    # React + TypeScript + Vite + Tailwind v3
│   └── src/
│       ├── App.tsx                         # UI principal (Orb + Transcript + Metrics)
│       ├── hooks/useRealtimeSession.ts     # Hook central (WS, audio, state)
│       ├── audio/capture.ts                # AudioWorklet mic (24kHz PCM16)
│       ├── audio/playback.ts               # AudioWorklet speaker
│       └── components/
│           ├── Orb.tsx                     # Orb animado (4 estados)
│           ├── TranscriptPanel.tsx         # Chat bubbles slide-in
│           └── MetricsPanel.tsx            # Dashboard de observabilidade
│
├── common/                 # Módulos compartilhados (config, audio_utils, executor)
├── shared/                 # Stubs gRPC gerados (stt_service, tts_service, llm_service)
└── docker-compose.gpu.yml  # STT + TTS + LLM com GPU
```

## Arquitetura

### Pipeline Voice-to-Voice

```
Mic → PCM16 24kHz → WebSocket → Silero VAD → ASR(gRPC) → LLM(gRPC+tools) → SentencePipeline → TTS(gRPC) → PCM16 24kHz → Speaker
```

### Decisões de Design

- **Todos providers via gRPC:** API server é um orquestrador puro — ASR (:50060), TTS (:50070), LLM (:50080) são microserviços gRPC independentes
- **LLM stateless:** recebe lista completa de messages a cada chamada. Histórico gerenciado por `RealtimeSession`
- **Sentence pipelining:** LLM streama texto → split em frases → TTS sintetiza em paralelo (asyncio queues, prefetch de 4 frases). Primeiro áudio toca antes do LLM terminar de gerar
- **Server-side VAD:** Silero ML (ONNX), 32ms chunks @ 8kHz. Callbacks `on_speech_started`/`on_speech_stopped`
- **Dual audio rate:** API fala 24kHz PCM16 (padrão cliente), interno usa 8kHz (padrão telefonia). `audio/codec.py` faz resampling
- **Provider auto-discovery:** `ProviderRegistry[T]` com import lazy. Adicionar arquivo de provider → disponível automaticamente
- **Handler dispatch:** `_HANDLER_MAP` string-based com `getattr(self, name)` evita circular refs
- **SlowClientError:** eventos droppable (`audio.delta`) silenciosamente descartados no timeout. Eventos estruturais (`response.done`) levantam erro → conexão terminada
- **Windowed context:** apenas últimos 8 items da conversa enviados ao LLM

### Tool Calling

- `ToolRegistry` em `tools/registry.py` — register async handlers com schemas, timeout, filler
- **Dual mode:** se ToolRegistry tem handlers → execução server-side. Se não → fallback client-side (compat OpenAI Realtime API)
- **Fluxo:** LLM emite tool_call → filler TTS enviado → tool executada → resultado no histórico → LLM re-chamado
- **Filler dinâmico:** frases randomizadas por ferramenta em `_build_dynamic_filler()` no `session.py`
- **Filler NÃO vai no histórico:** previne LLM de imitar frases de espera em vez de chamar ferramentas
- Config: `TOOL_ENABLE_MOCK`, `TOOL_ENABLE_WEB_SEARCH`, `TOOL_TIMEOUT` (10s), `TOOL_MAX_ROUNDS` (5)

### Observabilidade

Cada resposta emite `macaw.metrics` via WebSocket com timing por estágio:
- `asr_ms`, `llm_ttft_ms`, `llm_total_ms`, `llm_first_sentence_ms`
- `tts_synth_ms`, `tts_wait_ms`, `pipeline_first_audio_ms`
- `e2e_ms`, `total_ms`, `tool_timings[]`
- `turn`, `session_duration_s`, `barge_in_count`

Frontend exibe no `MetricsPanel.tsx` com barras visuais e sumário (avg/min/max).

## Configuração

Tudo via env vars. Referência completa em `src/api/.env.example`.

### Variáveis Críticas

| Variável | Padrão | Descrição |
|----------|--------|-----------|
| `ASR_PROVIDER` | `remote` | Provider ASR (gRPC) |
| `ASR_REMOTE_TARGET` | `localhost:50060` | Endereço do STT server |
| `TTS_PROVIDER` | `remote` | Provider TTS (gRPC) |
| `TTS_REMOTE_TARGET` | `localhost:50070` | Endereço do TTS server |
| `LLM_PROVIDER` | `remote` | Provider LLM (gRPC) |
| `LLM_REMOTE_TARGET` | `localhost:50080` | Endereço do LLM server |
| `LLM_SYSTEM_PROMPT` | *(built-in)* | Prompt do agente de voz |
| `TOOL_ENABLE_WEB_SEARCH` | `false` | Habilita DuckDuckGo |
| `TOOL_ENABLE_MOCK` | `false` | Habilita tools bancários mock |

### Audio

| Constante | Valor | Onde |
|-----------|-------|------|
| `API_SAMPLE_RATE` | 24000 | Client ↔ Server (WebSocket) |
| `INTERNAL_SAMPLE_RATE` | 8000 | Providers (gRPC) |
| `SAMPLE_WIDTH` | 2 | PCM16 (2 bytes/sample) |
| Formatos suportados | PCM16, G.711 mu-law, G.711 A-law | `audio/codec.py` |

## Padrões de Teste

- **Fakes:** `FakeWebSocket` com `asyncio.Event` drain, `FakeASR/FakeLLM/FakeTTS` configuráveis
- **Sem sleeps:** testes determinísticos, `delay_after` para timing-sensitive (ex: cancellation)
- **pytest-asyncio:** `asyncio_mode = "auto"` no `pyproject.toml`
- **Fixtures:** em `tests/conftest.py` — session pronta com todos os fakes injetados

## Cuidados Importantes

### System Prompt

- DEVE usar acentuação correta do português (você, não, é, está, cotação). O LLM aprende pelo exemplo do prompt — sem acentos no prompt = sem acentos na resposta
- Inclui instrução explícita: "NUNCA escreva sem acentos"
- Instrução de tool calling: "NUNCA gere texto dizendo que vai pesquisar — chame a ferramenta diretamente"

### Filler e Histórico

- Fillers NÃO são armazenados no histórico da conversa
- Se forem armazenados, o LLM começa a gerar texto tipo "Vou pesquisar..." em vez de chamar a ferramenta

### Qwen3 Thinking

- vLLM provider desabilita thinking mode (`enable_thinking: False`) para evitar `<think>` blocks na voz
- `_strip_think()` em `session.py` remove blocos residuais

### Emoji

- Emojis são removidos antes do TTS em dois pontos:
  1. `session.py` → `_clean_for_voice()` (path de tool calling)
  2. `sentence_pipeline.py` → `_EMOJI_RE` (path de streaming)

### Servidor

- Após mudanças no backend: reiniciar o server (`kill` + `python main.py`)
- Sessões WebSocket existentes usam o código antigo — reconectar o browser
- Verificar logs em `/tmp/macaw-api.log` para confirmar novo comportamento

## Deploy

Ver `docs/DEPLOYMENT.md` para configuração completa da Vast.ai (vLLM, STT, TTS, SSH tunnels, portas).

Ver `docs/GPU_PROVISIONING.md` para comparação de custos entre Vast.ai, AWS, GCP e Azure.
