# Macaw Voice Agent — Documento Técnico do Sistema

> Status atual do sistema em 2026-03-14
>
> Este documento descreve a arquitetura, componentes, decisões técnicas e estado operacional
> do macaw-voice-agent para que qualquer engenheiro consiga entender o sistema por completo.

---

## 1. Visão Geral

O Macaw Voice Agent é um agente de voz conversacional voice-to-voice que implementa
o protocolo OpenAI Realtime API. O sistema recebe áudio do microfone do usuário,
transcreve em texto (ASR), gera uma resposta via LLM com suporte a tool calling,
sintetiza o texto em áudio (TTS), e transmite de volta ao usuário — tudo em streaming.

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                             NAVEGADOR (Frontend)                            │
│                                                                             │
│  ┌─────────────┐    ┌──────────────────┐    ┌───────────────────────────┐   │
│  │ AudioWorklet │    │ useRealtimeSession│    │  React Components        │   │
│  │ capture.js   │───→│  (WebSocket)     │───→│  Orb + Transcript +     │   │
│  │ playback.js  │←───│                  │    │  MetricsPanel            │   │
│  └─────────────┘    └────────┬─────────┘    └───────────────────────────┘   │
└──────────────────────────────┼──────────────────────────────────────────────┘
                               │ WebSocket (PCM16 24kHz)
                               │ ws://host:8765/v1/realtime
┌──────────────────────────────┼──────────────────────────────────────────────┐
│                          API SERVER (Python)                                │
│                                                                             │
│  ┌──────────┐   ┌──────┐   ┌──────────┐   ┌───────────┐   ┌────────────┐  │
│  │ ws_server │──→│ VAD  │──→│   ASR    │──→│    LLM    │──→│   TTS      │  │
│  │          │   │Silero│   │ (gRPC)   │   │(Anthropic/│   │  (gRPC)    │  │
│  │          │←──│      │   │          │   │ vLLM)     │   │            │  │
│  └──────────┘   └──────┘   └──────────┘   └─────┬─────┘   └────────────┘  │
│                                                  │                          │
│                                           ┌──────┴──────┐                   │
│                                           │ ToolRegistry │                  │
│                                           │ web_search   │                  │
│                                           │ recall_memory│                  │
│                                           │ mock_tools   │                  │
│                                           └─────────────┘                   │
└─────────────────────────────────────────────────────────────────────────────┘
         │ gRPC :50060                              │ gRPC :50070
┌────────┴────────┐                        ┌────────┴────────┐
│   STT Service   │                        │   TTS Service   │
│  Whisper/Qwen3  │                        │  Kokoro/Qwen3   │
│  FastConformer  │                        │  FasterQwen3    │
└─────────────────┘                        └─────────────────┘
```

### Stack Tecnológico

| Camada | Tecnologia | Versão |
|--------|-----------|--------|
| **Frontend** | React + TypeScript + Vite | 18.3 / 5.6 / 5.4 |
| **Estilização** | Tailwind CSS | 3.4 |
| **Áudio (browser)** | Web Audio API + AudioWorklet | Nativo |
| **Transporte** | WebSocket | RFC 6455 |
| **API Server** | Python + websockets (asyncio) | 3.11+ / 13.0 |
| **ASR** | Faster-Whisper (CTranslate2) | large-v3-turbo |
| **LLM** | Claude / Qwen3-8B-AWQ (vLLM) | claude-sonnet-4 / Qwen3 |
| **TTS** | Kokoro-ONNX / FasterQwen3-TTS | 82M / 1.7B |
| **VAD** | Silero VAD (ONNX) | 5.1 |
| **IPC** | gRPC (HTTP/2) | 1.78 |
| **Infra** | Docker + Vast.ai (A100) | CUDA 12.4 |

---

## 2. Arquitetura de Componentes

### 2.1 Frontend (src/web/)

**9 arquivos de aplicação** (~1.200 linhas TypeScript + 150 linhas JS worklet)

```
src/web/
├── public/
│   ├── capture-processor.js     # AudioWorklet: mic → PCM16 24kHz (ring buffer + resample)
│   └── playback-processor.js    # AudioWorklet: PCM16 24kHz → speaker (queue + resample)
├── src/
│   ├── main.tsx                 # React root
│   ├── App.tsx                  # Layout, orb state machine, panel toggles
│   ├── types.ts                 # Message, SessionStatus, ResponseMetrics, ToolTiming
│   ├── hooks/
│   │   └── useRealtimeSession.ts # Estado central: WebSocket + áudio + protocol handler
│   ├── components/
│   │   ├── Orb.tsx              # Esfera animada (idle/listening/speaking/thinking)
│   │   ├── TranscriptPanel.tsx  # Painel lateral de transcrição
│   │   └── MetricsPanel.tsx     # Dashboard de métricas por resposta
│   └── audio/
│       ├── capture.ts           # Wrapper getUserMedia + AudioWorklet
│       └── playback.ts          # Wrapper AudioContext + AudioWorklet
```

**Fluxo de áudio (captura):**
1. `getUserMedia()` com AEC, noise suppression, auto gain
2. `AudioContext(sampleRate: 24000)` → `capture-processor.js` (AudioWorklet)
3. Ring buffer circular (4800 samples), resample linear para 24kHz
4. Chunks de 480 amostras (20ms) → base64 → WebSocket

**Fluxo de áudio (reprodução):**
1. Eventos `response.audio.delta` recebidos via WebSocket
2. Base64 decode → `postMessage()` para `playback-processor.js`
3. Fila de chunks, resample se necessário → output buffer → alto-falante
4. Suporte a barge-in: mensagem "clear" limpa fila instantaneamente

**Orb state machine:**

| Estado | Condição | Animação | Cor |
|--------|----------|----------|-----|
| `idle` | Conectado, nenhuma ação | Pulse 4s (scale 1→1.03) | Indigo |
| `listening` | `isUserSpeaking === true` | Bounce 1.2s (scale + rotate) | Azul |
| `speaking` | `isAssistantSpeaking === true` | Dance 0.8s (scale + brightness) | Roxo |
| `thinking` | Resposta em andamento, sem áudio ainda | Pulse 2s | Indigo |
| `disconnected` | `status !== "connected"` | Nenhuma (opacity 0.3) | Cinza |

### 2.2 API Server (src/api/)

**28 arquivos de aplicação** (~5.600 linhas Python)

O servidor é o orquestrador central. Recebe áudio via WebSocket, coordena VAD, ASR,
LLM e TTS, e transmite áudio de volta ao cliente.

```
src/api/
├── main.py                     # Entrypoint: factory de providers + startup
├── config.py                   # 60+ variáveis de ambiente com validação e ranges
├── server/
│   ├── ws_server.py            # Listener WebSocket, auth, health, rate limit
│   └── session.py              # Sessão por conexão (1.775 linhas — coração do sistema)
├── providers/
│   ├── registry.py             # ProviderRegistry[T] genérico com lazy import
│   ├── asr.py                  # ASRProvider ABC
│   ├── asr_remote.py           # gRPC para STT service (batch + streaming)
│   ├── llm.py                  # LLMProvider ABC + sentence generation
│   ├── llm_anthropic.py        # Claude API (tool calling nativo)
│   ├── llm_openai.py           # OpenAI API
│   ├── llm_vllm.py             # vLLM local (Qwen3-8B-AWQ)
│   ├── tts.py                  # TTSProvider ABC
│   └── tts_remote.py           # gRPC para TTS service
├── pipeline/
│   ├── sentence_pipeline.py    # Producer-consumer: LLM → sentence split → TTS paralelo
│   └── conversation.py         # ConversationItem ↔ mensagens LLM
├── audio/
│   ├── codec.py                # Base64↔PCM, G.711 μ-law/A-law, resample (lookup tables)
│   ├── vad.py                  # Silero VAD (32ms chunks, 8kHz)
│   ├── smart_turn.py           # Detecção semântica de turno (prosódia + intonação)
│   └── utils.py                # Conversão de formatos
├── protocol/
│   ├── models.py               # SessionConfig, ConversationItem, TurnDetection
│   ├── events.py               # Builder functions para todos os eventos do protocolo
│   └── event_emitter.py        # Emissor serial com backpressure (droppable vs structural)
└── tools/
    ├── registry.py             # ToolRegistry com async handlers, timeout, fork()
    ├── handlers.py             # Mock tools para demo (6 handlers financeiros)
    ├── web_search.py           # DuckDuckGo search (zero API keys)
    └── recall_memory.py        # Busca na memória da conversação (per-session)
```

#### Ciclo de vida de uma sessão

```
1. Cliente conecta via WebSocket
   ├── Auth check (REALTIME_API_KEY se configurado)
   ├── Rate limit check (200 eventos/s, ban de 5s)
   └── Session criada (session_id: sess_*, conversation_id: conv_*)

2. Emite: session.created + conversation.created

3. Loop principal: recebe eventos do cliente
   ├── session.update → reconfigurar VAD, formatos de áudio
   ├── input_audio_buffer.append → alimentar VAD com PCM
   │   ├── VAD detecta speech_started → emite evento
   │   └── VAD detecta speech_stopped → transcreve (ASR)
   │       ├── Cria item do usuário na conversa
   │       └── Se turn_detection.create_response → inicia resposta
   ├── response.create → inicia resposta manualmente
   └── response.cancel → cancela resposta em andamento

4. Resposta (pipeline):
   ├── LLM recebe histórico completo (stateless)
   ├── Se tool call detectado:
   │   ├── Sintetiza filler TTS ("Um momento, por favor.")
   │   ├── Executa ferramenta (async, com timeout)
   │   ├── Injeta resultado no histórico
   │   └── Re-chama LLM (máx 5 rounds)
   ├── Sentence pipeline (producer-consumer):
   │   ├── LLM streama tokens → split em sentenças
   │   ├── Eager-first: yield no primeiro clause break (≥20 chars)
   │   ├── TTS prefetch: sintetiza 4 sentenças à frente
   │   └── Chunks de ~100ms de áudio emitidos ao cliente
   └── Emite: response.done + macaw.metrics

5. Cleanup:
   ├── Idle timeout: 10 min sem mensagem → desconecta
   ├── Barge-in: nova fala cancela resposta ativa
   └── Desconexão: cancela tasks, finaliza streams ASR
```

#### Backpressure: eventos droppable vs structural

O `EventEmitter` classifica eventos em duas categorias para lidar com clientes lentos:

| Droppable (descartados silenciosamente) | Structural (causam desconexão se timeout 5s) |
|-----------------------------------------|----------------------------------------------|
| `response.audio.delta` | `response.created` |
| `response.text.delta` | `response.done` |
| `response.audio_transcript.delta` | `conversation.item.created` |
| `response.function_call_arguments.delta` | `error` |

**Racional:** deltas podem ser perdidos sem divergência de estado; eventos estruturais
definem o estado da conversa e não podem ser ignorados.

### 2.3 STT Service (src/stt/)

**Serviço gRPC independente** (port 50060)

```protobuf
service STTService {
  rpc Transcribe(TranscribeRequest) returns (TranscribeResponse);        // Batch
  rpc TranscribeStream(stream AudioChunk) returns (stream TranscribeResult); // Streaming
}
```

**Providers disponíveis:**

| Provider | Modelo | Tipo | VRAM | Latência |
|----------|--------|------|------|----------|
| `whisper` | large-v3-turbo | Pseudo-streaming (LocalAgreement-2) | ~1.7 GB | ~200-500ms batch |
| `qwen` | Qwen3-ASR-1.7B | Batch (+ streaming via vLLM) | ~3 GB | ~300ms batch |
| `fastconformer` | NVIDIA FastConformer | True streaming (encoder cache) | ~2 GB | ~100ms chunk |
| `mock` | — | Batch | 0 | 0ms |

**Formato de entrada:** PCM16 8kHz mono (2 bytes/sample)

**Streaming (Whisper — LocalAgreement-2):**
1. Acumula áudio em buffer
2. A cada 1s, re-transcreve o buffer inteiro
3. Compara prefixos de palavras entre inferências consecutivas
4. Palavras que coincidem em 2 inferências seguidas → "confirmadas"
5. Retorna parciais estáveis sem esperar fim da fala

### 2.4 TTS Service (src/tts/)

**Serviço gRPC independente** (port 50070)

```protobuf
service TTSService {
  rpc Synthesize(SynthesizeRequest) returns (SynthesizeResponse);           // Batch
  rpc SynthesizeStream(SynthesizeRequest) returns (stream AudioChunk);      // Streaming
}
```

**Providers disponíveis:**

| Provider | Modelo | Tipo | VRAM | TTFB |
|----------|--------|------|------|------|
| `kokoro` | Kokoro-ONNX 82M | True streaming (async nativo) | ~0.5 GB | ~100ms |
| `faster` | FasterQwen3-TTS 1.7B | True streaming (CUDA graphs) | ~3 GB | ~156ms |
| `qwen` | Qwen3-TTS 0.6B/1.7B | Batch | ~1-3 GB | ~500ms |
| `mock` | — (sine 440Hz) | Batch | 0 | 0ms |

**Formato de saída:** PCM16 8kHz mono

**Taxas internas de amostragem:**
- Kokoro: 24kHz nativo → resample para 8kHz
- FasterQwen3: 12kHz nativo → resample para 8kHz
- Qwen3-TTS: 24kHz nativo → resample para 8kHz

### 2.5 Common (src/common/)

Módulos compartilhados entre STT e TTS:

| Módulo | Função |
|--------|--------|
| `config.py` | Configuração de env vars compartilhada (AUDIO_SAMPLE_RATE, GRPC_*, LOG_LEVEL) |
| `audio_utils.py` | `pcm_to_float32()`, `float32_to_pcm()`, `resample()` (np.interp) |
| `executor.py` | `run_inference()` — ThreadPoolExecutor limitado (max 2 workers) |

---

## 3. Protocolo de Comunicação

### 3.1 WebSocket (Cliente ↔ API Server)

**Endpoint:** `ws://host:8765/v1/realtime`
**Formato:** JSON (OpenAI Realtime API compatible)
**Áudio:** PCM16 24kHz base64-encoded (ou G.711 μ-law/A-law 8kHz)

**Cliente → Servidor:**

| Evento | Payload | Frequência |
|--------|---------|-----------|
| `session.update` | `{session: {modalities, audio_format, turn_detection}}` | 1x após connect |
| `input_audio_buffer.append` | `{audio: "base64..."}` | ~50/s (20ms chunks) |
| `input_audio_buffer.commit` | `{}` | Manual mode |
| `input_audio_buffer.clear` | `{}` | Sob demanda |
| `conversation.item.create` | `{item: ConversationItem}` | Tool results (client-side) |
| `response.create` | `{response?: {modalities, instructions}}` | Manual trigger |
| `response.cancel` | `{}` | Interrupção |

**Servidor → Cliente:**

| Evento | Payload | Significado |
|--------|---------|------------|
| `session.created` | `{session}` | Handshake completo |
| `session.updated` | `{session}` | Config aplicada |
| `input_audio_buffer.speech_started` | `{item_id}` | VAD: fala detectada |
| `input_audio_buffer.speech_stopped` | `{}` | VAD: silêncio detectado |
| `conversation.item.input_audio_transcription.completed` | `{item_id, transcript}` | ASR resultado |
| `response.created` | `{response}` | Início da resposta |
| `response.output_item.added` | `{item}` | Novo item de assistente |
| `response.audio.delta` | `{delta: "base64..."}` | Chunk de áudio TTS |
| `response.audio_transcript.delta` | `{delta: "text"}` | Token de transcrição |
| `response.audio.done` | `{}` | Áudio finalizado |
| `response.done` | `{response}` | Resposta completa |
| `macaw.metrics` | `{metrics: {...}}` | Métricas de observabilidade |

### 3.2 gRPC (API Server ↔ STT/TTS)

**STT (port 50060):**
- `Transcribe` (unary): áudio completo → texto
- `TranscribeStream` (bidirectional): chunks de áudio → transcrições parciais + final

**TTS (port 50070):**
- `Synthesize` (unary): texto → áudio completo
- `SynthesizeStream` (server-streaming): texto → chunks de áudio sequenciais

**Config gRPC compartilhada:**
- Max message: 10MB
- Keepalive: 30s ping, 10s timeout
- Health check: registrado para monitoramento
- Reflection: habilitada para introspecção

---

## 4. Formatos de Áudio

O sistema lida com múltiplos formatos de áudio em diferentes camadas:

```
                    24kHz PCM16                8kHz PCM16               16kHz float32
BROWSER ──────────────────────→ API SERVER ──────────────→ STT SERVICE ──────────────→ MODELO ASR
                                                                        (resample interno)

                    24kHz PCM16                8kHz PCM16               24/12kHz float32
BROWSER ←────────────────────── API SERVER ←────────────── TTS SERVICE ←────────────── MODELO TTS
                                                                        (resample interno)
```

| Ponto | Sample Rate | Encoding | Bits |
|-------|-------------|----------|------|
| Browser ↔ API Server | 24 kHz | PCM16 LE ou G.711 | 16 ou 8 |
| API Server ↔ STT/TTS | 8 kHz | PCM16 LE | 16 |
| Interno Whisper | 16 kHz | float32 | 32 |
| Interno Kokoro | 24 kHz | float32 | 32 |
| Interno FasterQwen3 | 12 kHz | float32 | 32 |

**G.711:** lookup tables pré-computadas (65.536 entradas) — ~5ms vs ~100ms por encoding.

---

## 5. Pipeline de Sentenças (Sentence Pipeline)

O componente mais crítico para a latência é o `SentencePipeline` (`pipeline/sentence_pipeline.py`,
270 linhas), que implementa um padrão producer-consumer para paralelizar LLM e TTS:

```
LLM (streaming tokens)
  │
  ├── Acumula tokens em buffer
  ├── Eager-first: yield no primeiro clause break (,;:-) se ≥20 chars
  ├── Depois: yield em fim de sentença (.!?)
  │
  ▼
Sentence Queue (max 6)
  │
  ├── TTS Worker 1: sintetiza sentença 1
  ├── TTS Worker 2: sintetiza sentença 2 (prefetch)
  ├── TTS Worker 3: sintetiza sentença 3 (prefetch)
  ├── TTS Worker 4: sintetiza sentença 4 (prefetch)
  │
  ▼
Audio Queue
  │
  ├── Chunk ~100ms de áudio
  ├── Base64 encode
  └── Emit response.audio.delta
```

**Configurações:**
- `PIPELINE_SENTENCE_QUEUE_SIZE=6` — buffer de sentenças
- `PIPELINE_TTS_PREFETCH_SIZE=4` — quantas sentenças sintetizar à frente
- `PIPELINE_MAX_SENTENCE_CHARS=150` — tamanho máximo de sentença
- `PIPELINE_TTS_TIMEOUT=15s` — timeout por sentença TTS

**Impacto:** a primeira sentença chega ao usuário enquanto o LLM ainda está gerando
a segunda. Isso reduz a latência percebida em ~200-400ms.

---

## 6. Tool Calling

### 6.1 Dois modos de execução

**Server-Side (ToolRegistry ativo):**
```
LLM emite tool_call
  → Sintetiza filler TTS ("Vou consultar seu saldo.")
  → ToolRegistry.execute(name, args) com timeout
  → Cria items: function_call + function_call_output
  → Re-chama LLM com resultado (máx 5 rounds)
  → Resposta final via sentence pipeline
```

**Client-Side (sem ToolRegistry):**
```
LLM emite tool_call
  → Emite evento response.function_call_arguments.done
  → Cliente executa a tool externamente
  → Cliente envia conversation.item.create com resultado
  → Servidor re-chama LLM
```

### 6.2 Ferramentas disponíveis

| Ferramenta | Tipo | Descrição |
|------------|------|-----------|
| `web_search` | Produção | DuckDuckGo search (general/news), zero API keys |
| `recall_memory` | Per-session | Busca keyword no histórico da conversa |
| `lookup_customer` | Mock/Demo | Consulta dados de cliente fictício |
| `get_account_balance` | Mock/Demo | Consulta saldo fictício |
| `get_card_info` | Mock/Demo | Consulta info de cartão fictício |
| `get_recent_transactions` | Mock/Demo | Consulta transações fictícias |
| `create_support_ticket` | Mock/Demo | Cria ticket de suporte fictício |
| `transfer_to_human` | Mock/Demo | Transfere para humano fictício |

### 6.3 Filler TTS durante execução

Enquanto a ferramenta executa, o servidor sintetiza uma frase de filler
para manter a conversa natural. Frases são randomizadas por ferramenta:

```python
# Exemplo: web_search
fillers = [
    "Vou pesquisar isso para você.",
    "Um momento enquanto eu busco essa informação.",
    "Deixa eu procurar aqui...",
]
```

**Importante:** fillers NÃO são armazenados no histórico da conversa para
evitar contaminação do contexto do LLM.

---

## 7. VAD (Voice Activity Detection)

### 7.1 Silero VAD

**Modelo:** Silero VAD v5 (ONNX, ~2MB)
**Chunks:** 32ms a 8kHz (256 amostras)
**Threshold:** configurável via `VAD_AGGRESSIVENESS` (0-3)

**Parâmetros:**

| Config | Default | Descrição |
|--------|---------|-----------|
| `VAD_AGGRESSIVENESS` | 3 | Nível de agressividade (0=relaxed, 3=strict) |
| `VAD_SILENCE_MS` | 500 | Silêncio necessário para declarar fim de fala |
| `VAD_PREFIX_PADDING_MS` | 300 | Áudio mantido antes do início da fala |
| `VAD_MIN_SPEECH_MS` | 250 | Duração mínima para considerar fala válida |
| `VAD_MIN_SPEECH_RMS` | 500 | RMS mínimo para filtrar ruído/eco |

### 7.2 Smart Turn Detection

**Arquivo:** `audio/smart_turn.py` (116 linhas)

Análise semântica complementar ao VAD de silêncio. Examina prosódia e
padrões de intonação para determinar se o usuário terminou de falar
ou está apenas fazendo uma pausa natural.

**Benefício:** reduz ~30% das interrupções falsas comparado com VAD de silêncio puro.

### 7.3 Barge-in

Quando o VAD detecta nova fala durante uma resposta do assistente:
1. Verifica RMS > `VAD_MIN_SPEECH_RMS` (filtra eco/ruído)
2. Cancela `response_task` via `task.cancel()`
3. Frontend recebe `speech_started` → limpa fila de áudio
4. Nova resposta inicia automaticamente após ASR

---

## 8. Provider Registry

O sistema usa um padrão genérico de registry com lazy import para carregar
providers sob demanda:

```python
class ProviderRegistry(Generic[T]):
    _known_modules = {"remote": "providers.asr_remote", ...}

    def create(name: str) -> T:
        if name not in self._providers and name in self._known_modules:
            importlib.import_module(self._known_modules[name])
        return self._providers[name]()
```

**Benefício:** dependências pesadas (torch, anthropic, onnxruntime) são importadas
somente quando o provider correspondente é configurado. Isso mantém o startup rápido
e permite que o mesmo código rode com diferentes combinações de providers.

**Providers registrados:**

| Camada | Provider | Módulo | Dependências pesadas |
|--------|----------|--------|---------------------|
| ASR | `remote` | `asr_remote.py` | grpcio |
| ASR | `qwen` | `asr_qwen.py` | torch, qwen-asr |
| ASR | `whisper` | `asr_whisper.py` | faster-whisper, ctranslate2 |
| LLM | `anthropic` | `llm_anthropic.py` | anthropic |
| LLM | `openai` | `llm_openai.py` | openai |
| LLM | `vllm` | `llm_vllm.py` | openai (client) |
| TTS | `remote` | `tts_remote.py` | grpcio |
| TTS | `kokoro` | `tts_kokoro.py` | kokoro-onnx, onnxruntime |
| TTS | `edge` | `tts_edge.py` | edge-tts |
| TTS | `qwen` | `tts_qwen.py` | torch, qwen-tts |

---

## 9. Observabilidade

### 9.1 Métricas por resposta

A cada `response.done`, o servidor emite um evento `macaw.metrics` com timing
detalhado de cada estágio:

```json
{
  "type": "macaw.metrics",
  "metrics": {
    "response_id": "resp_abc123",
    "turn": 3,
    "session_duration_s": 45.2,

    "speech_ms": 1500.0,
    "speech_rms": 2150.0,
    "asr_ms": 250.0,
    "asr_mode": "batch",
    "input_chars": 42,

    "llm_ttft_ms": 150.0,
    "llm_total_ms": 520.0,
    "llm_first_sentence_ms": 180.0,

    "tts_synth_ms": 800.0,
    "tts_wait_ms": 100.0,
    "pipeline_first_audio_ms": 450.0,

    "e2e_ms": 1200.0,
    "total_ms": 2000.0,

    "sentences": 3,
    "audio_chunks": 15,
    "output_chars": 120,
    "barge_in_count": 0,

    "tool_rounds": 1,
    "tools_used": ["web_search"],
    "tool_timings": [
      {"name": "web_search", "exec_ms": 850, "ok": true}
    ]
  }
}
```

### 9.2 Dashboard no frontend

O `MetricsPanel` exibe:
- **Summary card:** médias, mínimos e máximos de E2E, ASR, LLM TTFT, TTS
- **Per-response cards:** barras horizontais coloridas por estágio, tool timings, detalhes
- **Badges:** contagem de turns, barge-ins, tool calls

### 9.3 Logging

- Logs estruturados via `logging` (não print)
- `LOG_LEVEL` configurável (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- Métricas periódicas a cada 5s: chunks de áudio, estado do VAD, items na conversa

---

## 10. Segurança e Proteções

| Proteção | Implementação | Local |
|----------|--------------|-------|
| **Autenticação WS** | `REALTIME_API_KEY` header check | `ws_server.py` |
| **Rate limiting** | 200 eventos/s, ban de 5s | `session.py` |
| **Idle timeout** | 10 min sem mensagem → desconexão | `session.py` |
| **Max connections** | `MAX_CONNECTIONS` (default 10) | `ws_server.py` |
| **Input validation** | Rejeita system role injection | `models.py` |
| **Config size limit** | Validação de tamanho em session.update | `models.py` |
| **Origin validation** | `WS_ALLOWED_ORIGINS` | `ws_server.py` |
| **Tool timeout** | `TOOL_TIMEOUT` (default 10s) | `tools/registry.py` |
| **Tool max rounds** | `TOOL_MAX_ROUNDS` (default 5) | `session.py` |
| **Audio buffer limit** | 10 minutos máximo | `session.py` |
| **Slow client detection** | Structural events: 5s timeout → desconexão | `event_emitter.py` |

---

## 11. Testes

**17 arquivos de teste** (~2.600 linhas)

### Coverage por módulo

| Módulo | Testes | Arquivo |
|--------|--------|---------|
| ToolRegistry | Registro, execução, schemas, timeout, fork | `test_tools.py` |
| Mock handlers | 6 handlers financeiros | `test_tools.py` |
| Web search | HTML parse, DuckDuckGo, content fetch | `test_web_search.py` |
| Recall memory | Busca, truncação, integração | `test_recall_memory.py` |
| Session lifecycle | Criação, cancellation, error handling | `test_session.py` |
| Conversation window | Windowing 8 items, orphan cleanup | `test_conversation_window.py` |
| Conversation items | Items → LLM messages, tool pairs | `test_conversation.py` |
| Sentence split | Pontuação, Unicode, edge cases | `test_llm_sentences.py` |
| Config validation | Ranges, modalities, formats | `test_config_validation.py` |
| Protocol events | Serialização de todos os eventos | `test_events.py` |
| VAD | State machine, chunks, callbacks | `test_vad.py` |
| Audio codec | PCM roundtrip, G.711, resample | `test_audio_codec.py` |
| WS server | Startup, session flow, integration | `test_ws_server.py` |
| SDK compat | OpenAI SDK connectivity | `test_sdk_compat.py` |
| STT server | Batch, streaming, errors, lifecycle | `test_stt_server.py` |
| TTS server | Batch, streaming, errors, timing | `test_tts_server.py` |

### Infraestrutura de teste

**Fake providers** (determinísticos, sem I/O externo):
- `FakeWebSocket` — grava mensagens enviadas, emite mensagens com delays configuráveis
- `FakeASR` — retorna "hello world" (batch, sem streaming)
- `FakeLLM` — retorna texto fixo
- `FakeTTS` — retorna áudio fixo (sine wave)

**Padrões:**
- Arrange-Act-Assert
- `asyncio.Event` para sincronização (sem sleep)
- `patch.dict(os.environ)` para config isolation
- Testes independentes (sem estado compartilhado)

**Execução:**
```bash
cd src/api && python -m pytest tests/ -v
cd src/stt && python -m pytest tests/ -v
cd src/tts && python -m pytest tests/ -v
```

---

## 12. Configuração

### 12.1 Variáveis de ambiente (API Server)

<details>
<summary>Tabela completa (clique para expandir)</summary>

| Variável | Default | Range | Descrição |
|----------|---------|-------|-----------|
| `WS_HOST` | `0.0.0.0` | — | Host do WebSocket |
| `WS_PORT` | `8765` | 0-65535 | Porta do WebSocket |
| `WS_PATH` | `/v1/realtime` | — | Path do WebSocket |
| `REALTIME_API_KEY` | (vazio) | — | API key para auth |
| `MAX_CONNECTIONS` | `10` | 1-1000 | Conexões simultâneas |
| `ASR_PROVIDER` | `remote` | remote/qwen/whisper | Provider ASR |
| `ASR_REMOTE_TARGET` | `localhost:50060` | — | Endereço gRPC STT |
| `ASR_REMOTE_TIMEOUT` | `30.0` | — | Timeout ASR |
| `ASR_REMOTE_STREAMING` | `false` | — | Habilitar streaming ASR |
| `ASR_LANGUAGE` | `pt` | — | Idioma ASR |
| `TTS_PROVIDER` | `remote` | remote/kokoro/qwen/edge | Provider TTS |
| `TTS_REMOTE_TARGET` | `localhost:50070` | — | Endereço gRPC TTS |
| `TTS_REMOTE_TIMEOUT` | `60.0` | — | Timeout TTS |
| `TTS_LANGUAGE` | `pt` | — | Idioma TTS |
| `TTS_VOICE` | `alloy` | — | Voz TTS |
| `LLM_PROVIDER` | `anthropic` | anthropic/openai/vllm | Provider LLM |
| `LLM_MODEL` | `claude-sonnet-4-20250514` | — | Modelo LLM |
| `LLM_MAX_TOKENS` | `1024` | 1-100k | Máximo de tokens |
| `LLM_TEMPERATURE` | `0.8` | 0.0-2.0 | Temperatura |
| `LLM_TIMEOUT` | `30` | 1-300s | Timeout LLM |
| `LLM_SYSTEM_PROMPT` | (default) | — | System prompt |
| `VAD_AGGRESSIVENESS` | `3` | 0-3 | Agressividade VAD |
| `VAD_SILENCE_MS` | `500` | 50-5000 | Silêncio para fim de fala |
| `VAD_PREFIX_PADDING_MS` | `300` | 0-5000 | Áudio pré-fala |
| `VAD_MIN_SPEECH_MS` | `250` | 50-5000 | Duração mínima de fala |
| `VAD_MIN_SPEECH_RMS` | `500` | 0-50k | RMS mínimo |
| `PIPELINE_SENTENCE_QUEUE_SIZE` | `6` | 1-100 | Fila de sentenças |
| `PIPELINE_TTS_PREFETCH_SIZE` | `4` | 1-50 | Prefetch TTS |
| `PIPELINE_MAX_SENTENCE_CHARS` | `150` | 50-1000 | Chars máx/sentença |
| `PIPELINE_TTS_TIMEOUT` | `15` | 1-120s | Timeout TTS/sentença |
| `TOOL_ENABLE_MOCK_TOOLS` | `false` | — | Habilitar mock tools |
| `TOOL_ENABLE_WEB_SEARCH` | `false` | — | Habilitar web search |
| `TOOL_TIMEOUT` | `10` | 1-60s | Timeout de tool |
| `TOOL_MAX_ROUNDS` | `5` | 1-20 | Rounds máx de tool |
| `TOOL_DEFAULT_FILLER` | "Um momento, por favor." | — | Filler padrão |
| `LOG_LEVEL` | `INFO` | DEBUG-CRITICAL | Nível de log |

</details>

### 12.2 Variáveis de ambiente (STT/TTS)

| Variável | Default | Descrição |
|----------|---------|-----------|
| `GRPC_HOST` | `0.0.0.0` | Host gRPC |
| `GRPC_PORT` | `50060` (STT) / `50070` (TTS) | Porta gRPC |
| `GRPC_MAX_MESSAGE_SIZE` | `10485760` | Tamanho máx mensagem (10MB) |
| `STT_PROVIDER` | `mock` | Provider STT |
| `TTS_PROVIDER` | `mock` | Provider TTS |
| `AUDIO_SAMPLE_RATE` | `8000` | Sample rate padrão |
| `INFERENCE_MAX_WORKERS` | `2` | Workers de inferência |

---

## 13. Deploy Atual (Vast.ai)

### Topologia

```
┌─────────────────────┐     SSH Tunnel        ┌──────────────────────┐
│   Máquina Local     │     :8100 → :8000     │  Vast.ai — A100      │
│                     │◄─────────────────────►│  vLLM                │
│   API Server        │     ssh5.vast.ai:21068│  Qwen3-8B-AWQ        │
│   (port 8765)       │                       │  (port 8000)         │
│                     │                       └──────────────────────┘
│   Frontend          │
│   (port 5173)       │     gRPC direto       ┌──────────────────────┐
│                     │◄─────────────────────►│  Vast.ai — GPU       │
└─────────────────────┘     :10473, :10584    │  STT: Whisper        │
                                               │  (port 10473)        │
                                               │  TTS: Kokoro         │
                                               │  (port 10584)        │
                                               └──────────────────────┘
```

### Configuração de produção

**Instância 1 — LLM:**
- GPU: NVIDIA A100
- Modelo: `Qwen/Qwen3-8B-AWQ` (4-bit, ~5GB VRAM)
- Server: vLLM com tool calling (Hermes parser)
- `--max-model-len 8192 --gpu-memory-utilization 0.90`
- Acesso: SSH tunnel `localhost:8100 → remote:8000`

**Instância 2 — STT + TTS:**
- STT: Faster-Whisper `large-v3-turbo` (~1.7GB VRAM), int8, CUDA
- TTS: Kokoro-ONNX (~0.5GB VRAM), voz `pf_dora` (PT-BR feminina)
- Acesso: gRPC público `142.127.68.223:10473` (STT) e `:10584` (TTS)

**SSH Tunnel:**
```bash
ssh -N -L 8100:localhost:8000 -p 21068 root@ssh5.vast.ai
```

---

## 14. Métricas de Latência (Observadas)

Baseado em operação real do sistema:

| Estágio | Média | Min | P95 | Target SOTA |
|---------|-------|-----|-----|-------------|
| ASR | ~200-500ms | ~100ms | ~600ms | <100ms |
| LLM TTFT | ~150-300ms | ~80ms | ~500ms | <150ms |
| LLM total | ~500-1500ms | ~200ms | ~2000ms | <500ms |
| TTS (1st chunk) | ~100-300ms | ~50ms | ~500ms | <100ms |
| Pipeline 1st audio | ~400-800ms | ~200ms | ~1200ms | <300ms |
| **E2E** | **~800-1200ms** | **~400ms** | **~2000ms** | **<500ms** |

**Nota:** métricas variam significativamente com tamanho do input, complexidade
da resposta, e uso de tools. Targets SOTA baseados em pesquisa documentada
em `docs/SOTA_VOICE_AGENTS_RESEARCH.md`.

---

## 15. Decisões Arquiteturais

### LLM Stateless
O LLM recebe o histórico completo a cada chamada. A sessão é dona do histórico.
**Por quê:** replay safety, independência entre sessões, sem memory leaks no LLM.

### Sentence-Level Pipelining
LLM e TTS rodam em paralelo via producer-consumer.
**Por quê:** reduz latência percebida em ~200-400ms (usuário ouve primeira sentença
enquanto LLM gera a segunda).

### Server-Side VAD + Smart Turn
VAD roda no servidor (não no browser).
**Por quê:** detecção consistente, independente de browser, complementada
com análise semântica para reduzir falsos positivos.

### Microserviços STT/TTS separados
ASR e TTS rodam como serviços gRPC independentes.
**Por quê:** deploy independente em GPUs distintas, escalabilidade, isolamento
de falhas, possibilidade de swap de modelos sem afetar API server.

### Windowed Conversation History
LLM recebe apenas os últimos 8 items da conversa.
**Por quê:** reduz context size (menos tokens = menor latência LLM), evita
context overflow em conversas longas. Orphan tool calls são limpos automaticamente.

### G.711 Lookup Tables
Tabelas de 65.536 entradas pré-computadas no import.
**Por quê:** encoding/decoding G.711 em ~5ms vs ~100ms com cálculo por sample.

### ToolRegistry.fork() per-session
Cada sessão recebe cópia do registry.
**Por quê:** `recall_memory` precisa acessar histórico da sessão específica,
sem visibilidade cross-session.

### Droppable vs Structural Events
Eventos de delta podem ser descartados; eventos estruturais causam desconexão.
**Por quê:** evita que clientes lentos acumulem backlog infinito, enquanto
garante que o estado da conversa permaneça consistente.

---

## 16. Estrutura de Arquivos Completa

```
macaw-voice-agent/
├── README.md                           # Documentação do projeto
├── CLAUDE.md                           # Instruções para Claude Code
├── CHANGELOG.md                        # Registro de mudanças
├── docs/
│   ├── TECHNICAL_STATUS.md             # Este documento
│   ├── DEPLOYMENT.md                   # Guia de deploy Vast.ai
│   ├── GPU_PROVISIONING.md             # Provisioning de GPUs
│   ├── ASR_STREAMING_RESEARCH.md       # Pesquisa ASR streaming
│   ├── VOICE_AGENT_TOOLS_RESEARCH.md   # Pesquisa tool calling
│   └── SOTA_VOICE_AGENTS_RESEARCH.md   # Pesquisa SOTA voice agents
├── src/
│   ├── api/                            # API Server (5.600 linhas Python)
│   │   ├── main.py                     # Entrypoint (106 linhas)
│   │   ├── config.py                   # Configuração (113 linhas)
│   │   ├── server/
│   │   │   ├── ws_server.py            # WebSocket server (195 linhas)
│   │   │   └── session.py              # Sessão — coração do sistema (1.775 linhas)
│   │   ├── providers/
│   │   │   ├── registry.py             # Registry genérico (52 linhas)
│   │   │   ├── asr.py                  # ASR ABC (65 linhas)
│   │   │   ├── asr_remote.py           # ASR gRPC (214 linhas)
│   │   │   ├── llm.py                  # LLM ABC + sentences (192 linhas)
│   │   │   ├── llm_anthropic.py        # Claude (222 linhas)
│   │   │   ├── llm_openai.py           # OpenAI (165 linhas)
│   │   │   ├── llm_vllm.py            # vLLM (165 linhas)
│   │   │   ├── tts.py                  # TTS ABC (64 linhas)
│   │   │   └── tts_remote.py           # TTS gRPC (111 linhas)
│   │   ├── pipeline/
│   │   │   ├── sentence_pipeline.py    # LLM→TTS pipeline (270 linhas)
│   │   │   └── conversation.py         # Items↔Messages (144 linhas)
│   │   ├── audio/
│   │   │   ├── codec.py               # Codecs de áudio (194 linhas)
│   │   │   ├── vad.py                 # Silero VAD (220 linhas)
│   │   │   ├── smart_turn.py          # Turn detection semântico (116 linhas)
│   │   │   └── utils.py              # Helpers (57 linhas)
│   │   ├── protocol/
│   │   │   ├── models.py             # Modelos de dados (301 linhas)
│   │   │   ├── events.py             # Event builders (347 linhas)
│   │   │   └── event_emitter.py      # Emissor com backpressure (82 linhas)
│   │   ├── tools/
│   │   │   ├── registry.py           # ToolRegistry (211 linhas)
│   │   │   ├── handlers.py           # Mock handlers (294 linhas)
│   │   │   ├── web_search.py         # DuckDuckGo (153 linhas)
│   │   │   └── recall_memory.py      # Memória da conversa (131 linhas)
│   │   └── tests/                     # 14 arquivos de teste (~2.200 linhas)
│   ├── stt/                           # STT Service (~1.400 linhas Python)
│   │   ├── server.py                  # gRPC server (237 linhas)
│   │   ├── providers/
│   │   │   ├── base.py               # ABC + Mock (220 linhas)
│   │   │   ├── whisper_stt.py         # Faster-Whisper (277 linhas)
│   │   │   ├── qwen_stt.py           # Qwen3-ASR (329 linhas)
│   │   │   └── fastconformer_stt.py  # NeMo FastConformer (319 linhas)
│   │   ├── Dockerfile                # CPU
│   │   ├── Dockerfile.gpu            # Qwen3-ASR GPU
│   │   ├── Dockerfile.whisper        # Faster-Whisper GPU
│   │   └── tests/
│   │       └── test_stt_server.py    # (204 linhas)
│   ├── tts/                           # TTS Service (~1.000 linhas Python)
│   │   ├── server.py                  # gRPC server (226 linhas)
│   │   ├── providers/
│   │   │   ├── base.py               # ABC + Mock (180 linhas)
│   │   │   ├── kokoro_tts.py         # Kokoro-ONNX (196 linhas)
│   │   │   ├── qwen_tts.py           # Qwen3-TTS (204 linhas)
│   │   │   └── faster_tts.py         # FasterQwen3-TTS (234 linhas)
│   │   ├── Dockerfile                # CPU
│   │   ├── Dockerfile.gpu            # Qwen3-TTS GPU
│   │   ├── Dockerfile.kokoro-gpu     # Kokoro-ONNX GPU
│   │   └── tests/
│   │       └── test_tts_server.py    # (194 linhas)
│   ├── common/                        # Módulos compartilhados (226 linhas)
│   │   ├── config.py                 # Config env vars (92 linhas)
│   │   ├── audio_utils.py            # PCM↔float32, resample (87 linhas)
│   │   └── executor.py              # ThreadPool para inferência (47 linhas)
│   ├── shared/grpc_gen/              # Stubs protobuf gerados
│   ├── web/                          # Frontend React (~1.350 linhas TS/JS)
│   └── docker-compose.gpu.yml       # Compose STT+TTS
└── .claude/                          # Configuração Claude Code
    ├── settings.json                 # Permissões, hooks, env
    ├── rules/                        # Regras por path
    ├── agents/                       # 5 agentes especializados
    └── skills/                       # 10 skills (slash commands)
```

**Total de código de aplicação:** ~9.600 linhas Python + ~1.350 linhas TypeScript/JS
**Total de testes:** ~2.600 linhas (17 arquivos)

---

## 17. Limitações Conhecidas

| Limitação | Impacto | Mitigação possível |
|-----------|---------|-------------------|
| Latência E2E ~800-1200ms (vs SOTA <500ms) | Conversação menos natural | Streaming ASR, co-location, modelo LLM menor |
| Sem reconexão automática no frontend | Usuário precisa reconectar manualmente | Implementar retry com backoff exponencial |
| Histórico fixo em 8 items | Perda de contexto em conversas longas | `recall_memory` tool para buscar histórico |
| Session.py com 1.775 linhas | Complexidade concentrada | Extrair sub-responsabilidades |
| Sem TLS em dev | Inseguro para produção | Usar WSS + TLS em produção |
| Qwen3-8B tool calling intermitente | ~10-15% falha em tool calls | Prompt engineering, fallback para texto |
| Resampling por interpolação linear | Artefatos leves em ratios grandes | DSP mais sofisticado (sinc) |
| Sem métricas de longo prazo | Perda de histórico entre sessões | Exportar para Prometheus/Grafana |

---

## 18. Glossário

| Termo | Definição |
|-------|-----------|
| **ASR** | Automatic Speech Recognition — transcrição de fala em texto |
| **TTS** | Text-to-Speech — síntese de texto em fala |
| **LLM** | Large Language Model — modelo de linguagem para geração de respostas |
| **VAD** | Voice Activity Detection — detecção de atividade vocal |
| **TTFT** | Time to First Token — tempo até o primeiro token do LLM |
| **TTFB** | Time to First Byte — tempo até o primeiro byte de áudio TTS |
| **E2E** | End-to-End — latência total do fim da fala até início do áudio de resposta |
| **Barge-in** | Interrupção do assistente quando o usuário começa a falar |
| **Filler** | Frase de preenchimento durante execução de tool ("Um momento...") |
| **PCM** | Pulse Code Modulation — formato de áudio digital não comprimido |
| **G.711** | Codec de áudio para telefonia (μ-law / A-law) |
| **gRPC** | Google Remote Procedure Call — protocolo de comunicação binário (HTTP/2) |
| **Sentence Pipeline** | Padrão producer-consumer que paraleliza LLM streaming com TTS |
| **Smart Turn** | Detecção semântica de turno baseada em prosódia/intonação |
| **Droppable Event** | Evento que pode ser descartado sem divergência de estado |
| **Structural Event** | Evento crítico que define o estado da conversa |
