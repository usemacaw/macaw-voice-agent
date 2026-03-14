# Plano de Refatoração — Macaw Voice Agent

> Gerado em 2026-03-14 a partir de revisão arquitetural profunda.
> **Executado em 2026-03-14** — Fases 1, 2 e 3 (parcial) implementadas.
> Todos os 174 testes passam (160 API + 14 STT/TTS).

---

## Visão Geral

| Fase | Escopo | Esforço | Risco |
|------|--------|---------|-------|
| **Fase 1** | Quick wins — bugs silenciosos e DRY | 1-2 dias | Baixo |
| **Fase 2** | Refatorações estruturais | 3-5 dias | Médio |
| **Fase 3** | Melhorias futuras (trigger-based) | Sob demanda | Baixo-médio |

**Regra:** cada item gera um commit atômico com testes. Nenhum item depende de outro dentro da mesma fase, salvo onde indicado.

---

## Fase 1 — Quick Wins (1-2 dias)

### 1.1 Corrigir métricas TTFT/total em Anthropic e OpenAI providers

**Problema:** `last_ttft_ms` e `last_stream_total_ms` só são atribuídos no `VLLMProvider`. Nos providers Anthropic e OpenAI, o timing é calculado como variável local e logado, mas **nunca atribuído a `self`**. Resultado: `llm_ttft_ms` e `llm_total_ms` no `macaw.metrics` são sempre `0.0` para esses providers.

**Arquivos afetados:**
- `src/api/providers/llm_anthropic.py` — linhas 130-131 (`generate_stream`) e 170-171 (`generate_stream_with_tools`)
- `src/api/providers/llm_openai.py` — linhas 77-78 (`generate_stream`) e 120-121 (`generate_stream_with_tools`)

**Referência correta (vLLM):**
```python
# src/api/providers/llm_vllm.py:77
self.last_ttft_ms = (time.perf_counter() - t0) * 1000
```

**O que fazer:**

1. Em `llm_anthropic.py`, nos dois métodos (`generate_stream` e `generate_stream_with_tools`):
   - Onde calcula `ttft_ms` como local, atribuir a `self.last_ttft_ms`
   - Ao final do generator, calcular e atribuir `self.last_stream_total_ms = (time.perf_counter() - t0) * 1000`

2. Em `llm_openai.py`, mesmas alterações nos dois métodos.

3. Verificar que `LLMProvider` (ABC em `llm.py:29-30`) já define os atributos com default `0.0`.

**Validação:**
- Rodar o server com `LLM_PROVIDER=anthropic`, fazer uma chamada de voz
- Verificar no `macaw.metrics` que `llm_ttft_ms` e `llm_total_ms` são > 0
- Adicionar teste unitário: mock do stream, assert que `provider.last_ttft_ms > 0` após consumir o generator

**Testes a criar:**
```
tests/test_llm_metrics.py:
  - test_anthropic_sets_ttft_ms
  - test_anthropic_sets_total_ms
  - test_openai_sets_ttft_ms
  - test_openai_sets_total_ms
```

---

### 1.2 Extrair helper compartilhado de tool stream parsing (OpenAI/vLLM)

**Problema:** `llm_openai.py:110-153` e `llm_vllm.py:111-153` têm ~40 linhas idênticas de parsing de tool call deltas do protocolo OpenAI streaming. Violação DRY — se o protocolo mudar, dois arquivos precisam ser atualizados.

**Arquivos afetados:**
- `src/api/providers/llm_openai.py` — linhas 110-153
- `src/api/providers/llm_vllm.py` — linhas 111-153

**O que fazer:**

1. Criar `src/api/providers/_openai_stream.py` (módulo privado, prefixo `_`):
   ```python
   async def parse_openai_tool_stream(
       stream,
       on_first_token: Callable[[], None] | None = None,
   ) -> AsyncGenerator[LLMStreamEvent, None]:
       """Parse OpenAI-compatible streaming response with tool calls."""
       active_tool_calls: dict[int, dict] = {}
       first_token = False

       async for chunk in stream:
           delta = chunk.choices[0].delta if chunk.choices else None
           if not delta:
               continue

           if not first_token:
               first_token = True
               if on_first_token:
                   on_first_token()

           if delta.content:
               yield LLMStreamEvent(type="text_delta", text=delta.content)

           if delta.tool_calls:
               for tc in delta.tool_calls:
                   idx = tc.index
                   if idx not in active_tool_calls:
                       tc_id = tc.id or ""
                       tc_name = tc.function.name if tc.function and tc.function.name else ""
                       active_tool_calls[idx] = {"id": tc_id, "name": tc_name}
                       yield LLMStreamEvent(
                           type="tool_call_start",
                           tool_call_id=tc_id,
                           tool_name=tc_name,
                       )
                   if tc.function and tc.function.arguments:
                       info = active_tool_calls[idx]
                       yield LLMStreamEvent(
                           type="tool_call_delta",
                           tool_call_id=info["id"],
                           tool_name=info["name"],
                           tool_arguments_delta=tc.function.arguments,
                       )

       for info in active_tool_calls.values():
           yield LLMStreamEvent(
               type="tool_call_end",
               tool_call_id=info["id"],
               tool_name=info["name"],
           )
   ```

2. Em `llm_openai.py`, substituir o bloco duplicado por:
   ```python
   def _mark_first():
       nonlocal first_token
       first_token = True
       self.last_ttft_ms = (time.perf_counter() - t0) * 1000
       logger.info(f"LLM first event in {self.last_ttft_ms:.0f}ms")

   async for event in parse_openai_tool_stream(stream, on_first_token=_mark_first):
       yield event
   ```

3. Mesma substituição em `llm_vllm.py` (com a adição do `extra_body` nos kwargs antes do stream).

**Validação:**
- Testes existentes de tool calling devem continuar passando
- `pytest tests/ -k "tool" -v`

---

### 1.3 Remover `requires_confirmation` do ToolDef (dead code)

**Problema:** Campo definido na dataclass `ToolDef` (`registry.py:37-38`), aceito no método `register()` (`registry.py:80-109`), armazenado na instância, mas **nunca lido em nenhum execution path**. É dead code — feature planejada mas não implementada.

**Arquivos afetados:**
- `src/api/tools/registry.py` — linhas 38, 87, 103

**O que fazer:**

1. Remover `requires_confirmation: bool = False` da dataclass `ToolDef`
2. Remover o parâmetro `requires_confirmation` do método `register()`
3. Remover do construtor `ToolDef(...)` na linha 103
4. Grep pelo projeto inteiro para encontrar qualquer caller que passe `requires_confirmation=True` e remover
5. Se algum handler em `handlers.py` seta esse campo, remover

**Validação:**
- `pytest tests/ -v` — todos os testes passam
- `grep -rn "requires_confirmation" src/` — nenhum resultado

---

### 1.4 Consolidar resampling duplicado em smart_turn.py

**Problema:** `smart_turn.py:73-80` reimplementa resampling inline com `np.interp`/`np.linspace` em vez de chamar `common.audio_utils.resample()`. Código idêntico em dois lugares.

**Arquivos afetados:**
- `src/api/audio/smart_turn.py` — linhas 73-80
- `src/common/audio_utils.py` — linhas 49-70 (a implementação canônica)

**O que fazer:**

1. Em `smart_turn.py`, importar `resample` de `common.audio_utils`:
   ```python
   from common.audio_utils import resample
   ```

2. Substituir o bloco inline (linhas 73-80) por:
   ```python
   if source_sample_rate != _SMART_TURN_SAMPLE_RATE:
       samples = resample(samples, source_sample_rate, _SMART_TURN_SAMPLE_RATE)
       if len(samples) == 0:
           return True, 1.0
   ```

**Validação:**
- Testes de smart turn continuam passando
- Verificar que o import path `common.audio_utils` funciona do contexto do API server (já funciona — `sentence_pipeline.py` e outros usam `common.*`)

---

## Fase 2 — Refatorações Estruturais (3-5 dias)

### 2.1 Decompor `response_runner.py` (SRP)

**Problema:** `response_runner.py` tem 1086 linhas com 8+ responsabilidades. `_run_with_tools` tem 173 linhas. `_emit_tool_response_audio` e `_emit_tool_response_audio_streamed` compartilham estrutura duplicada. Filler phrases PT-BR hardcoded no mesmo arquivo que o motor de execução.

**Arquivos afetados:**
- `src/api/server/response_runner.py` — arquivo inteiro

**O que fazer:**

#### 2.1.1 Extrair FillerManager

Criar `src/api/server/filler.py`:

```python
class FillerManager:
    """Gera e sintetiza filler phrases contextuais durante tool calling."""

    def __init__(self, tts: TTSProvider, emitter: EventEmitter):
        self._tts = tts
        self._emitter = emitter

    def build_filler(self, tool_name: str, arguments_json: str) -> str:
        """Gera frase de filler contextual baseada na ferramenta."""
        ...  # Mover _build_dynamic_filler() + constantes _SEARCH_FILLERS, etc.

    async def send_filler_audio(
        self, text: str, response_id: str, item_id: str, output_index: int, content_index: int
    ) -> None:
        """Sintetiza e envia áudio de filler via WebSocket."""
        ...  # Mover _send_filler_audio()
```

**Mover de `response_runner.py`:**
- Constantes: `_SEARCH_FILLERS`, `_MEMORY_FILLERS`, `_GENERIC_FILLERS` (linhas 68-93)
- Função: `_build_dynamic_filler()` (linhas 95-113)
- Método: `_send_filler_audio()` (linhas 606-631)

#### 2.1.2 Unificar emit_tool_response_audio

Os métodos `_emit_tool_response_audio` (linhas 633-729, 97 linhas) e `_emit_tool_response_audio_streamed` (linhas 731-777, 47 linhas) compartilham:
- Criação de assistant item
- Emissão de `response_output_item_added`
- Chamada a `_run_audio_response`
- Emissão de `response_audio_done`

Unificar em um único método:

```python
async def _emit_tool_response_audio(
    self,
    response_id: str,
    output_index: int,
    response_start: float,
    # Se collected_text é fornecido, usa direto. Se não, faz streaming LLM→TTS.
    collected_text: str | None = None,
    messages: list[dict] | None = None,
    system: str = "",
    temperature: float = 0.8,
    max_tokens: int = 1024,
) -> None:
```

#### 2.1.3 Extrair text cleaning

Mover `_clean_for_voice()` e as regex `_THINK_RE`, `_EMOJI_RE` para um módulo utilitário (pode ser `src/api/audio/text_clean.py` ou inline no pipeline, já que `sentence_pipeline.py` tem sua própria `_EMOJI_RE`).

**Resultado esperado:**
- `response_runner.py` cai de ~1086 para ~750 linhas
- `filler.py` ~120 linhas (coeso e testável isoladamente)
- `_run_with_tools` continua grande mas perde a responsabilidade de filler

**Validação:**
- Todos os testes existentes passam (`pytest tests/ -v`)
- Testes novos para `FillerManager`:
  ```
  test_filler_build_web_search_with_query
  test_filler_build_web_search_without_query
  test_filler_build_recall_memory
  test_filler_build_generic
  ```

---

### 2.2 Mover `generate_sentences()` do LLM ABC para o pipeline

**Problema:** `generate_sentences()` (linhas 105-155 de `llm.py`) e `split_long_sentence()` (linhas 158-184) implementam lógica de sentence splitting — responsabilidade do pipeline, não do provider. O `SentencePipeline` é o único consumidor.

**Arquivos afetados:**
- `src/api/providers/llm.py` — linhas 60-184 (regex + generate_sentences + split_long_sentence)
- `src/api/pipeline/sentence_pipeline.py` — consome `generate_sentences()`

**O que fazer:**

1. Criar `src/api/pipeline/sentence_splitter.py`:
   ```python
   """Sentence splitting utilities for the LLM→TTS pipeline."""

   import re
   from collections.abc import AsyncGenerator
   from providers.llm import LLMProvider

   _SENTENCE_END = re.compile(...)  # Mover de llm.py
   _CLAUSE_BREAK = re.compile(...)
   _BREAK_POINTS = re.compile(...)

   async def generate_sentences(
       llm: LLMProvider,
       messages: list[dict],
       system: str = "",
       tools: list[dict] | None = None,
       temperature: float = 0.8,
       max_tokens: int = 1024,
   ) -> AsyncGenerator[str, None]:
       """Stream complete sentences from LLM output."""
       ...  # Lógica atual de llm.py:generate_sentences

   def split_long_sentence(sentence: str, max_chars: int) -> list[str]:
       ...  # Lógica atual de llm.py:split_long_sentence
   ```

2. Em `sentence_pipeline.py`, trocar:
   ```python
   # Antes:
   async for sentence in self._llm.generate_sentences(messages, system, ...)
   # Depois:
   from pipeline.sentence_splitter import generate_sentences
   async for sentence in generate_sentences(self._llm, messages, system, ...)
   ```

3. Em `llm.py`, remover `generate_sentences()`, `split_long_sentence()`, e as regex associadas. Manter apenas a ABC pura com `generate_stream` e `generate_stream_with_tools`.

**Validação:**
- `pytest tests/ -v`
- Verificar que `SentencePipeline` funciona via teste de integração existente

---

### 2.3 Health check dinâmico nos microserviços gRPC

**Problema:** STT e TTS servers definem health como `SERVING` no startup e **nunca atualizam**. Se o modelo crashar (OOM, CUDA error), o health check mente. Na API, `ws_server.py:100-112` quebra encapsulação com `hasattr(self._asr, "_stub")`.

**Arquivos afetados:**
- `src/tts/server.py` — health setup
- `src/stt/server.py` — health setup
- `src/api/server/ws_server.py` — linhas 100-112 (`_check_provider_health`)
- `src/api/providers/asr.py` — ABC
- `src/api/providers/tts.py` — ABC

**O que fazer:**

#### 2.3.1 Adicionar health tracking nos Servicers

Em `TTSServicer` e `STTServicer`, adicionar contagem de erros consecutivos:

```python
class TTSServicer:
    def __init__(self, provider, health_servicer):
        self._provider = provider
        self._health = health_servicer
        self._consecutive_errors = 0
        self._MAX_ERRORS_BEFORE_UNHEALTHY = 5

    async def Synthesize(self, request, context):
        try:
            result = await self._provider.synthesize(request.text)
            self._consecutive_errors = 0
            return result
        except Exception as e:
            self._consecutive_errors += 1
            if self._consecutive_errors >= self._MAX_ERRORS_BEFORE_UNHEALTHY:
                self._health.set("tts.TTSService", HealthCheckResponse.NOT_SERVING)
                logger.error(f"Provider degraded after {self._consecutive_errors} errors")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return tts_pb2.SynthesizeResponse()
```

Quando o provider voltar a funcionar (erro count reseta para 0), restaurar `SERVING`.

#### 2.3.2 Adicionar `health_check()` ao provider ABC

Em `src/api/providers/asr.py` e `tts.py`:

```python
class ASRProvider(ABC):
    async def health_check(self) -> bool:
        """Return True if provider is healthy. Override for custom checks."""
        return True
```

Em `src/api/providers/asr_remote.py` e `tts_remote.py`, override:

```python
async def health_check(self) -> bool:
    """Check gRPC channel connectivity."""
    try:
        return self._channel is not None and self._stub is not None
    except Exception:
        return False
```

#### 2.3.3 Limpar `_check_provider_health` no ws_server

Substituir `hasattr(self._asr, "_stub")` por:

```python
if hasattr(self._asr, "health_check"):
    status["asr_healthy"] = await self._asr.health_check()
```

Ou melhor, como `health_check()` está no ABC, simplesmente:

```python
status["asr_healthy"] = await self._asr.health_check()
status["tts_healthy"] = await self._tts.health_check()
```

**Validação:**
- Simular falha de provider em teste e verificar que health muda para NOT_SERVING
- Simular recovery e verificar retorno a SERVING
- `_check_provider_health` não usa mais `hasattr(_stub)`

---

### 2.4 Extrair boilerplate gRPC para `common/grpc_server.py`

**Problema:** `stt/server.py` (238 linhas) e `tts/server.py` (227 linhas) têm estrutura quase idêntica: keepalive options, max message size, health check, reflection, graceful shutdown.

**Arquivos afetados:**
- `src/tts/server.py`
- `src/stt/server.py`

**O que fazer:**

1. Criar `src/common/grpc_server.py`:
   ```python
   """Shared gRPC server infrastructure for STT/TTS microservices."""

   import grpc
   from grpc_health.v1 import health, health_pb2_grpc, health_pb2
   from grpc_reflection.v1alpha import reflection

   DEFAULT_OPTIONS = [
       ("grpc.keepalive_time_ms", 30000),
       ("grpc.keepalive_timeout_ms", 10000),
       ("grpc.keepalive_permit_without_calls", True),
       ("grpc.max_receive_message_length", 4 * 1024 * 1024),
       ("grpc.max_send_message_length", 50 * 1024 * 1024),
   ]

   class GrpcMicroservice:
       """Base for gRPC microservices with health check, reflection, and graceful shutdown."""

       def __init__(self, service_name: str, port: int, provider):
           self._service_name = service_name
           self._port = port
           self._provider = provider
           self._server = None
           self._health_servicer = None

       @property
       def health_servicer(self):
           return self._health_servicer

       async def start(self, add_servicer_fn):
           """Start gRPC server. add_servicer_fn receives (server, health_servicer)."""
           self._server = grpc.aio.server(options=DEFAULT_OPTIONS)
           self._health_servicer = health.HealthServicer()
           add_servicer_fn(self._server, self._health_servicer)
           health_pb2_grpc.add_HealthServicer_to_server(self._health_servicer, self._server)
           reflection.enable_server_reflection([...], self._server)
           self._health_servicer.set(self._service_name, health_pb2.HealthCheckResponse.SERVING)
           await self._server.start()

       async def stop(self, grace: float = 5.0):
           """Graceful shutdown."""
           self._health_servicer.set(self._service_name, health_pb2.HealthCheckResponse.NOT_SERVING)
           await self._server.stop(grace)
           if hasattr(self._provider, "disconnect"):
               await self._provider.disconnect()
   ```

2. Simplificar `tts/server.py` e `stt/server.py` para ~60-80 linhas cada:
   ```python
   class TTSServer:
       def __init__(self, provider):
           self._micro = GrpcMicroservice("tts.TTSService", TTS_CONFIG["port"], provider)

       async def run(self):
           def add_servicers(server, health):
               servicer = TTSServicer(provider, health)
               tts_pb2_grpc.add_TTSServiceServicer_to_server(servicer, server)
           await self._micro.start(add_servicers)
   ```

**Validação:**
- Ambos os servers iniciam e respondem corretamente
- Health check funciona
- gRPC reflection funciona
- Graceful shutdown funciona

---

### 2.5 Migrar config dicts para frozen dataclasses

**Problema:** `api/config.py` exporta dicts legados (`WS_CONFIG`, `ASR_CONFIG`, etc.) E frozen dataclasses (`VadPolicy`, `LLM`, etc.) — dual representation confusa. `common/config.py` usa apenas dicts sem type safety.

**Arquivos afetados:**
- `src/api/config.py` — linhas 39-114 (dicts) e 120-176 (dataclasses)
- `src/common/config.py` — linhas 59-92 (dicts)

**O que fazer:**

#### 2.5.1 `common/config.py` — converter para dataclasses

```python
@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 8000
    sample_width: int = 2
    channels: int = 1

@dataclass(frozen=True)
class STTConfig:
    provider: str = "whisper"
    port: int = 50060
    model_size: str = "base"

@dataclass(frozen=True)
class TTSConfig:
    provider: str = "kokoro"
    port: int = 50070
    voice: str = "af_heart"

AUDIO = AudioConfig(...)
STT = STTConfig(...)
TTS = TTSConfig(...)
```

#### 2.5.2 `api/config.py` — eliminar dicts, usar apenas dataclasses

- Converter `WS_CONFIG` → `ConnectionPolicy` (já existe parcialmente)
- Converter `ASR_CONFIG`, `TTS_CONFIG`, `AUDIO_CONFIG` → dataclasses
- Converter `LOG_CONFIG` → `LogPolicy`
- Remover dicts, exportar apenas instâncias frozen
- Atualizar todos os consumidores: trocar `CONFIG["key"]` por `CONFIG.key`

**Consumidores a atualizar (grep `_CONFIG\[`):**
- `ws_server.py` — usa `WS_CONFIG`
- `asr_remote.py` — usa `ASR_CONFIG`
- `tts_remote.py` — usa `TTS_CONFIG`
- `codec.py` — usa `AUDIO_CONFIG` (se houver)
- `common/` microserviços — usam `AUDIO_CONFIG`, `STT_CONFIG`, `TTS_CONFIG`

**Validação:**
- Todos os testes passam
- Autocompleção funciona em IDEs
- Typo em atributo gera `AttributeError` em vez de `KeyError` silencioso

---

## Fase 3 — Melhorias Futuras (Sob Demanda)

### 3.1 Propagação de request ID via gRPC metadata

**Trigger:** quando debugging de latência em produção se tornar inviável por falta de correlação entre API server e microserviços.

**Problema:** chamadas batch (`Transcribe`, `Synthesize`) não têm request ID. Não há trace context propagado do API server para STT/TTS. Correlação depende de timestamps manuais.

**Arquivos afetados:**
- `src/api/providers/asr_remote.py` — client gRPC
- `src/api/providers/tts_remote.py` — client gRPC
- `src/stt/server.py` — server gRPC
- `src/tts/server.py` — server gRPC

**O que fazer:**

1. No client gRPC (API server), propagar `request_id` via metadata:
   ```python
   metadata = [("x-request-id", request_id)]
   response = await self._stub.Synthesize(request, metadata=metadata)
   ```

2. No server gRPC (microserviços), extrair e logar:
   ```python
   request_id = dict(context.invocation_metadata()).get("x-request-id", "unknown")
   logger.info(f"[{request_id}] Synthesize: {len(request.text)} chars")
   ```

3. Gerar `request_id` no `AudioInputHandler` ao iniciar transcrição, propagar para `ResponseRunner`, que propaga para TTS.

**Validação:**
- Logs do API server e microserviços contêm o mesmo `request_id` para um turn
- Grep nos logs por um ID específico retorna o fluxo completo

---

### 3.2 Client-side retry com backoff para gRPC

**Trigger:** quando falhas transientes de STT/TTS em produção forem recorrentes (OOM temporário, GPU throttling, network hiccup).

**Problema:** sem retry, qualquer falha transiente de inferência é surfaceada como erro para o usuário, mesmo que a próxima tentativa funcionaria.

**Arquivos afetados:**
- `src/api/providers/asr_remote.py`
- `src/api/providers/tts_remote.py`

**O que fazer:**

1. Implementar retry decorator com exponential backoff + jitter:
   ```python
   async def _retry_grpc(fn, max_retries=2, base_delay=0.1):
       for attempt in range(max_retries + 1):
           try:
               return await fn()
           except grpc.aio.AioRpcError as e:
               if e.code() in (grpc.StatusCode.UNAVAILABLE, grpc.StatusCode.INTERNAL):
                   if attempt < max_retries:
                       delay = base_delay * (2 ** attempt) + random.uniform(0, 0.05)
                       await asyncio.sleep(delay)
                       continue
               raise
   ```

2. Aplicar nos métodos `transcribe()` e `synthesize()` dos remote providers.

3. **Não** aplicar retry em streaming (complexidade alta, benefício baixo).

**Validação:**
- Teste: mock gRPC que falha na primeira tentativa e sucede na segunda
- Verificar que retry não ultrapassa o timeout total da pipeline
- Métricas: logar `retry_count` no `macaw.metrics`

---

### 3.3 Refatorar VAD state machine

**Trigger:** quando for necessário adicionar novos estados ao VAD (ex: "noise gate", "typing detection", "hold turn").

**Problema:** `_process_chunk` em `vad.py` (linhas 111-198+) tem ~87 linhas com if/else aninhado 4 níveis. State machine implícita como condicionais nested.

**Arquivos afetados:**
- `src/api/audio/vad.py` — método `_process_chunk`

**O que fazer:**

1. Definir estados explícitos:
   ```python
   class VADState(Enum):
       IDLE = "idle"                    # Esperando speech
       SPEECH_STARTING = "starting"    # Speech detectado, aguardando confirmação
       SPEAKING = "speaking"           # Usuário falando
       SILENCE_DETECTED = "silence"    # Silêncio após fala, aguardando turn end
       SMART_TURN_WAIT = "smart_wait"  # Aguardando smart turn prediction
   ```

2. Extrair handlers por estado:
   ```python
   def _process_chunk(self, chunk: bytes) -> None:
       prob = self._run_inference(chunk)
       is_speech = prob >= self._threshold

       handler = {
           VADState.IDLE: self._handle_idle,
           VADState.SPEECH_STARTING: self._handle_speech_starting,
           VADState.SPEAKING: self._handle_speaking,
           VADState.SILENCE_DETECTED: self._handle_silence,
           VADState.SMART_TURN_WAIT: self._handle_smart_wait,
       }[self._state]

       handler(chunk, is_speech, prob)
   ```

3. Cada handler é um método de 10-20 linhas com transição de estado explícita.

**Validação:**
- Todos os testes de VAD existentes passam sem alteração
- Novos testes para cada transição de estado individual
- Comportamento observável idêntico (mesmos callbacks, mesmos timings)

---

### 3.4 `ResponseMetrics` como dataclass tipada

**Trigger:** quando o número de métricas crescer ou quando bugs por typo em keys de dict de métricas aparecerem.

**Problema:** métricas de resposta são passadas como `dict[str, object]` entre `AudioInputHandler` e `ResponseRunner` via `session.py`. Sem tipo definido, sem contrato explícito.

**Arquivos afetados:**
- `src/api/server/session.py` — linha 120 (`self._response_metrics: dict[str, object] = {}`)
- `src/api/server/response_runner.py` — lê e escreve no dict
- `src/api/server/audio_input.py` — carrega `response_metrics`

**O que fazer:**

1. Criar `src/api/protocol/metrics.py`:
   ```python
   @dataclass
   class ResponseMetrics:
       turn: int = 0
       asr_ms: float = 0.0
       llm_ttft_ms: float = 0.0
       llm_total_ms: float = 0.0
       llm_first_sentence_ms: float = 0.0
       tts_synth_ms: float = 0.0
       tts_wait_ms: float = 0.0
       pipeline_first_audio_ms: float = 0.0
       e2e_ms: float = 0.0
       total_ms: float = 0.0
       tool_timings: list[dict] = field(default_factory=list)
       session_duration_s: float = 0.0
       barge_in_count: int = 0

       def to_dict(self) -> dict:
           return asdict(self)
   ```

2. Substituir `dict[str, object]` por `ResponseMetrics` em `session.py`, `response_runner.py`, e `audio_input.py`.

**Validação:**
- Testes de métricas existentes passam
- Typo em atributo gera `AttributeError` imediato

---

### 3.5 Fechar singletons de `web_search.py` no shutdown

**Trigger:** quando resource leaks aparecerem em logs ou quando o módulo for carregado/descarregado repetidamente (ex: em testes).

**Problema:** `_ddgs` (DuckDuckGo client) e `_http_client` (httpx.AsyncClient) são criados no import e nunca fechados.

**Arquivos afetados:**
- `src/api/tools/web_search.py` — linhas 24, 27-31

**O que fazer:**

1. Converter para lazy initialization com cleanup:
   ```python
   _ddgs: DDGS | None = None
   _http_client: httpx.AsyncClient | None = None

   def _get_ddgs() -> DDGS:
       global _ddgs
       if _ddgs is None:
           _ddgs = DDGS()
       return _ddgs

   def _get_http_client() -> httpx.AsyncClient:
       global _http_client
       if _http_client is None:
           _http_client = httpx.AsyncClient(
               timeout=5.0, follow_redirects=True,
               headers={"User-Agent": "Mozilla/5.0 (compatible; OpenVoiceAPI/1.0)"},
           )
       return _http_client

   async def cleanup():
       global _ddgs, _http_client
       if _http_client:
           await _http_client.aclose()
           _http_client = None
       _ddgs = None
   ```

2. Chamar `web_search.cleanup()` no shutdown do server (em `main.py` ou `ws_server.stop()`).

**Validação:**
- Testes de web search continuam passando
- Sem warnings de unclosed client em testes com `pytest -W error`

---

### 3.6 Vetorizar MockTTS com NumPy

**Trigger:** quando MockTTS for usado em benchmarks ou load testing.

**Problema:** `MockTTS.synthesize()` em `tts/providers/base.py:106-122` gera áudio sample-by-sample com Python loop + `struct.pack`. Ineficiente.

**Arquivos afetados:**
- `src/tts/providers/base.py` — linhas 106-122

**O que fazer:**

```python
async def synthesize(self, text: str) -> bytes:
    sample_rate = AUDIO_CONFIG["sample_rate"]
    duration = max(1.0, len(text) * 0.05)
    frequency = 440
    n_samples = int(sample_rate * duration)

    t = np.arange(n_samples, dtype=np.float32) / sample_rate
    envelope = np.minimum(1.0, t * 10) * np.minimum(1.0, (duration - t) * 10)
    samples = (16000 * envelope * np.sin(2 * np.pi * frequency * t)).astype(np.int16)

    pcm_data = samples.tobytes()
    logger.info(f"TTS (mock): {len(pcm_data)} bytes")
    return pcm_data
```

**Validação:**
- Output PCM é audível e idêntico em forma de onda
- Performance: < 1ms para 1s de áudio (vs ~50ms com loop Python)

---

### 3.7 `list.pop(0)` → `deque` em session items

**Trigger:** se `_MAX_CONVERSATION_ITEMS` crescer significativamente ou se profiling mostrar hotspot.

**Problema:** `session.py:184-194` usa `self._items.pop(0)` que é O(n). Com `_MAX_CONVERSATION_ITEMS = 200`, impacto é negligível, mas `deque(maxlen=200)` seria O(1).

**Arquivos afetados:**
- `src/api/server/session.py` — linhas 184-194

**O que fazer:**

1. Trocar `self._items: list` por `self._items: deque` com `maxlen=_MAX_CONVERSATION_ITEMS`
2. `deque` auto-evicta o mais antigo ao atingir maxlen — elimina `_enforce_items_limit()` inteiro
3. Verificar que nenhum código usa indexação `self._items[i]` ou slicing `self._items[a:b]` (deque suporta indexação mas slicing é O(k))

**Validação:**
- Testes existentes passam
- Verificar que auto-evicção da deque mantém os items mais recentes

---

### 3.8 Batching de audio deltas no EventEmitter

**Trigger:** quando p99 de latência degradar e profiling mostrar `json.dumps` como hotspot.

**Problema:** `event_emitter.py:59` faz `json.dumps()` por cada `response.audio.delta`. Centenas de chamadas por resposta, sem batching.

**Arquivos afetados:**
- `src/api/protocol/event_emitter.py` — linha 59

**O que fazer:**

1. Acumular 2-3 audio deltas consecutivos antes de serializar:
   ```python
   async def emit_audio_batch(self, deltas: list[dict]) -> None:
       """Batch multiple audio deltas into fewer WebSocket sends."""
       for delta in deltas:
           if "event_id" not in delta:
               delta["event_id"] = self._next_event_id()
       # Send as JSON array or individual messages depending on client support
       for delta in deltas:
           raw = json.dumps(delta, separators=(",", ":"))
           await self._ws.send(raw)
   ```

2. Alternativa mais simples: usar `orjson` como drop-in para `json.dumps` (~5x mais rápido).

**Validação:**
- Benchmark antes/depois com `pytest-benchmark` ou manual
- Client React continua recebendo e reproduzindo áudio normalmente

---

### 3.9 Construtor de `RealtimeSession` — mover trabalho real

**Trigger:** quando testes de session ficarem difíceis por causa de setup no `__init__`.

**Problema:** `session.py:89-93` faz fork do ToolRegistry e importa `register_recall_handler` dentro do `__init__`, dificultando teste isolado.

**Arquivos afetados:**
- `src/api/server/session.py` — linhas 89-93

**O que fazer:**

1. Aceitar `ToolRegistry` já forkado como parâmetro do construtor:
   ```python
   def __init__(self, ws, asr, llm, tts, tools: ToolRegistry, config: SessionConfig, ...):
       self._tools = tools  # já forkado pelo caller
   ```

2. Mover o fork + register para `ws_server.py` (ou onde a session é criada):
   ```python
   session_tools = global_tools.fork()
   register_recall_handler(session_tools, memory)
   session = RealtimeSession(ws, asr, llm, tts, tools=session_tools, ...)
   ```

**Validação:**
- Testes podem injetar ToolRegistry mock sem side effects
- Comportamento em produção inalterado

---

## Ordem de Execução Recomendada

```
Fase 1 (paralelo, independente):
  1.1 Fix métricas TTFT ──────┐
  1.2 Extract tool stream ────┤── podem ser feitos em paralelo
  1.3 Remove dead code ───────┤
  1.4 Consolidar resample ────┘

Fase 2 (sequencial, com dependências):
  2.5 Config → dataclasses (primeiro, pois outros items importam config)
  2.4 Extract gRPC boilerplate (depende de 2.5 para common/config)
  2.3 Health check dinâmico (depende de 2.4 para GrpcMicroservice)
  2.2 Mover generate_sentences (independente)
  2.1 Decompor response_runner (independente, mas maior — fazer por último)

Fase 3 (sob demanda, order doesn't matter):
  3.1 Request ID propagation
  3.2 gRPC retry
  3.3 VAD state machine
  3.4 ResponseMetrics dataclass
  3.5 web_search cleanup
  3.6 MockTTS numpy
  3.7 deque em session
  3.8 Audio delta batching
  3.9 Session constructor cleanup
```

---

## Métricas de Sucesso

| Métrica | Antes | Depois (Fase 2) |
|---------|-------|-----------------|
| Maior arquivo | `response_runner.py` 1086 linhas | ~750 linhas |
| `llm_ttft_ms` Anthropic | Sempre 0.0 | Valor real |
| Linhas duplicadas OpenAI/vLLM | ~40 | 0 |
| Health check accuracy | Sempre SERVING | Reflete estado real |
| gRPC boilerplate duplicado | ~200 linhas | 0 (shared) |
| Config type safety | Dict (runtime KeyError) | Dataclass (AttributeError + IDE) |
