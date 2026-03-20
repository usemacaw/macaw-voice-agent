# Plano: Pipeline End-to-End Streaming

> Eliminar todos os gargalos batch entre Mic e Speaker.
> Meta: primeiro áudio de resposta **antes do usuário terminar de falar**.

---

## 1. Objetivo

Transformar o pipeline do Macaw Voice Agent de **semi-streaming** (batch em ASR e sentence split) para **streaming real end-to-end**, onde:

1. O ASR emite transcrições parciais **enquanto o usuário ainda fala**
2. O LLM começa a gerar **antes do usuário terminar** (com as primeiras palavras)
3. O TTS emite áudio **frame a frame** conforme o LLM gera tokens (já implementado)

```
ANTES (atual):
  Mic ──[stream]──▶ VAD ──[BATCH: espera silêncio]──▶ ASR ──[BATCH: transcreve tudo]──▶ LLM ──[BATCH: acumula frase]──▶ TTS ──▶ Speaker

DEPOIS (meta):
  Mic ──[stream]──▶ VAD+ASR ──[stream: parciais]──▶ LLM ──[stream: tokens]──▶ TTS ──[stream: frames]──▶ Speaker
                         ↑                              ↑                          ↑
                    Parciais a cada              Dispara com ≥3          Frame a frame
                    200-500ms                    palavras                 (TTFA 64ms)
```

**Métrica de sucesso:** Tempo entre o usuário **terminar** de falar e o **primeiro áudio** de resposta ser reproduzido (E2E latency) reduzido de ~1500ms para **≤500ms**.

---

## 2. Diagnóstico: Os 5 Gargalos Batch

| # | Gargalo | Localização | Latência adicionada | Prioridade |
|---|---------|-------------|--------------------:|:---:|
| **B1** | VAD espera silêncio completo | `audio_input.py:233` | ~200ms (silence persistence) | Alta |
| **B2** | ASR pseudo-streaming (re-transcreve tudo) | `audio_input.py:268` | ~300-1000ms (Whisper full) | **Crítica** |
| **B3** | Transcript completo antes do LLM | `session.py:174` | Bloqueado por B1+B2 | Alta |
| **B4** | Acumula ≥20 chars + clause break | `sentence_splitter.py:60` | ~200-500ms (depende do LLM) | Média |
| **B5** | TTS recebe frase completa | `sentence_pipeline.py:254` | Eliminado pelo streaming TTS | ✅ Feito |

**Ordem de ataque:** B2 → B1 → B3 → B4 (B5 já resolvido)

---

## 3. Técnicas

### 3.1 ASR Streaming Real (elimina B2)

**Técnica: Streaming ASR com emissão de parciais a cada chunk**

O ASR atual (Whisper + LocalAgreement) re-transcreve o áudio completo a cada ~1000ms. Para streaming real, precisamos de um ASR que emita tokens/palavras incrementalmente.

**Opções avaliadas:**

| ASR | Streaming Real | Latência | PT-BR | Self-hosted |
|-----|:-:|---:|:-:|:-:|
| **Whisper Streaming (faster-whisper)** | ✅ (com LocalAgreement melhorado) | ~500ms entre parciais | ✅ | ✅ |
| **Deepgram Nova-3** | ✅ (nativo) | ~200ms entre parciais | ✅ | ❌ (API) |
| **Groq Whisper** | ✅ (nativo) | ~100ms | ✅ | ❌ (API) |
| **Moonshine** | ✅ (causal) | ~200ms | ⚠️ | ✅ |
| **Whisper Turbo + VAD chunking** | ⚠️ (pseudo) | ~300ms | ✅ | ✅ |

**Escolha recomendada:** Melhorar o Whisper streaming existente com emissão de parciais mais frequente (cada 300ms em vez de 1000ms) + adicionar suporte a Deepgram/Groq como providers de streaming real.

**Implementação:**

```python
# Nova interface ASR — adiciona callback de parciais
class ASRProvider(ABC):
    # Existente
    async def transcribe(self, audio: bytes) -> str: ...
    async def start_stream(self, stream_id: str) -> None: ...
    async def feed_chunk(self, audio: bytes, stream_id: str) -> None: ...
    async def finish_stream(self, stream_id: str) -> str: ...

    # NOVO: callback de parciais
    async def feed_chunk_with_partial(
        self, audio: bytes, stream_id: str
    ) -> Optional[str]:
        """Feed audio chunk and return partial transcript if available.

        Returns None if no new partial, or the updated partial text.
        Called every ~100ms with new audio.
        """
        return None  # Default: sem parciais (backward compat)

    @property
    def supports_partial_results(self) -> bool:
        return False
```

### 3.2 Desacoplar VAD de Transcript (elimina B1)

**Técnica: VAD como detector de estado, não como gate de transcrição**

Atualmente, o VAD **bloqueia** a transcrição até detectar silêncio. No modelo streaming, o VAD serve apenas para:
1. Detectar início de fala → ativar ASR stream
2. Detectar fim de fala → finalizar ASR stream + confirmar transcript

**Mudança:**

```
ANTES:
  VAD speech_started → acumula áudio → VAD speech_stopped → ASR.transcribe(todo_audio)

DEPOIS:
  VAD speech_started → ASR.start_stream()
  Cada chunk de áudio → ASR.feed_chunk_with_partial() → emite parcial se disponível
  VAD speech_stopped → ASR.finish_stream() → confirma final
```

**Implementação em `audio_input.py`:**

```python
async def _feed_audio_to_asr(self, pcm_8k: bytes) -> None:
    """Feed audio to ASR during speech and emit partials."""
    if not self._is_speaking or not self._asr.supports_partial_results:
        return

    partial = await self._asr.feed_chunk_with_partial(pcm_8k, self._asr_stream_id)
    if partial and partial != self._last_partial:
        self._last_partial = partial
        # Emite evento de transcript parcial ao cliente
        await self._emitter.emit(
            events.input_audio_transcription_delta(self._pending_speech_item_id, partial)
        )
        # Notifica session para possível trigger do LLM
        if self._on_partial_transcript:
            await self._on_partial_transcript(self._pending_speech_item_id, partial)
```

### 3.3 LLM Triggered por Parciais (elimina B3)

**Técnica: Disparo antecipado do LLM com primeiras palavras**

Em vez de esperar o transcript completo, disparamos o LLM quando temos confiança suficiente nas primeiras palavras.

**Estratégia: "Speculative Response"**

```
Parcial 1: "Qual é"           → Muito curto, espera
Parcial 2: "Qual é o saldo"   → ≥3 palavras estáveis → DISPARA LLM
Parcial 3: "Qual é o saldo da minha conta"  → Atualiza contexto (LLM já rodando)
Final:     "Qual é o saldo da minha conta corrente?"  → Confirma ou corrige
```

**Regras de disparo:**
- ≥3 palavras estáveis (apareceram em 2 parciais consecutivos)
- OU ≥5 palavras em qualquer parcial (alta confiança)
- OU parcial contém keyword de intenção (ex: "saldo", "transferir", "ajuda")

**Implementação em `session.py`:**

```python
async def _on_partial_transcript(self, item_id: str, partial: str) -> None:
    """Handle partial ASR transcript — may trigger early LLM."""
    words = partial.strip().split()

    # Verificar estabilidade (mesmo prefixo que parcial anterior)
    stable_words = self._count_stable_words(partial)

    if stable_words >= 3 and not self._response_started:
        logger.info(f"Early LLM trigger: {stable_words} stable words: '{partial}'")
        # Cria item de conversa com transcript parcial
        item = ConversationItem(
            id=item_id,
            role="user",
            content=[ContentPart(type="input_audio", transcript=partial)],
            status="in_progress",  # Não "completed" — pode ser atualizado
        )
        self._conversation.upsert(item)
        await self._start_response()
        self._response_started = True
```

**Tratamento de correção:**
Se o transcript final difere significativamente do parcial que trigou o LLM:
1. Se o LLM já gerou resposta → manter (a resposta provavelmente está correta)
2. Se a diferença é grande (>50% das palavras mudaram) → cancelar resposta e re-gerar

### 3.4 Reduzir Latência do Sentence Split (melhora B4)

**Técnica: "Eager First Token" — enviar ao TTS mais cedo**

O sentence splitter já tem lógica de "eager first sentence" (≥20 chars + clause break). Podemos ser mais agressivos:

```python
# Atual: min_eager_chars = 20
# Proposta: min_eager_chars = 10, e aceitar vírgula como delimitador
```

**Ou melhor: Token-level TTS streaming**

O Qwen3-TTS suporta Dual-Track (texto + áudio tokens em paralelo). Em teoria, poderíamos alimentar tokens LLM diretamente ao TTS sem esperar frases. Mas isso requer mudanças na arquitetura do TTS que são complexas.

**Recomendação pragmática:** Reduzir `min_eager_chars` de 20 para 10 e adicionar mais delimitadores. Ganho estimado: ~100-200ms na primeira frase.

---

## 4. Plano de Execução

### Fase 1: ASR Streaming Real (Semana 1-2)

| # | Task | Arquivo | Ambiente | Complexidade |
|---|------|---------|----------|:---:|
| 1.1 | Adicionar `feed_chunk_with_partial()` e `supports_partial_results` ao ABC | `src/api/providers/asr.py` | 🖥️ CPU | Baixa |
| 1.2 | Implementar parciais no Whisper provider (reduzir intervalo para 300ms) | `src/stt/providers/whisper_stt.py` | 🟢 GPU | Média |
| 1.3 | Atualizar gRPC proto para TranscribeStream retornar parciais | `src/shared/proto/stt_service.proto` | 🖥️ CPU | Baixa |
| 1.4 | Atualizar RemoteASR para consumir parciais via gRPC stream | `src/api/providers/asr_remote.py` | 🖥️ CPU | Média |
| 1.5 | Testes unitários do ASR com parciais (mocks) | `src/api/tests/` | 🖥️ CPU | Média |
| 1.6 | **⚠️ GPU:** Teste de integração com Whisper real | — | 🟢 GPU | Alta |

### Fase 2: Desacoplar VAD + Emitir Parciais (Semana 2-3)

| # | Task | Arquivo | Ambiente | Complexidade |
|---|------|---------|----------|:---:|
| 2.1 | Refatorar `audio_input.py`: feed_audio → ASR durante fala | `src/api/server/audio_input.py` | 🖥️ CPU | Alta |
| 2.2 | Adicionar callback `on_partial_transcript` ao AudioInputHandler | `src/api/server/audio_input.py` | 🖥️ CPU | Média |
| 2.3 | Emitir `input_audio_transcription_delta` events | `src/api/protocol/events.py` | 🖥️ CPU | Baixa |
| 2.4 | Testes com FakeASR que emite parciais | `src/api/tests/` | 🖥️ CPU | Média |
| 2.5 | Frontend: exibir transcript parcial em tempo real | `src/web/` | 🖥️ CPU | Média |

### Fase 3: LLM Trigger Antecipado (Semana 3-4)

| # | Task | Arquivo | Ambiente | Complexidade |
|---|------|---------|----------|:---:|
| 3.1 | Implementar `_on_partial_transcript` em session.py | `src/api/server/session.py` | 🖥️ CPU | Alta |
| 3.2 | Contagem de palavras estáveis (2 parciais consecutivos) | `src/api/server/session.py` | 🖥️ CPU | Média |
| 3.3 | Conversation upsert para items "in_progress" | `src/api/pipeline/conversation.py` | 🖥️ CPU | Média |
| 3.4 | Tratamento de correção (transcript final ≠ parcial) | `src/api/server/session.py` | 🖥️ CPU | Alta |
| 3.5 | Reduzir `min_eager_chars` de 20 para 10 | `src/api/pipeline/sentence_splitter.py` | 🖥️ CPU | Baixa |
| 3.6 | Testes E2E: parcial → LLM trigger → TTS streaming | `src/api/tests/` | 🖥️ CPU | Alta |
| 3.7 | **⚠️ GPU:** Benchmark E2E completo | — | 🟢 GPU | Alta |

### Fase 4: Otimização e Edge Cases (Semana 4)

| # | Task | Arquivo | Ambiente | Complexidade |
|---|------|---------|----------|:---:|
| 4.1 | Barge-in durante resposta especulativa | `src/api/server/session.py` | 🖥️ CPU | Média |
| 4.2 | Rate limiting de parciais (max 5/s) | `src/api/server/audio_input.py` | 🖥️ CPU | Baixa |
| 4.3 | Métricas: `asr_partial_count`, `early_trigger_ms`, `correction_count` | `src/api/protocol/events.py` | 🖥️ CPU | Baixa |
| 4.4 | Config: `ENABLE_EARLY_LLM_TRIGGER`, `MIN_STABLE_WORDS`, `PARTIAL_INTERVAL_MS` | `src/api/config.py` | 🖥️ CPU | Baixa |
| 4.5 | **⚠️ GPU:** Teste de dogfooding completo | — | 🟢 GPU | Alta |

---

## 5. Testes

### 5.1 Testes Unitários (CPU, mocks)

```python
# test_asr_partial.py
class TestASRPartialTranscripts:
    """Testa emissão de parciais pelo ASR."""

    async def test_feed_chunk_returns_partial(self):
        """ASR retorna parcial após N chunks."""
        asr = FakeASRWithPartials(partials=["Qual", "Qual é", "Qual é o saldo"])
        await asr.start_stream("s1")
        p1 = await asr.feed_chunk_with_partial(AUDIO_CHUNK, "s1")
        assert p1 == "Qual"
        p2 = await asr.feed_chunk_with_partial(AUDIO_CHUNK, "s1")
        assert p2 == "Qual é"

    async def test_partial_not_emitted_when_unchanged(self):
        """Não emite parcial se texto não mudou."""
        asr = FakeASRWithPartials(partials=["Olá", "Olá", "Olá tudo bem"])
        await asr.start_stream("s1")
        p1 = await asr.feed_chunk_with_partial(AUDIO_CHUNK, "s1")
        assert p1 == "Olá"
        p2 = await asr.feed_chunk_with_partial(AUDIO_CHUNK, "s1")
        assert p2 is None  # Sem mudança

    async def test_finish_stream_returns_final(self):
        """finish_stream retorna transcript final correto."""
        asr = FakeASRWithPartials(partials=["Qual é"], final="Qual é o saldo?")
        await asr.start_stream("s1")
        await asr.feed_chunk_with_partial(AUDIO_CHUNK, "s1")
        final = await asr.finish_stream("s1")
        assert final == "Qual é o saldo?"


# test_early_llm_trigger.py
class TestEarlyLLMTrigger:
    """Testa disparo antecipado do LLM."""

    async def test_trigger_after_3_stable_words(self):
        """LLM é disparado quando 3 palavras são estáveis."""
        session = create_test_session()
        # Simula 2 parciais com mesmo prefixo
        await session._on_partial_transcript("item1", "Qual é o")
        assert not session._response_started
        await session._on_partial_transcript("item1", "Qual é o saldo")
        # "Qual é o" apareceu em 2 parciais → 3 palavras estáveis
        assert session._response_started

    async def test_no_trigger_with_unstable_words(self):
        """LLM não dispara se palavras mudam entre parciais."""
        session = create_test_session()
        await session._on_partial_transcript("item1", "Qual é o")
        await session._on_partial_transcript("item1", "Quando é a")  # Mudou!
        assert not session._response_started  # Sem palavras estáveis

    async def test_correction_cancels_response(self):
        """Resposta cancelada se transcript final difere muito."""
        session = create_test_session()
        await session._on_partial_transcript("item1", "Qual é o saldo")
        # LLM trigou
        assert session._response_started
        # Transcript final muito diferente
        await session._on_final_transcript("item1", "Quando é a reunião?")
        # Resposta cancelada e re-gerada
        assert session._response_cancelled


# test_pipeline_e2e_streaming.py
class TestE2EStreamingPipeline:
    """Testa o pipeline completo em modo streaming."""

    async def test_first_audio_before_speech_ends(self):
        """Primeiro áudio de resposta chega ANTES do usuário parar de falar."""
        session = create_test_session(
            asr=FakeASRWithPartials(partials=["Qual é o saldo"], final="Qual é o saldo da conta?"),
            llm=FakeLLM(response="Seu saldo é R$ 1.500,00."),
            tts=FakeStreamingTTS(),
        )

        events = []
        # Simula áudio de fala (3 segundos)
        for i in range(30):  # 30 chunks de 100ms
            await session.handle_audio_chunk(AUDIO_CHUNK_100MS)
            events.extend(session.drain_events())

        # Verifica que áudio de resposta começou ANTES do speech_stopped
        audio_events = [e for e in events if e["type"] == "response.audio.delta"]
        speech_stopped = [e for e in events if e["type"] == "input_audio_buffer.speech_stopped"]

        assert len(audio_events) > 0, "Deve ter áudio de resposta"
        # Se speech_stopped não aconteceu, áudio veio antes
        # Se aconteceu, verificar timestamp
```

### 5.2 Testes de Integração (GPU)

```python
# test_integration_streaming.py (requer GPU + modelo real)

async def test_e2e_latency_with_real_models():
    """Mede latência E2E com modelos reais."""
    # Setup
    asr = WhisperSTT(model="large-v3-turbo")
    llm = AnthropicLLM(model="claude-sonnet")
    tts = MacawStreamingTTS()

    # Grava áudio de teste: "Qual é o saldo da minha conta?"
    test_audio = load_test_audio("qual_e_o_saldo.wav")

    t_start = time.perf_counter()

    # Feed áudio em chunks de 100ms
    session = create_session(asr, llm, tts)
    first_audio_time = None
    speech_end_time = None

    for chunk in chunk_audio(test_audio, 100):  # 100ms chunks
        await session.handle_audio_chunk(chunk)

        events = session.drain_events()
        for event in events:
            if event["type"] == "input_audio_buffer.speech_stopped":
                speech_end_time = time.perf_counter()
            if event["type"] == "response.audio.delta" and first_audio_time is None:
                first_audio_time = time.perf_counter()

    # Métricas
    e2e_latency = (first_audio_time - speech_end_time) * 1000  # ms após speech_stopped
    total_latency = (first_audio_time - t_start) * 1000  # ms total

    print(f"E2E latency (after speech): {e2e_latency:.0f}ms")
    print(f"Total latency (from start): {total_latency:.0f}ms")

    assert e2e_latency < 500, f"E2E latency too high: {e2e_latency}ms (target: <500ms)"
```

### 5.3 Benchmark de Latência

```python
# benchmark_e2e.py
"""
Mede cada estágio do pipeline com timestamps precisos.

Output esperado:
  Speech duration:     2500ms
  ASR first partial:    300ms  (após início da fala)
  ASR final:           2700ms  (200ms após speech_stopped)
  LLM trigger:          350ms  (com early trigger)  ← META: durante a fala
  LLM first token:      500ms  (150ms após trigger)
  LLM first sentence:   800ms
  TTS first audio:      864ms  (64ms após primeira frase)
  ─────────────────────────────
  E2E (speech_end → audio): 164ms  ← META: <500ms
  E2E (speech_start → audio): 864ms
"""
```

---

## 6. Definição de DONE

### DONE para cada fase:

**Fase 1 — ASR Streaming Real:**
- [ ] `feed_chunk_with_partial()` retorna parciais a cada ~300ms
- [ ] Parciais são emitidos via gRPC streaming (não re-transcrição completa)
- [ ] `input_audio_transcription_delta` event emitido ao cliente
- [ ] Testes unitários passando com FakeASR + parciais
- [ ] Teste de integração com Whisper real no GPU: parciais chegam em <500ms

**Fase 2 — VAD Desacoplado:**
- [ ] Áudio é enviado ao ASR durante a fala (não apenas após speech_stopped)
- [ ] Parciais fluem do ASR → AudioInputHandler → Session → Cliente
- [ ] Frontend exibe transcript parcial em tempo real (texto aparecendo gradualmente)
- [ ] VAD `speech_stopped` apenas confirma/finaliza o transcript

**Fase 3 — LLM Early Trigger:**
- [ ] LLM é disparado quando ≥3 palavras estáveis no transcript parcial
- [ ] Resposta começa a ser gerada **enquanto o usuário ainda fala**
- [ ] Se transcript final difere >50% → resposta cancelada e re-gerada
- [ ] `min_eager_chars` reduzido de 20 para 10
- [ ] E2E latency (speech_end → first_audio) ≤ 500ms

**Fase 4 — Otimização:**
- [ ] Barge-in funciona durante resposta especulativa
- [ ] Config flags para habilitar/desabilitar early trigger
- [ ] Métricas de parciais no `macaw.metrics` event
- [ ] Dogfooding: conversa natural sem artifacts perceptíveis

### DONE global — Pipeline E2E Streaming:

```
✅ DONE quando TODOS estes critérios forem verdadeiros:

1. ASR emite parciais a cada ~300ms durante a fala do usuário
2. LLM começa a gerar resposta ANTES do usuário parar de falar
3. TTS emite primeiro frame de áudio em ≤64ms após primeira frase do LLM
4. E2E latency (speech_end → first_audio) ≤ 500ms medido em benchmark
5. Conversa natural de dogfooding sem artifacts perceptíveis
6. Testes unitários: ≥15 novos testes passando
7. Teste de integração E2E passando no GPU
8. Feature controlada por config flag (ENABLE_EARLY_LLM_TRIGGER=true/false)
```

---

## 7. Riscos

| Risco | Probabilidade | Impacto | Mitigação |
|-------|:---:|:---:|---|
| Parciais instáveis geram trigger falso do LLM | Alta | Alto | Contagem de palavras estáveis (2 parciais consecutivos) |
| LLM gera resposta errada com parcial incompleto | Média | Alto | Cancelar e re-gerar se final difere >50% |
| Latência do Whisper streaming ainda alta | Média | Médio | Fallback: usar Deepgram/Groq para streaming real |
| Barge-in durante resposta especulativa causa confusão | Média | Médio | Cancelar resposta limpa + clear audio buffer |
| Overhead de parciais degrada performance | Baixa | Baixo | Rate limit: max 5 parciais/segundo |

---

## 8. Estimativa de Impacto

### Latência atual (medida):
```
Speech duration:        ~2000ms
VAD silence detect:      +200ms
ASR (Whisper full):      +500ms
LLM TTFT:               +200ms
Sentence accumulation:   +300ms
TTS TTFA:                 +64ms
────────────────────────────────
Total E2E:              ~3264ms
Após speech_end:        ~1264ms
```

### Latência projetada com E2E streaming:
```
Speech duration:        ~2000ms
ASR parcial (3 palavras): @500ms do início da fala
LLM trigger:             @550ms (50ms após parcial)
LLM TTFT:               +200ms → @750ms
Sentence (10 chars):    +150ms → @900ms
TTS TTFA:                +64ms → @964ms
────────────────────────────────────────
Primeiro áudio:          @964ms (DURANTE a fala do usuário!)
Após speech_end:         ~0ms (resposta já estava fluindo)
```

**Ganho: de 1264ms para ~0ms após speech_end** — porque a resposta já começou durante a fala.
