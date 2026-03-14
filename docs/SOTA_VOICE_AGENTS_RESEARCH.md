# SOTA Voice-to-Voice Agents — Deep Research

Pesquisa sobre o estado da arte em agentes de voz real-time (voice-to-voice), cobrindo arquiteturas, modelos, técnicas de otimização de latência, e benchmarks de produção.

Data: 2026-03-14

---

## 1. Arquiteturas de Voice Agents

Existem três paradigmas arquiteturais, cada um com trade-offs distintos:

### 1.1 Pipeline Cascata (STT → LLM → TTS)

```
Mic → VAD → ASR → LLM → TTS → Speaker
         sequencial, cada estágio espera o anterior
```

| Aspecto | Valor |
|---------|-------|
| Latência típica | 800–2000ms |
| Flexibilidade | Alta (swap qualquer componente) |
| Custo | ~$0.15/min (estável) |
| Qualidade TTS | Máxima (modelos especializados) |
| Tool calling | Nativo (LLM text-based) |

**Quando usar:** PSTN/telefonia (8kHz degrada S2S), custo importa, tool calling é crítico, MVP.

### 1.2 Speech-to-Speech (Half-Cascade)

```
Mic → Audio Encoder → LLM (audio-native) → TTS → Speaker
         o LLM recebe audio tokens diretamente, elimina ASR
```

| Aspecto | Valor |
|---------|-------|
| Latência típica | 200–600ms |
| Flexibilidade | Baixa (modelo monolítico) |
| Custo | ~10x pipeline (~$0.30–1.50/min, escala com contexto) |
| Qualidade TTS | Inferior a TTS dedicado |
| Tool calling | Limitado |

**Modelos:** GPT-4o Realtime (~280ms TTFT), Gemini 2.5 Flash Live (~280ms TTFT), Ultravox v0.7 (~190ms TTFT)

**Quando usar:** latência ultra-baixa essencial, budget alto, preservar tom emocional.

### 1.3 Native Audio (End-to-End)

```
Mic → Modelo Unificado → Speaker
         um único modelo processa e gera áudio diretamente
```

| Aspecto | Valor |
|---------|-------|
| Latência típica | 160–250ms |
| Flexibilidade | Mínima |
| Custo | Alto (treinamento massivo) |
| Qualidade | Raciocínio opaco (sem camada de texto) |
| Full-duplex | Sim |

**Modelos:** Moshi (Kyutai, 160ms TTFT, 7B params), Gemini 2.5 Flash Native Audio (~200ms)

**Quando usar:** pesquisa, latência absoluta mínima, full-duplex necessário.

### 1.4 Comparação Direta

| Métrica | Pipeline | Speech-to-Speech | Native Audio |
|---------|----------|-------------------|--------------|
| TTFT | 400–800ms | ~280ms | ~160–250ms |
| E2E | 800–2000ms | 300–600ms | 200–400ms |
| Tool calling | Nativo | Limitado | Nenhum |
| Custo/min | $0.15 | $0.30–1.50 | Infra própria |
| Flexibilidade | Alta | Baixa | Mínima |
| Produção-ready | Sim | Sim | Experimental |

---

## 2. Modelos SOTA por Componente

### 2.1 ASR (Speech-to-Text)

| Modelo | Tipo | Latência | Streaming | Destaques |
|--------|------|----------|-----------|-----------|
| **Whisper large-v3-turbo** | Open-source | ~87ms (2s áudio) | Não (batch) | Melhor WER open-source |
| **Parakeet-tdt-0.6B** (NVIDIA) | Open-source | Rápido | Não | "Hard to beat" em latência+acurácia |
| **Deepgram Nova-3** | API | ~150ms | Sim | $0.0043/min, multi-idioma |
| **AssemblyAI Universal-Streaming** | API | ~300ms imutável | Sim | Semantic endpointing nativo |
| **Soniox v4** | API | Ultra-baixo | Sim | StartOfTurn/EndOfTurn nativos |
| **Qwen3-ASR-1.7B** | Open-source | ~49ms (batch) | Sim (vLLM) | Multilíngue, streaming via vLLM |
| **SpecASR** | Técnica | 3–3.8x speedup | Sim | Speculative decoding para ASR |

**Insight SOTA:** Parakeet + VAD local (não streaming) pode ser mais rápido que streaming STT, porque o tempo do transcript final é o que importa, não os parciais.

### 2.2 LLM (Geração de Texto)

| Modelo | TTFT | Tokens/s | Tool Calling | Notas |
|--------|------|----------|--------------|-------|
| **GPT-4o-mini** | ~400ms | Alto | Sim | Balanço custo/velocidade |
| **Gemini 2.5 Flash** | ~400ms | Alto | Sim | 10x mais barato que GPT-4o para áudio |
| **Claude 4.5 Haiku** | ~360ms | Alto | Sim | Rápido, bom para voz |
| **Groq Scout** | ~80ms | Muito alto | Sim | TTFT mais baixo disponível |
| **Qwen3-8B-AWQ** (vLLM) | ~150ms* | ~80 tok/s | Sim (Hermes) | Self-hosted, 4-bit, ~5GB VRAM |
| **TSLAM-Mini-2B** (4-bit) | ~106ms | ~80 tok/s | Não | Telecom-otimizado |

*Depende do hardware. Em A100: ~100–150ms.

**Insight SOTA:** LLM é responsável por ~70% da latência total. TTFT é a métrica mais crítica, não throughput. Modelos menores (2–8B) com quantização 4-bit reduzem TTFT em até 40%.

### 2.3 TTS (Text-to-Speech)

| Modelo | TTFB | Streaming | Qualidade | Notas |
|--------|------|-----------|-----------|-------|
| **Cartesia Sonic 3** | ~40ms | Sim | Alta | Ultra-baixa latência, WebSocket |
| **Smallest.ai Lightning** | 300–400ms | Sim | Boa | Pronúncia multilíngue |
| **Kokoro 82M** | ~143ms | Sim | Boa | Open-source, streaming, leve |
| **ElevenLabs** | 75ms (API) | Sim | Muito alta | Mais natural, mais caro |
| **Deepgram Aura** | ~100ms | Sim | Boa | Baixo custo |
| **T-SYNTH** | ~286ms | Sim | Telecom | Otimizado para telefonia |

**Insight SOTA:** WebSocket pré-conectado economiza ~200ms vs HTTP por request. Streaming TTS reduz TTFB em 50–70%. Cartesia Sonic com ~40ms TTFB é o mais rápido disponível.

### 2.4 VAD e Turn Detection

| Abordagem | Latência | Acurácia | Notas |
|-----------|----------|----------|-------|
| **Silero VAD** (silence-based) | 200–800ms | Média | Padrão, configurável, ONNX |
| **LiveKit Semantic** (text-only) | <25ms inferência | Alta | Depende de VAD upstream |
| **Pipecat SmartTurn** (audio) | Baixa | Alta | Prosódia + padrões vocais |
| **AssemblyAI Semantic** (hybrid) | 160ms min silence | Muito alta | Texto + áudio, imutável |
| **Soniox Flux** (unified) | -200–600ms vs pipeline | Muito alta | VAD + endpointing + STT em um modelo |

**Insight SOTA:** Semantic endpointing reduz falsos positivos em ~30% vs silence-only. O endpointing é onde se ganha mais latência perceptual — mais importante que otimizações incrementais em ASR/TTS.

---

## 3. Técnicas de Otimização (Ranking por Impacto)

### Tier 1 — Alto Impacto (300–600ms de redução)

#### 3.1 Streaming End-to-End
```
Sem streaming:  ASR [====] → LLM [========] → TTS [======] → Audio
                                                              ↑ primeiro áudio

Com streaming:  ASR [====] → LLM [==
                              ↓ 1ª frase
                              TTS [==] → Audio
                                         ↑ primeiro áudio (muito antes)
```
- ASR streaming: -100–200ms
- LLM sentence streaming: primeiro áudio antes do LLM terminar
- TTS streaming: -200–400ms no TTFB

#### 3.2 Semantic Turn Detection
- Substituir silence threshold fixo (800ms) por modelo semântico
- Reduz silence detection de 800ms → 160–400ms
- Elimina falsos positivos (interrupções mid-sentence)
- **Configuração ótima por caso de uso:**

| Caso | Silence | Threshold | Min Speech |
|------|---------|-----------|------------|
| Q&A rápido | 400ms | 0.6 | 50ms |
| Conversa | 500ms | 0.5 | 100ms |
| Queries pensadas | 800ms | 0.4 | 150ms |
| Ambiente ruidoso | 600ms | 0.7 | 200ms |

#### 3.3 Seleção de Modelo LLM
- Trocar modelo premium por fast-tier: -200–600ms
- Groq Scout (80ms TTFT) vs GPT-4o (400–600ms TTFT) = -400ms
- Quantização 4-bit: -40% latência, >95% qualidade
- `max_tokens` limitado (60–80): primeira frase = resposta inteira

### Tier 2 — Médio Impacto (50–200ms cada)

#### 3.4 Co-localização Geográfica
- Todos os serviços no mesmo VPC/cluster: latência de rede → single-digit ms
- US East-West: +60–80ms, US-Europe: +80–150ms
- Containers de inferência próximos ao bot: mais importante que distância do cliente

#### 3.5 Connection Pooling e Pre-warming
- WebSocket pré-conectado para TTS: -200ms por request
- gRPC channels quentes: elimina cold-start de conexão
- TTS warmup routine: pré-carrega modelo e referência de voz
- HTTP keep-alive para vLLM: elimina handshake TCP/TLS

#### 3.6 Binary Serialization
- msgpack em vez de JSON entre LLM e TTS: -0.8–1.0s (paper telecom)
- Evita overhead de JSON.dumps() em eventos de alta frequência (audio.delta)

#### 3.7 Parallel Processing
- Guardrails em paralelo com LLM (não sequencial)
- RAG retrieval em paralelo com ASR finalization
- Pre-call APIs executadas concorrentemente no início da sessão

### Tier 3 — Otimizações Avançadas

#### 3.8 Speculative ASR
- Iniciar LLM com transcrição parcial, corrigir se mudar
- Funciona bem para frases previsíveis
- Risco: reprocessamento se ASR corrigir

#### 3.9 LLM Hedging
- Lançar 2 chamadas LLM em paralelo, usar a que responder primeiro
- Fallback automático se provider falhar
- Custo: 2x tokens, benefício: latência do mais rápido

#### 3.10 Response Caching
- Pré-sintetizar frases comuns (saudações, confirmações)
- Cache de áudio para respostas frequentes
- Resposta instantânea para greetings

#### 3.11 Audio Buffer Pooling
- Reutilizar buffers em vez de alocar novos
- memoryview para zero-copy ao longo do pipeline
- Ring buffer para VAD em vez de bytearray crescente

---

## 4. Benchmarks de Produção

### 4.1 Dados de Produção Real (4M+ chamadas)

| Percentil | Latência E2E |
|-----------|-------------|
| P50 (mediana) | 1.4–1.7s |
| P90 | 3.3–3.8s |
| P95 | 4.3–5.4s |
| P99 | 8.4–15.3s |

### 4.2 A Regra dos 300ms (Base Neurológica)

| Latência | Percepção Humana |
|----------|-----------------|
| <300ms | Instantâneo |
| 300–400ms | Estranhamento começa |
| >500ms | Usuário questiona se foi ouvido |
| >1000ms | Assume falha de conexão |
| >1500ms | Resposta de stress neurológico |

Gap médio entre turnos na conversa humana: **~200ms**.

### 4.3 Benchmarks de Referência por Arquitetura

| Sistema | Arquitetura | E2E | Hardware |
|---------|------------|-----|----------|
| **OpenAI Realtime API** | S2S (GPT-4o) | 300–500ms | Cloud |
| **Riya (Groq+Smallest.ai)** | Pipeline otimizado | 500–1100ms TTFR | Cloud |
| **Modal+Pipecat+Kokoro** | Pipeline open-source | ~1000ms mediana | GPU (Modal) |
| **Paper Telecom (H100)** | Pipeline (2B LLM) | 417–934ms | H100 80GB |
| **Moshi** | Native audio | ~200ms | GPU dedicada |
| **Cerebrium+Cartesia** | Pipeline managed | ~500ms | Cloud GPU |
| **Macaw (nosso, atual)** | Pipeline (Qwen3-8B) | ~800–1200ms* | A100 + remoto |

*Estimado baseado nos logs de métricas.

### 4.4 Targets Recomendados

| Métrica | Target SOTA | Produção OK | Nosso Target |
|---------|------------|-------------|--------------|
| ASR | <100ms | <200ms | <100ms |
| LLM TTFT | <150ms | <300ms | <150ms |
| LLM 1ª frase | <300ms | <500ms | <300ms |
| TTS TTFB | <100ms | <200ms | <150ms |
| Turn detection | <200ms | <400ms | <200ms |
| **E2E** | **<500ms** | **<800ms** | **<500ms** |

---

## 5. Frameworks e Plataformas

### 5.1 Open-Source

| Framework | Linguagem | Arquitetura | Destaques |
|-----------|-----------|-------------|-----------|
| **Pipecat** (Daily) | Python | Frame pipeline (Unix pipes) | 40+ providers, WebRTC, turn detection |
| **LiveKit Agents** | Python/Node | SFU + agent framework | WebRTC nativo, semantic turn, cloud deploy |
| **Macaw** (nosso) | Python | WebSocket + gRPC | OpenAI Realtime API compat, sentence pipeline |

### 5.2 Modelos End-to-End Open-Source

| Modelo | Params | Latência | Full-Duplex | Produção |
|--------|--------|----------|-------------|----------|
| **Moshi** (Kyutai) | 7B | 160ms | Sim | Gradium ($70M) |
| **Ultravox v0.7** (Fixie) | 8B (GLM) | 190ms TTFT | Não | Sim |
| **Mini-Omni2** | ~2B | Variável | Sim | Experimental |
| **VITA-1.5** | Variável | Baixa | Sim | Experimental |

---

## 6. Gaps Identificados no Macaw vs SOTA

### 6.1 O Que Já Temos (Bom)
- Sentence-level pipelining (LLM→TTS em paralelo)
- Server-side VAD (Silero)
- Streaming TTS (Kokoro com prefetch de 4 frases)
- Tool calling server-side com filler audio
- Métricas per-response (macaw.metrics)
- Provider plugável (registry pattern)

### 6.2 O Que Falta para SOTA

| Gap | Impacto | Técnica SOTA | Esforço |
|-----|---------|-------------|---------|
| **Semantic turn detection** | -200–400ms | Modelo semântico + VAD hybrid | Alto |
| **Streaming ASR** | -100–200ms | Partial→LLM while ASR continues | Médio |
| **Binary serialization** | -50–100ms | msgpack para eventos de áudio | Baixo |
| **LLM hedging/fallback** | Resiliência | 2 providers em paralelo | Médio |
| **Circuit breaker** | Resiliência | Provider fallback automático | Médio |
| **Connection pre-warming** | -50ms | gRPC/HTTP persistent warm | Baixo |
| **Audio buffer pooling** | -20–50ms | memoryview, zero-copy | Baixo |
| **WebRTC transport** | -100–300ms | Substituir WebSocket por WebRTC | Alto |
| **Distributed tracing** | Observabilidade | Request ID em todos os logs | Baixo |
| **Percentil metrics** | Observabilidade | P50/P95/P99 por sessão | Baixo |
| **Response caching** | -latência total | Cache de saudações/confirmações | Baixo |
| **Adaptive VAD** | Qualidade | Ajustar threshold por ruído ambiente | Médio |

### 6.3 Plano de Evolução Sugerido

**Fase 1 — Quick Wins (1–2 dias, -200–400ms)**
1. Binary serialization para audio.delta events
2. Connection pre-warming (gRPC channels quentes)
3. Audio buffer pooling (memoryview)
4. Response caching para saudações
5. Distributed tracing (request ID)
6. Percentil metrics (P50/P95/P99)

**Fase 2 — Structural (1–2 semanas, -200–400ms)**
1. Semantic turn detection (modelo hybrid)
2. Streaming ASR com early LLM trigger
3. Circuit breaker para providers
4. LLM hedging (dual provider)
5. Adaptive VAD thresholds

**Fase 3 — SOTA (2–4 semanas, -100–200ms)**
1. WebRTC transport (substituir WebSocket)
2. Speculative ASR (LLM com transcrição parcial)
3. Ultravox/Moshi evaluation (S2S para casos simples)
4. TTS com Cartesia Sonic (~40ms TTFB)
5. LLM KV-cache warming entre turns

---

## 7. Referências

### Papers Acadêmicos
- [Toward Low-Latency End-to-End Voice Agents for Telecommunications](https://arxiv.org/abs/2508.04721) — Pipeline ASR+RAG+LLM(2B)+TTS em H100, 417–934ms E2E
- [Moshi: a speech-text foundation model for real-time dialogue](https://arxiv.org/abs/2410.00037) — Full-duplex, 160ms, 7B params, inner monologue
- [Mini-Omni2: Towards Open-source GPT-4o](https://arxiv.org/abs/2410.11190) — Visão + fala + texto em modelo único
- [VITA-1.5: Towards GPT-4o Level Real-Time Vision and Speech](https://openreview.net/forum?id=8PUzLga3lU) — Multi-modal sem ASR/TTS separados
- [SpecASR: Accelerating LLM-based ASR via Speculative Decoding](https://arxiv.org/abs/2507.18181) — 3–3.8x speedup sem perda de acurácia
- [Can Speech LLMs Think while Listening?](https://arxiv.org/html/2510.07497v1) — Processamento simultâneo de fala e raciocínio

### Artigos Técnicos
- [The Voice AI Stack for Building Agents in 2026](https://www.assemblyai.com/blog/the-voice-ai-stack-for-building-agents) — AssemblyAI, arquiteturas e stack completo
- [Engineering for Real-Time Voice Agent Latency](https://cresta.com/blog/engineering-for-real-time-voice-agent-latency) — Cresta, otimizações de produção
- [Voice AI Latency: What's Fast, What's Slow](https://hamming.ai/resources/voice-ai-latency-whats-fast-whats-slow-how-to-fix-it) — Hamming AI, benchmarks de 4M+ chamadas
- [From 7 Seconds to 500ms: Voice Agent Optimization Secrets](https://dev.to/sundar_ramanganesh_1057a/from-7-seconds-to-500ms-the-voice-agent-optimization-secrets-4j9h) — Jornada de otimização passo a passo
- [One-Second Voice-to-Voice Latency with Modal, Pipecat, and Open Models](https://modal.com/blog/low-latency-voice-bot) — Pipeline open-source com Parakeet+Qwen3+Kokoro
- [Real-Time vs Turn-Based Voice Agent Architecture](https://softcery.com/lab/ai-voice-agents-real-time-vs-turn-based-tts-stt-architecture) — Comparação detalhada das 3 arquiteturas
- [How Intelligent Turn Detection Solves Voice Agent Development](https://www.assemblyai.com/blog/turn-detection-endpointing-voice-agent) — Semantic endpointing técnico

### Plataformas e Frameworks
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — GPT-4o native speech, 300–500ms
- [OpenAI gpt-realtime](https://openai.com/index/introducing-gpt-realtime/) — Modelo otimizado para voz
- [LiveKit Agents](https://github.com/livekit/agents) — Framework open-source, WebRTC, semantic turn
- [Pipecat](https://github.com/pipecat-ai/pipecat) — Framework open-source, 40+ providers
- [Ultravox](https://github.com/fixie-ai/ultravox) — Speech LLM, ~190ms TTFT, open-source
- [Moshi](https://github.com/kyutai-labs/moshi) — Full-duplex S2S, 160ms, open-source
- [Cerebrium Voice Agent](https://www.cerebrium.ai/blog/deploying-a-global-scale-ai-voice-agent-with-500ms-latency) — Deploy global com 500ms, $0.03/min
