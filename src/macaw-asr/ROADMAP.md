# Roadmap: Inferência Otimizada — macaw-asr

> Roadmap executável para inferência de alta performance com múltiplos modelos ASR.
> Cada milestone tem Definition of Done (DoD) verificável.
> Otimizações são **compartilhadas** via abstrações no engine/runner.
> Otimizações **model-specific** ficam dentro de cada `models/*.py`.

---

## Status

| Milestone | Status | Resultado |
|-----------|--------|-----------|
| M0 — Bugfixes | ✅ Done | EOS token, batch==streaming |
| M1 — Warmup | ✅ Done | Multi-shape warmup (0.5s, 1s, 3s) + fast_finish path |
| M2 — Multi-Model | ✅ Done | supports_streaming, supports_cuda_graphs, compilable_module(), strategy=None |
| M3 — Shared Optimizations | ✅ Done | torch.compile via compilable_module(), enable_compile config |
| M4 — Qwen-Specific Optimizations | ✅ Done | GPU mel as default (2ms vs 15ms), CPU fallback |
| M5 — Métricas Granulares | ✅ Done | prepare_ms, prefill_ms, decode_ms, decode_per_token_ms, total_ms + startup_timings |
| M6 — Benchmark Suite | ✅ Done | 102 testes reais, WER 28%, RTF 0.08-0.12 |
| M7 — torch.compile + CUDA Graphs | ✅ Done | compile mode=default OK, CUDA graphs incompatível (dynamic KV) |

---

## Milestone 1 — Warmup Framework (Genérico)

**Objetivo:** Warmup que funciona para qualquer modelo. Primeiro request não paga cold start.

### 1.1 Warmup Multi-Shape no ASRModel ABC

**Contexto:** Cada modelo tem shapes diferentes (Qwen: mel frames, Whisper: log-mel 80x3000, Parakeet: features). O warmup precisa exercitar os code paths reais do modelo, não apenas 1 shape fixa.

**Arquivos:** `models/base.py`, `runner/engine.py`

**DoD:**
- [ ] `ASRModel.warmup()` recebe `config: EngineConfig` (tem max_new_tokens, etc.)
- [ ] Default warmup no ABC: `prepare_inputs(1s) → generate(3 steps)` — funciona para qualquer modelo
- [ ] Modelos podem override para warmup específico (Qwen: + fast_finish, Whisper: + log-mel cache)
- [ ] `engine.start()` loga: `warmup_ms=X`
- [ ] Teste: `test_first_request_no_cold_start` — primeiro request ≤ 1.2x do segundo

### 1.2 Warmup do Fast Path (model-specific, opcional)

**DoD:**
- [ ] `QwenASRModel.warmup()`: prepare_inputs(0.5s, 1s, 3s) + fast_finish + generate com 3 shapes
- [ ] Futuro: `WhisperASRModel.warmup()`: encoder + decoder com áudios de 1s, 5s, 15s
- [ ] Tempo de warmup logado separado do load

---

## Milestone 2 — Multi-Model Support (Streaming + Batch)

**Objetivo:** Arquitetura suporta Qwen (streaming autoregressive), Whisper (batch encoder-decoder), Parakeet (CTC/TDT batch). Cada modelo implementa o mesmo ABC mas com padrões de inferência diferentes.

### 2.1 Refinar ASRModel ABC para Batch vs Streaming

**Contexto:** Hoje o ABC assume autoregressive (prepare_inputs → generate com DecodeStrategy). Mas:
- **Whisper**: encoder processa todo o áudio → decoder gera texto. `generate()` pode usar `model.generate()` do HuggingFace direto, sem manual decode loop.
- **Parakeet**: CTC/TDT — sem decode loop. Uma forward pass → logits → CTC decode. DecodeStrategy não se aplica.

**Arquivos:** `models/base.py`, `decode/strategies.py`

**DoD:**
- [ ] `ASRModel.generate()` aceita `strategy: DecodeStrategy | None` — None para modelos sem decode loop
- [ ] `ASRModel.supports_streaming: bool` property (default False) — indica se modelo tem streaming nativo
- [ ] Novo `ASRModel.transcribe_audio(audio) → str` method de conveniência que faz prepare + generate + postprocess internamente — para modelos batch-only que não precisam da pipeline granular
- [ ] Qwen: usa strategy (autoregressive) + supports_streaming=True
- [ ] Futuro Whisper: strategy=None (usa HF generate), supports_streaming=False
- [ ] Futuro Parakeet: strategy=None (CTC decode), supports_streaming=True (via chunked)

### 2.2 Decode Strategy Registry

**Contexto:** Hoje só existe `GreedyWithEarlyStopping`. Whisper precisa de beam search com language detection. Parakeet usa CTC greedy/beam. Estratégias devem ser registráveis como modelos.

**Arquivos:** `decode/strategies.py`

**DoD:**
- [ ] `GreedyWithEarlyStopping` — existente (autoregressive: Qwen)
- [ ] `CTCGreedyDecode` — para Parakeet (colapsa repetidos + blank)
- [ ] Engine seleciona strategy baseado no modelo ou config
- [ ] Modelos que não usam strategy (Whisper com HF generate) ignoram

### 2.3 Model-Aware Session

**Contexto:** `StreamingSession` assume que todo modelo suporta `prepare_inputs → generate` pipeline. Para modelos batch-only (Whisper), streaming = acumular todo o áudio e transcrever no finish. Para streaming nativo (Qwen, Parakeet), background precompute faz sentido.

**Arquivos:** `runner/session.py`, `runner/engine.py`

**DoD:**
- [ ] Session verifica `model.supports_streaming` antes de criar background tasks
- [ ] Modelos batch-only: push_audio só acumula, finish faz tudo
- [ ] Modelos streaming: push_audio + background precompute (como hoje)
- [ ] Engine seleciona behavior automaticamente baseado no modelo

---

## Milestone 3 — Shared Inference Optimizations

**Objetivo:** Otimizações que se aplicam a QUALQUER modelo transformer. Implementadas no engine/runner, não no model.

### 3.1 torch.compile (Genérico)

**Contexto:** `torch.compile()` funciona com qualquer `nn.Module`. Pode ser aplicado no load de qualquer modelo transformer.

**Arquivos:** `config.py`, `runner/engine.py`

**DoD:**
- [ ] `EngineConfig.enable_compile: bool = False` (opt-in)
- [ ] `engine.start()` aplica `torch.compile(model._internal_model, mode='reduce-overhead')` após load
- [ ] Cada modelo expõe `compilable_module() → nn.Module | None` — retorna o módulo compilável ou None
- [ ] Qwen: `compilable_module()` retorna `self._thinker`
- [ ] Futuro Whisper: retorna `self._model`
- [ ] Fallback: se compile falhar, log warning e continua
- [ ] Teste: benchmark com/sem compile, mesmos tokens gerados

### 3.2 CUDA Graphs (Autoregressive Only)

**Contexto:** CUDA Graphs só se aplicam a modelos com manual decode loop (Qwen). Whisper e Parakeet não têm decode loop — CUDA graphs não ajudam.

**Arquivos:** `models/base.py`

**DoD:**
- [ ] `ASRModel.supports_cuda_graphs: bool` property (default False)
- [ ] Qwen: True. Implementa capture + replay no generate()
- [ ] Engine não tenta CUDA graphs se modelo não suporta
- [ ] Static KV cache pre-alocado para models que suportam
- [ ] Teste: tokens com/sem CUDA graphs são idênticos

### 3.3 Background Precomputation (Genérico)

**Contexto:** Já funciona para Qwen. O padrão é genérico: durante streaming, roda `prepare_inputs()` em background enquanto áudio ainda chega. Funciona para qualquer modelo.

**Arquivos:** `runner/session.py` — já implementado

**DoD:**
- [ ] Validar que funciona com Whisper (batch: prepare_inputs no finish, não no background)
- [ ] Validar que funciona com Parakeet (streaming nativo: background precompute faz sentido)
- [ ] Nenhuma mudança de código necessária — session já verifica `enable_background_compute`

---

## Milestone 4 — Qwen-Specific Optimizations

**Objetivo:** Otimizações que só se aplicam ao Qwen3-ASR.

### 4.1 GPU-First Mel Spectrogram

**Contexto:** `prepare_inputs()` do Qwen usa CPU processor (~1500ms cold, ~15ms warm). `fast_finish_inputs()` usa GPU mel (~2ms). Fazer GPU mel o default elimina o CPU processor do hot path.

**Arquivos:** `models/qwen.py`

**DoD:**
- [ ] `_prepare_inputs_gpu(audio, prefix)` — GPU mel + tokenização CPU + audio_pad insertion
- [ ] `prepare_inputs()` usa GPU path como default, CPU como fallback
- [ ] Teste: GPU path produz mesma transcrição que CPU path
- [ ] Benchmark: prepare_inputs ≤ 10ms (vs ~15ms warm / ~1500ms cold)

### 4.2 Optimized Manual Decode Loop

**Contexto:** O decode loop do Qwen faz 32 forward passes step-by-step com KV cache. Otimizações possíveis: pre-alocar tensors, evitar torch.cat em cada step.

**Arquivos:** `models/qwen.py`

**DoD:**
- [ ] Pre-alocar `seqs` tensor com `max_new_tokens` posições extras
- [ ] Evitar `torch.cat([seqs, next_token])` — usar index assignment
- [ ] Benchmark: decode_ms reduz ≥ 5%

---

## Milestone 5 — Métricas Granulares

**Objetivo:** Observabilidade para cada estágio de cada modelo. Padronizado no `ModelOutput`.

### 5.1 Timing Keys Padronizadas

**Arquivos:** `models/base.py`

**DoD:**
- [ ] `ModelOutput.timings` tem keys padronizadas (todos os modelos):
  - `prepare_ms`: tempo de prepare_inputs
  - `prefill_ms`: tempo do first forward pass
  - `decode_ms`: tempo do decode loop (0 para modelos sem loop)
  - `decode_per_token_ms`: decode_ms / n_tokens
  - `total_ms`: tempo total de generate()
- [ ] Keys opcionais (model-specific):
  - `mel_ms`: mel spectrogram (Qwen)
  - `encoder_ms`: encoder pass (Whisper)
  - `ctc_ms`: CTC decode (Parakeet)
- [ ] Teste: todos os modelos populam as keys padronizadas

### 5.2 Métricas de Engine

**Arquivos:** `runner/engine.py`

**DoD:**
- [ ] `engine.start()` loga: `model_load_ms`, `warmup_ms`, `compile_ms`, `total_startup_ms`
- [ ] `engine.transcribe()` loga: `resample_ms`, `prepare_ms`, `generate_ms`, `total_ms`
- [ ] `session.finish()` loga: `bg_wait_ms`, `recompute_mode`, `proc_ms`, `gen_ms`, `finish_ms`

---

## Milestone 6 — Benchmark Suite ✅

**Status:** Implementado. 91 testes reais na RTX 3090.

**Resultados baseline (Qwen3-ASR-0.6B, RTX 3090):**

| Métrica | Valor |
|---------|-------|
| WER (FLEURS PT-BR) | 28.6% |
| RTF @1s | 0.43 |
| RTF @5s | 0.04 |
| Streaming RTF | 0.08-0.12 |
| Prefill | 45-56ms |
| Decode | 29-33 tok/s |
| Warm request E2E | 370-510ms |
| Chunk push | 0.01ms p50 |
| 16 concurrent batch | 100% success |
| 32 concurrent streaming | 100% success |

**Testes parametrizados por modelo via `MACAW_ASR_TEST_MODEL` env var (Ollama pattern).**

---

## Milestone 7 — torch.compile + CUDA Graphs (Implementação)

**Pré-requisito:** M3 (design) aprovado e interfaces definidas.

### 7.1 torch.compile Integration

**DoD:**
- [ ] `MACAW_ASR_TEST_MODEL=qwen MACAW_ASR_COMPILE=1 pytest tests/` — roda com compile
- [ ] Benchmark mostra ≥ 10% redução no decode_ms
- [ ] Nenhum teste funcional quebra com compile habilitado

### 7.2 CUDA Graphs para Qwen

**DoD:**
- [ ] `MACAW_ASR_CUDA_GRAPHS=1` habilita
- [ ] Static KV cache implementado
- [ ] Benchmark mostra ≥ 5% redução adicional
- [ ] Tokens idênticos com/sem CUDA graphs

---

## Ordem de Execução

```
M0 ✅ → M6 ✅ → M1 (warmup) → M5 (métricas) → M2 (multi-model) → M3 (shared opts) → M4 (qwen opts) → M7 (compile+graphs)
```

**Justificativa:**
1. **M0 ✅:** bugs corrigidos
2. **M6 ✅:** benchmark suite valida tudo
3. **M1:** warmup genérico — base para medições confiáveis
4. **M5:** métricas — sem elas não sabemos se otimizações funcionam
5. **M2:** multi-model — define as interfaces que M3/M4 implementam
6. **M3:** shared optimizations — torch.compile funciona para todos
7. **M4:** Qwen-specific — GPU mel, decode loop otimizado
8. **M7:** compile + CUDA graphs — última camada de otimização

---

## Resumo de Impacto Esperado

| Milestone | Escopo | Impacto |
|-----------|--------|---------|
| M1 Warmup | Todos os modelos | Elimina cold start no primeiro request |
| M2 Multi-Model | Arquitetura | Suporte a Whisper, Parakeet, futuros |
| M3 Shared Opts | Todos os modelos transformer | torch.compile: 10-30% decode speedup |
| M4 Qwen Opts | Qwen only | GPU mel: prepare 1500ms → 5ms |
| M5 Métricas | Todos os modelos | Observabilidade completa |
| M7 CUDA Graphs | Autoregressive only | 5-15% adicional no decode |

**Alvo final (Qwen3-ASR-0.6B, RTX 3090):**

| Estágio | Atual | Alvo |
|---------|-------|------|
| prepare_inputs | 15ms (warm) / 1500ms (cold) | ≤ 5ms (GPU mel) |
| prefill | 45ms | ≤ 40ms (compile) |
| decode (10 tokens) | 350ms | ≤ 200ms (compile + CUDA graphs) |
| total generate() | 400ms | ≤ 250ms |
| E2E transcribe | 500ms | ≤ 260ms |
