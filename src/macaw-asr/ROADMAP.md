# Roadmap: Inferência Otimizada — Qwen3-ASR

> Roadmap executável para implementar inferência de alta performance no macaw-asr.
> Cada milestone tem Definition of Done (DoD) verificável.
> Ordem importa: bugs críticos primeiro, otimizações depois.

---

## Milestone 0 — Corrigir Bugs Críticos do Port

**Objetivo:** Garantir que o código portado funciona corretamente antes de otimizar.

### 0.1 Corrigir EOS Token Override

**Contexto:** `qwen.py:138` tenta sobrescrever `strategy._eos_id` com o EOS real do tokenizer, mas o mecanismo é frágil — acessa atributo privado e depende de `hasattr`. Se falhar, o modelo nunca para no EOS e gera 32 tokens (garbage) em toda utterance.

**Solução:** Modelo deve expor `eos_token_id` como propriedade pública. Engine cria strategy com o EOS correto após o model.load().

**Arquivos:** `models/base.py`, `models/qwen.py`, `models/mock.py`, `runner/engine.py`

**DoD:**
- [ ] `ASRModel` ABC tem propriedade abstrata `eos_token_id: int`
- [ ] `QwenASRModel.eos_token_id` retorna o token correto do tokenizer
- [ ] `MockASRModel.eos_token_id` retorna 0
- [ ] `ASREngine._create_strategy()` usa `self._model.eos_token_id` em vez de placeholder
- [ ] `qwen.py:generate()` NÃO faz override dinâmico de `strategy._eos_id`
- [ ] Teste: `test_strategy_uses_model_eos_token` verifica que strategy recebe EOS correto
- [ ] Teste: `test_mock_model_eos_token` verifica propriedade do mock

### 0.2 Corrigir DecodeContext List Mutation

**Contexto:** `DecodeContext.recent_tokens` é uma lista mutável que cresce a cada `should_stop()`. Se o context for reutilizado entre sessions, a lista vaza tokens da session anterior. Não causa crash, mas desperdiça memória e pode causar false positives na repetition detection.

**Solução:** DecodeContext deve ser criado fresh por decode loop. Não reutilizar entre sessions.

**Arquivos:** `decode/strategies.py`, `runner/engine.py`

**DoD:**
- [ ] `DecodeContext` é criado dentro de `model.generate()`, não passado de fora
- [ ] OU: `DecodeContext.__post_init__` garante que `recent_tokens` é sempre nova lista
- [ ] Teste: `test_decode_context_isolation` — dois decode loops consecutivos não compartilham tokens
- [ ] Teste: `test_repetition_detection_across_sessions` — session 2 não vê tokens da session 1

### 0.3 Validar fast_finish_inputs na Integração

**Contexto:** `QwenASRModel.fast_finish_inputs()` foi portado do original mas nunca testado com GPU real. Depende de `_torch_extract_fbank_features`, `_get_feat_extract_output_lengths`, e manipulação de `AUDIO_PAD_TOKEN`. Qualquer divergência gera inputs corrompidos → modelo gera lixo.

**Solução:** Teste de integração que compara output de fast_finish vs full prepare_inputs.

**Arquivos:** `models/qwen.py`, `tests/test_qwen_integration.py`

**DoD:**
- [ ] Teste (requer GPU): gera inputs via `prepare_inputs(full_audio)` e via `fast_finish_inputs(full_audio, cached_partial, partial_len)` — verifica que `input_ids` shape é idêntico
- [ ] Teste (requer GPU): verifica que `generate()` com fast_finish inputs produz texto equivalente ao full path
- [ ] Teste é marcado `@pytest.mark.gpu` e skipado em CI sem GPU
- [ ] Logging: `fast_finish_inputs` loga shape de input_ids antes/depois do adjust para debug

---

## Milestone 1 — Warmup Agressivo

**Objetivo:** Primeiro request não paga custo de compilação CUDA. Todos os code paths devem ser exercitados no warmup.

### 1.1 Warmup Multi-Path

**Contexto:** Warmup atual roda 1 prefill + 1 decode step com áudio de 1s. Isso compila kernels para uma shape específica, mas pode não cobrir shapes reais (áudios de 0.5s, 3s, 5s). O ideal é aquecer com múltiplas shapes.

**Arquivos:** `models/qwen.py`

**DoD:**
- [ ] `warmup()` roda 3 inferências: áudio de 0.5s, 1s, 3s
- [ ] Cada inferência faz prefill + 3 decode steps (cobre decode loop)
- [ ] `warmup()` também roda `fast_finish_inputs()` com cache dummy
- [ ] Log: tempo total de warmup em ms
- [ ] Métrica: primeiro request real tem latência ≤ 1.1x dos requests seguintes (sem cold start)

### 1.2 Warmup do Mel Spectrogram

**Contexto:** `processor()` (CPU) e `_torch_extract_fbank_features()` (GPU) têm cold start na primeira execução (alocação de buffers, JIT de operações). Precisam ser aquecidos.

**Arquivos:** `models/qwen.py`

**DoD:**
- [ ] `warmup()` chama `prepare_inputs()` (aquece CPU processor)
- [ ] `warmup()` chama `fast_finish_inputs()` com inputs dummy (aquece GPU mel)
- [ ] Tempo total de warmup ≤ 5s (não pode bloquear startup por muito tempo)

---

## Milestone 2 — torch.compile no Decode Loop

**Objetivo:** Reduzir latência do decode step via kernel fusion. Alvo: 10-30% de redução no decode_ms.

### 2.1 Compilar o Thinker Model

**Contexto:** `torch.compile()` analisa o forward pass e funde operações (matmul+bias, softmax+masking, etc.), reduzindo launches de kernel CUDA. O decode loop executa 32 forward passes — cada ms economizado por step = 32ms total.

**Arquivos:** `models/qwen.py`, `config.py`

**DoD:**
- [ ] `EngineConfig` tem campo `enable_compile: bool = False` (opt-in)
- [ ] `QwenASRModel.load()` aplica `torch.compile(thinker, mode='reduce-overhead')` quando habilitado
- [ ] Fallback: se torch.compile falhar (versão antiga, model incompatível), log warning e continua sem compile
- [ ] Teste (requer GPU): benchmark com e sem compile — decode_ms deve reduzir ≥ 10%
- [ ] Warmup roda com modelo compilado (primeiro compile + warmup ~30s é aceitável)
- [ ] Funciona com bfloat16

### 2.2 Compilar Apenas o Decode Step (Fallback)

**Contexto:** Se compile do modelo inteiro falhar (graph breaks no Qwen), compilar apenas a porção do decode step: forward pass do thinker com `input_ids[:, -1:]`.

**Arquivos:** `models/qwen.py`

**DoD:**
- [ ] Se full compile falhar, tenta `torch.compile` em uma wrapper function que faz um decode step
- [ ] Log: "Full compile failed, using partial compile" ou "Compile disabled"
- [ ] Latência com partial compile ≥ 5% melhor que sem compile

---

## Milestone 3 — CUDA Graphs para Decode Step

**Objetivo:** Eliminar overhead de CPU→GPU dispatch no decode loop. Alvo: 5-15% adicional.

### 3.1 Capturar Decode Step como CUDA Graph

**Contexto:** Cada decode step lança dezenas de CUDA kernels. CPU precisa preparar e despachar cada um. CUDA Graphs capturam toda a sequência e replays como uma operação atômica, eliminando overhead de launch.

**Pré-requisito:** Milestone 2 (torch.compile) deve estar estável.

**Arquivos:** `models/qwen.py`

**DoD:**
- [ ] Na warmup, após compile, captura 1 decode step como CUDA Graph
- [ ] Decode loop usa `graph.replay()` em vez de chamada direta ao model
- [ ] Input/output tensors são pre-alocados e reutilizados (in-place update)
- [ ] Fallback: se CUDA Graph falhar (shape dinâmica, model incompatível), usa decode normal
- [ ] Teste (requer GPU): decode com CUDA Graph produz mesmos tokens que sem Graph
- [ ] Teste (requer GPU): benchmark mostra redução ≥ 5% no decode_ms
- [ ] Funciona com KV cache (shapes crescentes podem ser problemáticas)

### 3.2 Static KV Cache Shape

**Contexto:** CUDA Graphs requerem shapes fixas. KV cache cresce a cada decode step. Solução: pre-alocar KV cache com `max_new_tokens` posições extras.

**Arquivos:** `models/qwen.py`

**DoD:**
- [ ] KV cache pre-alocado após prefill com `seq_len + max_new_tokens` posições
- [ ] Decode step escreve no KV cache in-place (sem realocação)
- [ ] Decode step usa index para saber qual posição preencher
- [ ] Memória extra ≤ 50MB (para max_new_tokens=32)
- [ ] Teste: verificar que tokens gerados com static cache == tokens com dynamic cache

---

## Milestone 4 — GPU-First Mel Spectrogram

**Objetivo:** Eliminar CPU processor como bottleneck. Computar mel 100% em GPU.

### 4.1 GPU Mel como Default (não apenas fast_finish)

**Contexto:** Hoje, `prepare_inputs()` usa o CPU processor (~1500ms). `fast_finish_inputs()` usa GPU mel (~0.5ms), mas só é chamado quando há cache do background. Se GPU mel funcionar standalone, podemos eliminar o CPU processor do hot path.

**Arquivos:** `models/qwen.py`

**DoD:**
- [ ] Novo método `_prepare_inputs_gpu(audio, prefix)` que:
  - Computa mel no GPU (`_torch_extract_fbank_features`)
  - Tokeniza o prompt no CPU (rápido, ~1ms)
  - Monta input_ids com audio_pad tokens calculados via `_get_feat_extract_output_lengths`
  - Retorna tensors prontos no device
- [ ] `prepare_inputs()` chama `_prepare_inputs_gpu()` como primeiro path
- [ ] Fallback para CPU processor se GPU mel falhar
- [ ] Teste (requer GPU): output de `_prepare_inputs_gpu()` produz mesma transcrição que CPU processor
- [ ] Benchmark: prepare_inputs com GPU mel ≤ 10ms (vs ~1500ms CPU)

### 4.2 Eliminar CPU Processor do Hot Path

**Contexto:** Se 4.1 funcionar, o CPU processor só é necessário no warmup e como fallback. O hot path (batch + streaming) usa GPU mel exclusivamente.

**Arquivos:** `models/qwen.py`, `runner/session.py`

**DoD:**
- [ ] Background precompute em `session.py` usa GPU mel (não CPU processor)
- [ ] `finish()` path nunca chama CPU processor (fast_finish OU GPU mel)
- [ ] Batch `transcribe()` usa GPU mel
- [ ] CPU processor mantido apenas em `warmup()` e como fallback explícito
- [ ] Métrica: prepare_inputs médio ≤ 5ms para áudio de 1-5s

---

## Milestone 5 — Métricas Granulares

**Objetivo:** Observabilidade completa para cada estágio da inferência. Sem métricas, não sabemos se as otimizações funcionam.

### 5.1 Structured Metrics no ModelOutput

**Contexto:** `ModelOutput.timings` é um dict genérico. Precisa de campos tipados para cada estágio.

**Arquivos:** `models/base.py`, `models/qwen.py`

**DoD:**
- [ ] `ModelOutput.timings` tem keys padronizadas:
  - `mel_ms`: tempo do mel spectrogram (GPU ou CPU)
  - `tokenize_ms`: tempo da tokenização
  - `prefill_ms`: tempo do prefill step
  - `decode_ms`: tempo total do decode loop
  - `decode_per_token_ms`: decode_ms / n_tokens
  - `total_ms`: tempo total de generate()
- [ ] `QwenASRModel.generate()` popula todos os campos
- [ ] `SessionMetrics` agrega across chunks
- [ ] Teste: todos os campos de timing são > 0 para inferência real

### 5.2 Métricas de Warmup e Load

**Arquivos:** `models/qwen.py`, `runner/engine.py`

**DoD:**
- [ ] `engine.start()` loga:
  - `model_load_ms`: tempo para carregar pesos
  - `warmup_ms`: tempo de warmup
  - `compile_ms`: tempo de torch.compile (se habilitado)
  - `total_startup_ms`: soma
- [ ] Formato: `"ASREngine started: model=qwen load=2340ms warmup=890ms compile=15200ms total=18430ms"`

---

## Milestone 6 — Benchmark Suite

**Objetivo:** Benchmark reproduzível para medir impacto de cada otimização.

### 6.1 Benchmark Script

**Arquivos:** `tests/benchmark_inference.py`

**DoD:**
- [ ] Script que roda N inferências (default 50) com áudio real (ou sintético)
- [ ] Mede: p50, p90, p99 de cada estágio (mel, prefill, decode, total)
- [ ] Compara: batch vs streaming, com/sem compile, com/sem CUDA graph
- [ ] Output: tabela markdown com resultados
- [ ] Executável via `macaw-asr benchmark` (CLI command)
- [ ] Baseline salvo em `benchmarks/baseline.json` para comparação

### 6.2 Regression Gate

**DoD:**
- [ ] Script compara results contra baseline
- [ ] Falha se p50 regredir > 10% em qualquer estágio
- [ ] Integrável com CI (exit code 0/1)

---

## Resumo de Impacto Esperado

| Milestone | Estágio Impactado | Redução Esperada | Acumulado |
|-----------|-------------------|-------------------|-----------|
| M0 | Correctness | N/A (bugfix) | — |
| M1 | First request | Elimina cold start | — |
| M2 | decode_ms | 10-30% | 10-30% |
| M3 | decode_ms | 5-15% | 15-40% |
| M4 | prepare_inputs_ms | 99% (1500ms → 5ms) | — |
| M5 | Observability | N/A (métricas) | — |
| M6 | Validation | N/A (benchmark) | — |

**Alvo final:**
- `prepare_inputs`: ≤ 5ms (GPU mel)
- `prefill`: ≤ 50ms
- `decode (32 tokens)`: ≤ 60ms (com compile + CUDA graph)
- `total generate()`: ≤ 120ms
- `e2e (preprocess + generate + postprocess)`: ≤ 130ms

---

## Ordem de Execução

```
M0 (bugs) → M1 (warmup) → M5 (métricas) → M4 (GPU mel) → M2 (compile) → M3 (CUDA graphs) → M6 (benchmark)
```

**Justificativa da ordem:**
1. **M0 primeiro:** código bugado não pode ser otimizado
2. **M1 depois:** warmup é pré-requisito para medições confiáveis
3. **M5 antes de M2/M3:** sem métricas, não sabemos se otimizações funcionam
4. **M4 antes de M2:** GPU mel elimina o maior bottleneck (1500ms), compile otimiza o segundo (decode)
5. **M2 antes de M3:** CUDA graphs dependem de compile estável
6. **M6 por último:** benchmark suite valida tudo junto
