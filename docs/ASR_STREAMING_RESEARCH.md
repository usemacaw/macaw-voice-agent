# ASR Streaming Research — Aprendizados Completos

> Documento gerado em 2026-03-13 com todos os aprendizados da pesquisa e experimentacao
> de ASR streaming para o macaw-voice-agent com foco em Portugues Brasileiro (PT-BR).

---

## 1. Contexto e Objetivo

### Pipeline do Sistema

```
Microfone (browser) → WebSocket → VAD (Silero + Smart Turn)
    → ASR (batch ou streaming) → LLM (Qwen2.5-7B vLLM streaming)
    → TTS (Kokoro ONNX streaming) → WebSocket → Speaker
```

### Objetivo

Reduzir a latencia E2E (speech_stopped → first_audio_sent) que estava em ~2.5-3s,
sendo ~800ms gastos apenas no ASR batch (Whisper large-v3-turbo).

### Infraestrutura

| Componente | Hardware | Localizacao |
|---|---|---|
| STT (ASR) | RTX 2080 Ti 11GB | Vast.ai (ssh4.vast.ai:10690) |
| TTS (Kokoro) | Mesmo servidor | Vast.ai (compartilha GPU) |
| LLM (vLLM) | RTX A4000 16GB | Vast.ai (ssh7.vast.ai:12056) |
| API + VAD | CPU local | Maquina local |

### Formato de Audio

- API (client <-> server): PCM16 24kHz
- Interno (providers): PCM16 8kHz mono
- ASR input: PCM16 16kHz (resample automatico)

---

## 2. Modelos Testados

### 2.1 Whisper large-v3-turbo (via faster-whisper) — VENCEDOR

| Metrica | Valor |
|---|---|
| Params | ~800M |
| VRAM | ~3GB (int8) |
| Latencia batch | 400-800ms |
| Latencia streaming | 350-550ms |
| Qualidade PT-BR | Excelente |
| Streaming nativo | Nao (pseudo-streaming via LocalAgreement) |

**Resultado:** Melhor qualidade para PT-BR. Transcreve corretamente "cartao de credito",
"saldo", frases longas e curtas. Erros raros e menores.

**Configuracao final:**
```env
STT_PROVIDER=whisper
WHISPER_MODEL=large-v3-turbo
WHISPER_DEVICE=cuda
WHISPER_COMPUTE_TYPE=int8
WHISPER_BEAM_SIZE=1
WHISPER_VAD_FILTER=true
WHISPER_STREAM_CHUNK_MS=800
STT_LANGUAGE=pt
```

### 2.2 Whisper medium (via faster-whisper) — DESCARTADO

| Metrica | Valor |
|---|---|
| Params | ~769M |
| VRAM | ~2GB (int8) |
| Latencia batch | ~820ms |
| Qualidade PT-BR | Ruim |

**Resultado:** Qualidade muito inferior. Transcreveu "caixao de creios" em vez de
"cartao de creditos". Descartado imediatamente.

**Aprendizado:** Para PT-BR, o modelo `medium` nao e suficiente. O `large-v3-turbo`
tem qualidade muito superior com latencia similar.

### 2.3 NVIDIA Parakeet-TDT 0.6B v3 — DESCARTADO

| Metrica | Valor |
|---|---|
| Modelo | nvidia/parakeet-tdt-0.6b-v3 |
| Params | 600M |
| Classe NeMo | EncDecRNNTBPEModel |
| VRAM | ~6.3GB |
| Latencia batch | 130-430ms (rapido!) |
| Qualidade PT-BR | INACEITAVEL |
| Streaming nativo | Sim (TDT = Token Duration Transducer) |
| Idiomas | 25 (multilingual, auto-detect) |

**Problemas criticos:**

1. **Transcreve em ingles:** O modelo auto-detecta idioma e frequentemente escolhe
   ingles para audio PT-BR. Transcreveu "We robbed." e "I don't know." para frases
   em portugues.

2. **Sem mecanismo de forcar idioma:** O modelo RNNT nao tem parametro `language`
   no `transcribe()`. A assinatura e:
   ```python
   transcribe(audio, batch_size, return_hypotheses, num_workers, verbose,
              timestamps, override_config)
   ```
   Nao aceita `language`, `source_lang`, ou qualquer parametro de idioma.

3. **Tokens de idioma existem mas nao sao usaveis:** O vocabulario contem tokens
   como `<|pt|>`, `<|en|>`, `<|startoftranscript|>`, `<|predict_lang|>`, mas
   nao ha mecanismo para injecta-los no decoder RNNT durante inferencia.

4. **partial_hypothesis nao resolve:** Tentar injetar tokens de idioma via
   `partial_hypothesis` no `TranscribeConfig` nao funciona — os tokens sao
   preservados mas nao condicionam o modelo a produzir output em portugues.

5. **Treinado em Portugues Europeu:** A documentacao oficial confirma:
   > "Performance differences may be partly attributed to Portuguese variant
   > differences — our training data uses European Portuguese while most
   > benchmarks use Brazilian Portuguese."

**Aprendizado:** Modelos RNNT multilinguais com auto-detect de idioma sao
perigosos para PT-BR. Sem mecanismo de forcar idioma, a deteccao automatica
falha frequentemente.

### 2.4 NVIDIA FastConformer PT-BR (Hybrid) — DESCARTADO

| Metrica | Valor |
|---|---|
| Modelo | nvidia/stt_pt_fastconformer_hybrid_large_pc |
| Params | 113M |
| Classe NeMo | EncDecHybridRNNTCTCBPEModel |
| VRAM | ~2GB |
| Latencia batch | 100-130ms (muito rapido!) |
| Qualidade PT-BR | RUIM |
| Streaming nativo | Parcial (conformer_stream_step funciona) |
| Dados treino | ~2200h PT-BR |

**Resultados de transcricao:**

| Falado | Transcrito | Problema |
|---|---|---|
| "saldo" | "caldo" / "paudo" | Errado |
| "Gostaria de saber" | "Voltaria saber" | Errado |
| "qual o total" | "Que a tabeltal" | Muito errado |
| "Olá tudo bem" | "Olá tudo bem." | OK |

**Problemas:**

1. **Qualidade insuficiente para PT-BR coloquial:** Apesar de treinado em 2200h
   de PT-BR, o modelo tem apenas 113M params — muito pequeno para capturar a
   variacao do portugues brasileiro falado.

2. **Streaming limitado:** O modelo tem `att_context_size: [-1, -1]` (full context,
   nao cache-aware). Isso significa:
   - `setup_streaming()` e `transcribe_streaming()` NAO existem
   - `transcribe_simulate_cache_aware_streaming()` falha com
     `NotImplementedError: simulate streaming does not support EncDecHybridRNNTCTCBPEModel`
   - `conformer_stream_step()` FUNCIONA mas com qualidade degradada
     (modelo nao foi treinado para streaming com contexto limitado)

3. **Sem modelo streaming PT-BR disponivel:** Os modelos com sufixo `_streaming`
   so existem para ingles:
   ```
   stt_en_fastconformer_hybrid_large_streaming_80ms
   stt_en_fastconformer_hybrid_large_streaming_480ms
   stt_en_fastconformer_hybrid_large_streaming_1040ms
   stt_en_fastconformer_hybrid_large_streaming_multi
   ```

**Aprendizado:** Latencia fantastica (100-130ms) mas qualidade inaceitavel.
Modelo muito pequeno (113M) para PT-BR coloquial. Nao existe versao streaming
para portugues no NeMo model zoo.

### 2.5 NVIDIA Canary-1B v2 — NAO TESTADO

| Metrica | Valor |
|---|---|
| Modelo | nvidia/canary-1b-v2 |
| Params | 978M |
| Classe NeMo | EncDecMultiTaskModel (encoder-decoder) |
| VRAM | ~6GB |
| Streaming | NAO |

**Por que nao foi testado:**
- Sem streaming (processa arquivos completos)
- Tambem treinado em Portugues Europeu
- Ocuparia 6GB de VRAM (conflitaria com TTS no mesmo GPU)

**Vantagem unica:** Tem parametro `source_lang='pt'` para forcar idioma.
Poderia ser util como fallback batch, mas Whisper large-v3-turbo ja resolve.

### 2.6 Qwen3-ASR 0.6B — DESCARTADO (sessao anterior)

| Metrica | Valor |
|---|---|
| Params | 600M |
| Qualidade PT-BR | Muito ruim |

**Resultado:** "A transcricao regrediu demais" — modelo muito pequeno para PT-BR.
vLLM 0.8.4 no servidor tambem era incompativel (`ValueError: Qwen3ASRForConditionalGeneration
has no vLLM implementation`).

---

## 3. NVIDIA NeMo — Aprendizados Detalhados

### 3.1 Arquiteturas de Modelo ASR no NeMo

| Classe | Tipo | Streaming | Language Control |
|---|---|---|---|
| EncDecCTCBPEModel | CTC | Nao nativo | Nao |
| EncDecRNNTBPEModel | RNNT/TDT | Sim (se cache-aware) | Nao (auto-detect) |
| EncDecHybridRNNTCTCBPEModel | Hibrido CTC+RNNT | Parcial | Nao |
| EncDecMultiTaskModel | Encoder-Decoder | Nao | Sim (source_lang) |

### 3.2 Streaming no NeMo — O Que Funciona e O Que Nao

**Requisitos para streaming real no NeMo:**

1. **Modelo treinado com cache-aware attention** — `att_context_size` deve ser
   algo como `[70, 13]`, NAO `[-1, -1]`. O valor `[-1, -1]` significa full context
   (offline only).

2. **Modelos com sufixo `_streaming`** — sao os unicos treinados para streaming:
   ```
   stt_en_fastconformer_hybrid_large_streaming_80ms   (80ms latencia)
   stt_en_fastconformer_hybrid_large_streaming_480ms  (480ms latencia)
   stt_en_fastconformer_hybrid_large_streaming_1040ms (1040ms latencia)
   ```

3. **APIs de streaming:**
   - `setup_streaming()` + `transcribe_streaming()` — API alto nivel, so em modelos _streaming
   - `conformer_stream_step()` — API baixo nivel, funciona em qualquer ConformerEncoder
     mas com qualidade degradada se modelo nao e cache-aware
   - `FrameBatchASR` — utility class, requer cache-aware, falha com modelos offline
     (`"Subtraction, the - operator, with a bool tensor is not supported"`)
   - `transcribe_simulate_cache_aware_streaming()` — simula streaming,
     nao suporta modelos hibridos (EncDecHybridRNNTCTCBPEModel)

### 3.3 conformer_stream_step — Como Usar

```python
# Setup
model.encoder.setup_streaming_params(
    chunk_size=8,      # encoder output frames (8 frames = 640ms)
    shift_size=8,      # sem overlap
    left_chunks=4,     # 4 chunks anteriores como contexto
)

# Cache inicial
cache_channel, cache_time, cache_len = model.encoder.get_initial_cache_state(
    batch_size=1, dtype=torch.float32, device='cuda:0'
)

# Loop de streaming
for audio_chunk in stream:
    processed_signal, length = model.preprocessor(
        input_signal=audio_tensor, length=audio_len
    )

    result = model.conformer_stream_step(
        processed_signal=processed_signal,
        processed_signal_length=length,
        cache_last_channel=cache_channel,
        cache_last_time=cache_time,
        cache_last_channel_len=cache_len,
        keep_all_outputs=True,
        previous_hypotheses=prev_hyp,
        return_transcription=True,
    )

    greedy_preds, all_hyp, cache_channel, cache_time, cache_len, best_hyp = result[:6]
```

**Performance:** Chunk 0 ~1000ms (warmup GPU), chunks seguintes ~42ms.
Funciona mecanicamente mas qualidade e ruim com modelos nao-cache-aware.

### 3.4 Tokens Especiais no Vocabulario

Modelos multilinguais como Parakeet-TDT possuem tokens especiais:

```
<|startoftranscript|>  (id=4)
<|pnc|>               (punctuation, id=5)
<|nopnc|>             (id=6)
<|predict_lang|>      (id=22)
<|nopredict_lang|>    (id=23)
<|pt|>                (id=151)
<|en|>                (id=64)
<|spk0|> ... <|spk9|> (speaker diarization)
<|emo:neutral|>, <|emo:happy|>, etc.
```

Esses tokens sao parte do vocabulario SentencePiece (8192 tokens) mas nao ha
API para injecta-los como prompt/contexto durante inferencia RNNT.

### 3.5 Conflitos de Dependencia

**Protobuf:** NeMo requer `protobuf~=5.29.5`, gRPC requer `>=6.31.1`.
Workaround: `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`

**CUDA:** Parakeet-TDT emite warning sobre CUDA toolkit 12.4 vs 12.6:
```
No conditional node support for Cuda.
Cuda graphs with while loops are disabled, decoding speed will be slower
```
Isso desabilita CUDA graphs e torna o decoding mais lento.

---

## 4. Pseudo-Streaming com Whisper (LocalAgreement)

### 4.1 Conceito

Como Whisper nao suporta streaming nativo, implementamos pseudo-streaming:
durante a fala, o audio e acumulado e re-transcrito periodicamente.
O algoritmo LocalAgreement-2 confirma texto quando 2 inferencias consecutivas
concordam no mesmo prefixo.

### 4.2 Algoritmo LocalAgreement-2

```
1. Audio chunks chegam durante a fala (via gRPC streaming)
2. A cada ~800ms, re-transcreve TODO o audio acumulado
3. Compara palavras atuais com palavras da inferencia anterior
4. Prefixo comum (mesmas palavras no mesmo indice) → CONFIRMADO
5. Sufixo divergente → NAO-CONFIRMADO (pode mudar)
6. Retorna: texto confirmado + texto nao-confirmado como parcial
7. No finish_streaming: transcreve audio completo (resultado final)
```

**Exemplo de fluxo:**
```
Inferencia 1 (800ms): "Olá bom"
Inferencia 2 (1.6s):  "Olá bom dia como"
  → Confirmado: "Olá bom" (prefixo comum)
  → Parcial retornado: "Olá bom dia como"

Inferencia 3 (2.4s):  "Olá bom dia como vai"
  → Confirmado: "Olá bom dia como" (prefixo comum com inf. 2)
  → Parcial retornado: "Olá bom dia como vai"

Finish (3.0s):         "Olá bom dia, como vai você?"
  → Final retornado (transcricao completa)
```

### 4.3 Implementacao

O streaming e implementado em `whisper_stt.py`:

- **start_streaming():** Inicializa buffer de audio e estado LocalAgreement
- **process_chunk():** Acumula audio, re-transcreve a cada `WHISPER_STREAM_CHUNK_MS`,
  atualiza palavras confirmadas
- **finish_streaming():** Transcreve audio completo para resultado final
- **_update_local_agreement():** Compara palavras atuais com anteriores,
  confirma prefixo comum
- **_run_transcribe():** Execucao Whisper em thread via `run_inference()`

### 4.4 Parametros Testados

| Parametro | 1000ms | 800ms | Observacao |
|---|---|---|---|
| ASR latencia media | 418ms | 495ms | 800ms tem mais overhead |
| E2E latencia media | 1978ms | 2040ms | Similar |
| Melhora em falas longas | — | -519ms | 800ms melhor para falas >3s |
| Falas curtas | Melhor | +150ms overhead | 1000ms melhor para <1.5s |

**Configuracao escolhida:** 800ms (melhor para falas longas, que sao mais comuns).

---

## 5. Metricas de Latencia do Pipeline

### 5.1 Breakdown por Etapa (configuracao final)

```
Speech starts
  ↓
  [fala do usuario: 1.5-4.5s]
  ↓
Speech stops (VAD + Smart Turn)
  ↓
  ASR finish_streaming:     ~400-550ms
  ↓
  LLM first token:          ~480-730ms
  ↓
  First sentence complete:  ~300-500ms adicionais
  ↓
  TTS first audio:          ~300-400ms adicionais
  ↓
First audio sent to user
```

### 5.2 E2E Latency Medida (speech_stopped → first_audio_sent)

| Teste | Frase | E2E |
|---|---|---|
| 1 | "Olá, bom dia" (1.7s fala) | 1504-1800ms |
| 2 | Frase longa (4.5s fala) | 2256-2775ms |
| 3 | "1, 2, 3, 4" (2.4s fala) | 2085-2200ms |
| 4 | Frase curta (1.3s fala) | 1744-1903ms |

**Media E2E: ~2.0s** (vs ~2.5-3.0s antes da otimizacao)

### 5.3 Onde o Tempo E Gasto

| Etapa | % do E2E | Tempo medio | Otimizavel? |
|---|---|---|---|
| ASR | 25% | ~450ms | Parcialmente (streaming ja reduz) |
| LLM | 30% | ~550ms | GPU mais rapida ou modelo menor |
| Sentence parsing | 15% | ~300ms | Depende do LLM output |
| TTS | 20% | ~350ms | Ja e streaming |
| Network overhead | 10% | ~200ms | Servidores mais proximos |

---

## 6. Decisoes Arquiteturais

### 6.1 Por que Whisper batch+streaming > RNNT streaming nativo

| Aspecto | Whisper (pseudo-streaming) | RNNT (streaming nativo) |
|---|---|---|
| Qualidade PT-BR | Excelente | Ruim (113M) ou instavel (600M) |
| Latencia ASR | ~450ms | ~100-130ms |
| Streaming real | Nao (re-transcricao) | Sim (frame-by-frame) |
| Controle de idioma | Sim (`language="pt"`) | Nao (auto-detect falha) |
| Maturidade | Muito alta | Media (NeMo complexo) |
| Dependencias | `faster-whisper` (leve) | NeMo (pesado, conflitos) |

**Conclusao:** A diferenca de qualidade e tao grande que compensa a latencia extra.
Melhor ter 450ms com transcricao correta do que 100ms com "caldo" em vez de "saldo".

### 6.2 Arquitetura do Streaming gRPC

```
API (local)                          STT Server (Vast.ai)
    │                                       │
    │  speech_started                       │
    ├──────────────────────────────────────►│ start_streaming()
    │                                       │   → init audio_buffer, LocalAgreement state
    │                                       │
    │  audio chunks (cada 32ms)             │
    ├──────────────────────────────────────►│ process_chunk()
    │                                       │   → acumula audio
    │  (a cada ~800ms)                      │   → re-transcreve todo audio
    │◄──────────────────────────────────────┤   → retorna parcial (LocalAgreement)
    │  parcial: "Olá bom dia"               │
    │                                       │
    │  speech_stopped (end_of_stream)       │
    ├──────────────────────────────────────►│ finish_streaming()
    │                                       │   → transcreve audio completo
    │◄──────────────────────────────────────┤   → retorna texto final
    │  final: "Olá, bom dia."               │
```

### 6.3 Pre-buffer de Chunks

Problema: `start_stream()` e async e pode demorar. Chunks de audio que chegam
antes do start completar seriam perdidos.

Solucao: `_pre_buffers` no `RemoteASR` armazena ate 500 chunks (~16s) antes
do start completar. Apos start, faz flush dos chunks acumulados.

---

## 7. Problemas Encontrados e Solucoes

### 7.1 FrameBatchASR falha com modelo offline

**Erro:** `"Subtraction, the - operator, with a bool tensor is not supported"`

**Causa:** FrameBatchASR requer modelos treinados com cache-aware streaming
(`att_context_size != [-1, -1]`). Modelos offline usam full attention mask
(booleano), que nao suporta subtracao.

**Solucao:** Usar `conformer_stream_step()` (API baixo nivel) ou abandonar
o modelo e usar Whisper.

### 7.2 Parakeet-TDT transcreve em ingles

**Erro:** Audio PT-BR transcrito como "We robbed.", "I don't know."

**Causa:** Modelo multilingual com auto-detect de idioma. Para audios curtos
ou com ruido, o detector escolhe ingles.

**Investigacao realizada:**
- TranscribeConfig nao tem campo `language`
- Tokens `<|pt|>` existem no vocabulario mas nao ha API para injecta-los
- `partial_hypothesis` preserva tokens mas nao condiciona o decoder
- `decoding.compute_langs = False` — deteccao de idioma esta desativada por padrao
- Nenhum metodo `set_prompt()`, `change_language()` ou similar existe

**Solucao:** Modelo descartado. Nao ha workaround.

### 7.3 Protobuf version conflict

**Erro:** NeMo requer `protobuf~=5.29.5`, gRPC requer `>=6.31.1`

**Solucao:** `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` forca implementacao
Python pura que e compativel com ambas versoes.

### 7.4 Whisper hallucinations em audio curto

**Problema:** Falas muito curtas (<1.5s) podem ser transcritas incorretamente.
Ex: "saldo" → "papo", "bom" → "bom, ne?"

**Causa:** Whisper e treinado com chunks de 30s. Audio muito curto tem pouco
contexto para o modelo, levando a alucinacoes.

**Mitigacao parcial:**
- `vad_filter=True` remove silencio antes/depois
- `beam_size=1` (greedy) reduz alucinacoes vs beam search
- VAD `min_speech_ms=250` filtra falas muito curtas

### 7.5 SSH exit code 255 no Vast.ai

**Problema:** SSH retorna exit code 255 por causa de `setlocale: LC_ALL: cannot
change locale (en_US.UTF-8)`.

**Solucao:** Ignorar exit codes e verificar resultados separadamente.

---

## 8. Modelos NeMo Disponiveis (Referencia)

### Modelos PT-BR
```
stt_pt_fastconformer_hybrid_large_pc    (113M, offline, hibrido CTC+RNNT)
```
Nenhum modelo streaming para PT-BR.

### Modelos Streaming (somente ingles)
```
stt_en_fastconformer_hybrid_large_streaming_80ms
stt_en_fastconformer_hybrid_large_streaming_480ms
stt_en_fastconformer_hybrid_large_streaming_1040ms
stt_en_fastconformer_hybrid_large_streaming_multi
```

### Modelos Multilinguais
```
stt_multilingual_fastconformer_hybrid_large_pc
stt_multilingual_fastconformer_hybrid_large_pc_blend_eu
```
Nenhum com streaming. Treinados em Portugues Europeu.

### Modelos Multilinguais com Controle de Idioma
```
nvidia/parakeet-tdt-0.6b-v3     (RNNT, auto-detect, SEM controle)
nvidia/canary-1b-v2             (MultiTask, source_lang='pt', SEM streaming)
```

---

## 9. Recomendacoes para o Futuro

### 9.1 Otimizacoes de Latencia Pendentes

1. **GPU mais rapida para LLM:** O LLM (Qwen2.5-7B) leva ~500ms para first token
   na RTX A4000. Uma A100 ou H100 reduziria para ~150-200ms.

2. **Modelo LLM menor:** Qwen2.5-3B ou similar poderia reduzir latencia do LLM
   em 30-40% com perda minima de qualidade para respostas curtas.

3. **Servidores co-localizados:** STT, TTS e LLM no mesmo datacenter eliminaria
   ~100-200ms de latencia de rede.

4. **Speculative decoding no LLM:** vLLM suporta speculative decoding que pode
   reduzir latencia significativamente.

### 9.2 Quando Re-avaliar NeMo/FastConformer

- Se NVIDIA lancar modelo `stt_pt_fastconformer_*_streaming` (streaming PT-BR)
- Se Parakeet-TDT ganhar parametro de idioma forcado
- Se surgir modelo RNNT com >500M params treinado em PT-BR
- Se Canary ganhar modo streaming

### 9.3 Alternativas a Monitorar

| Modelo | Status | Potencial |
|---|---|---|
| Moonshine v2 | Ingles only (2026) | Se adicionar PT-BR, latencia ~50ms |
| Distil-Whisper | Disponivel | Menor mas pode perder qualidade PT-BR |
| whisper-streaming (UFAL) | Maduro | Implementacao mais sofisticada de LocalAgreement |
| WhisperKit | Apple only | N/A para server |

---

## 10. Configuracao Final do Sistema

### .env da API
```env
ASR_PROVIDER=remote
ASR_REMOTE_TARGET=142.127.68.223:10473
ASR_REMOTE_STREAMING=true
ASR_LANGUAGE=pt

TTS_PROVIDER=remote
TTS_REMOTE_TARGET=142.127.68.223:10584

LLM_PROVIDER=vllm
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ
LLM_MAX_TOKENS=30
VLLM_BASE_URL=http://173.206.147.184:41209/v1

VAD_SILENCE_MS=150
VAD_PREFIX_PADDING_MS=300
VAD_MIN_SPEECH_MS=250
```

### Comando STT Server (Vast.ai)
```bash
cd /app && PYTHONPATH=. \
  GRPC_PORT=50060 \
  STT_PROVIDER=whisper \
  WHISPER_MODEL=large-v3-turbo \
  STT_LANGUAGE=pt \
  WHISPER_STREAM_CHUNK_MS=800 \
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python \
  python3 -m stt.server
```

### Processos Ativos no Vast.ai (RTX 2080 Ti)
```
STT: Whisper large-v3-turbo (int8) — ~3GB VRAM
TTS: Kokoro ONNX — ~1GB VRAM
```

### Processo LLM (RTX A4000)
```
vLLM: Qwen2.5-7B-Instruct-AWQ — ~8GB VRAM
```
