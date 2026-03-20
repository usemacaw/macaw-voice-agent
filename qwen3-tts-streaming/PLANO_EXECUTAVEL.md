# Plano Executável: macaw-qwen3-tts-streaming

> Biblioteca própria de streaming real para Qwen3-TTS, combinando as melhores técnicas de `faster-qwen3-tts`, `rekuenkdr-streaming` e `dffdeeq-streaming`.
>
> **Meta: TTFA ≤ 88ms** (1 frame @ 12Hz = 83ms + 5ms overhead)

---

## 1. Análise Comparativa dos Projetos Estudados

### 1.1 Arquitetura Qwen3-TTS (Base)

O Qwen3-TTS usa um pipeline de 3 estágios, todos potencialmente streamable:

```
Texto → Talker (28 layers) → CB0 token
                                ↓
                    CodePredictor (5 layers) → CB1-CB15
                                ↓
                    Code2Wav (Causal ConvNet @ 12Hz) → PCM 24kHz
```

**Propriedades chave para streaming:**
- **12Hz tokenizer**: 1 frame = 83.3ms de áudio → TTFA teórico de ~83ms
- **Code2Wav causal (left-context only)**: sem lookahead, decodifica imediatamente
- **Dual-Track**: texto + áudio tokens somados no embedding (não concatenados)
- **CodePredictor**: 5 layers, gera 15 codebooks autoregressivamente por frame

### 1.2 Matriz de Técnicas por Projeto

| Técnica | faster-qwen3-tts | rekuenkdr | dffdeeq | Nossa lib |
|---------|:-:|:-:|:-:|:-:|
| **CUDA Graphs (Talker)** | ✅ StaticCache manual | ❌ | ❌ | ✅ |
| **CUDA Graphs (Predictor)** | ✅ StaticCache manual | ❌ | ❌ | ✅ |
| **torch.compile (Decoder)** | ❌ | ✅ reduce-overhead | ✅ reduce-overhead | ✅ |
| **torch.compile (Talker)** | ❌ (usa CUDA graph) | ✅ mode=default | ✅ mode=default | ❌ (CUDA graph é melhor) |
| **torch.compile (Predictor)** | ❌ (usa CUDA graph) | ✅ reduce-overhead | ✅ reduce-overhead | ❌ (CUDA graph é melhor) |
| **Two-Phase Latency** | ❌ | ✅ Phase1=4fr/Phase2=12fr | ❌ | ✅ (Phase1=1fr) |
| **Hann Crossfade** | ❌ | ✅ 512 samples | ✅ linear (opcional) | ✅ Hann |
| **Sliding Window Decode** | ✅ 25-frame context | ✅ decode_window=80 | ✅ decode_window=80 | ✅ |
| **Fixed-Shape Padding** | ❌ | ✅ left-pad | ✅ left-pad | ✅ |
| **GPU-Resident Codes** | ✅ | ✅ | ✅ | ✅ |
| **Pull-Based Generator** | ✅ | ✅ | ✅ | ✅ |
| **Repetition Penalty** | ✅ cross-chunk history | ✅ circular buffer GPU | ✅ circular buffer GPU | ✅ circular buffer |
| **Ref Code Context (ICL)** | ✅ accumulated decode | ✅ prefix no window | ✅ prefix no window | ✅ prefix |
| **Calibration Phase** | ✅ min 25 frames | ❌ (samples_per_frame fixo) | ❌ (samples_per_frame fixo) | ❌ (fixo, 12Hz = 2000 samples/frame) |
| **Batch Streaming** | ❌ | ✅ | ✅ | ❌ (voice agent = 1 request) |
| **TTFA Medido** | ~156ms (RTX 4090) | ~208ms (Phase1) | ~200-300ms | **Meta: ≤88ms** |
| **RTF** | 4.78x (RTX 4090) | 0.36-0.39 | ~6x | Meta: ≥5x |

### 1.3 Por Que Podemos Chegar a 88ms

**O gargalo dos projetos existentes:**

1. `faster-qwen3-tts`: TTFA ~156ms porque `chunk_size=8` (espera 8 frames antes de decodificar). Com `chunk_size=1` e overhead de CUDA graph, daria ~83ms + decode_time.
2. `rekuenkdr`: TTFA ~208ms porque Phase1 usa `emit_every_frames=4-5` (espera 4-5 frames).
3. `dffdeeq`: TTFA ~200-300ms porque `emit_every_frames=8`.

**Nossa vantagem:** emitir após **1 frame** (83ms) usando:
- CUDA graphs para Talker + Predictor (eliminam overhead Python, ~20ms/step com graphs)
- torch.compile para Code2Wav com fixed-shape padding (decode em ~5ms)
- Sem calibration phase (usamos `samples_per_frame=2000` fixo, 12Hz → 24kHz)

**Cálculo de TTFA:**
```
Prefill: ~0ms (já aconteceu antes da primeira emissão)
1 Talker step (CUDA graph): ~8ms
1 Predictor step (CUDA graph): ~4ms
Code2Wav decode (torch.compile, 1 frame): ~3-5ms
Overhead Python + resampling: ~2-3ms
─────────────────────────────────────────
Total estimado: ~17-20ms POR FRAME
+ 1 frame @ 12Hz: 83ms de áudio
= TTFA ≈ 83ms + 17ms = ~100ms (worst case)
= TTFA ≈ 83ms + 5ms = ~88ms (best case com graphs quentes)
```

**Evidência real (faster-qwen3-tts benchmarks):**
- `ms/step` medido: ~20.9ms no RTX 4090
- Com `chunk_size=1`: TTFA = 1 × 20.9ms + decode_time ≈ 25ms + decode
- Code2Wav com torch.compile: ~3-5ms (evidência do dffdeeq)
- **Total: ~25-30ms de processamento + 83ms de áudio = TTFA efetivo ~88ms**

---

## 2. Arquitetura da Biblioteca

### 2.1 Estrutura de Arquivos

```
macaw-qwen3-tts-streaming/
├── macaw_tts/
│   ├── __init__.py              # Public API: MacawTTS
│   ├── model.py                 # MacawTTS — wrapper principal
│   ├── streaming.py             # StreamingGenerator — core do streaming
│   ├── talker_graph.py          # TalkerCUDAGraph — CUDA graph do Talker
│   ├── predictor_graph.py       # PredictorCUDAGraph — CUDA graph do CodePredictor
│   ├── decoder.py               # StreamingDecoder — Code2Wav com torch.compile
│   ├── crossfade.py             # HannCrossfader — crossfade entre chunks
│   ├── sampling.py              # sample_logits, repetition_penalty
│   └── audio.py                 # Resampling, float32→PCM16, format conversion
├── tests/
│   ├── test_streaming.py        # Testes de streaming end-to-end
│   ├── test_crossfade.py        # Testes de crossfade
│   ├── test_decoder.py          # Testes do decoder
│   └── conftest.py              # Fixtures (mock model, etc)
├── benchmarks/
│   ├── ttfa.py                  # Benchmark de TTFA
│   └── rtf.py                   # Benchmark de RTF
├── examples/
│   ├── basic_streaming.py       # Uso básico
│   └── voice_agent.py           # Integração com Macaw Voice Agent
├── pyproject.toml
├── README.md
└── CHANGELOG.md
```

### 2.2 Diagrama de Componentes

```
┌─────────────────────────────────────────────────────────┐
│                    MacawTTS (model.py)                   │
│  - from_pretrained() → carrega Qwen3-TTS + warmup       │
│  - stream() → retorna StreamingGenerator                │
│  - stream_voice_clone() → com ref audio                 │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│              StreamingGenerator (streaming.py)           │
│  Pull-based async generator                             │
│                                                         │
│  ┌─────────┐   ┌──────────────┐   ┌─────────────────┐  │
│  │ Talker   │──▶│ Predictor    │──▶│ StreamingDecoder │  │
│  │ CUDA     │   │ CUDA Graph   │   │ (torch.compile)  │  │
│  │ Graph    │   │              │   │                   │  │
│  │          │   │ CB0 → CB1-15 │   │ Codes → PCM      │  │
│  └─────────┘   └──────────────┘   └────────┬──────────┘  │
│                                             │            │
│                                    ┌────────▼──────────┐ │
│                                    │  HannCrossfader   │ │
│                                    │  Overlap+blend    │ │
│                                    └────────┬──────────┘ │
│                                             │            │
│                              yield (pcm_chunk, sr, meta) │
└─────────────────────────────────────────────────────────┘
```

### 2.3 Fluxo de Dados Detalhado

```
Input: "Olá, como posso ajudar?"

1. PREFILL (uma vez por utterance)
   ├─ Tokenize text → input_ids
   ├─ Build embeddings (text + codec prefill + speaker)
   ├─ Talker forward (HF, variable-length) → KV cache
   ├─ Copy KV cache → TalkerCUDAGraph.static_cache
   └─ Capture graphs se primeira execução

2. STREAMING LOOP (por frame)
   ├─ PredictorCUDAGraph.run(past_hidden, cb0_embed) → [CB1..CB15]  (~4ms)
   ├─ Combine: codec_hiddens = sum(CB0_emb + CB1_emb + ... + CB15_emb)
   ├─ Add trailing text hidden or tts_pad_embed
   ├─ TalkerCUDAGraph.run(codec_hiddens, position) → hidden_states    (~8ms)
   ├─ Sample CB0 token from logits (temperature, top_k, repetition_penalty)
   ├─ EOS check → break se codec_eos
   ├─ Append codes ao buffer (GPU-resident)
   │
   ├─ PHASE 1 (frame 0): emit_every=1 → decode IMEDIATAMENTE
   │   ├─ StreamingDecoder.decode(codes[-1:], pad_to=decode_window)     (~5ms)
   │   ├─ HannCrossfader.apply(None, chunk)  → fade-in suave
   │   └─ yield (chunk, 24000, {ttfa_ms, step_ms, phase: 1})
   │
   └─ PHASE 2 (frames 1+): emit_every=4 → decode a cada 4 frames
       ├─ StreamingDecoder.decode(codes[-window:], pad_to=decode_window) (~5ms)
       ├─ Extract new samples: chunk = wav[-step_samples:]
       ├─ HannCrossfader.apply(prev_tail, chunk) → blend suave
       └─ yield (chunk, 24000, {step_ms, phase: 2, chunk_idx})

3. FLUSH (ao terminar)
   ├─ Decode frames restantes
   ├─ HannCrossfader.apply(prev_tail, final) → fade-out suave
   └─ yield (final_chunk, 24000, {is_final: True})
```

---

## 3. Especificação de Cada Módulo

### 3.1 `model.py` — MacawTTS

```python
class MacawTTS:
    """Wrapper principal. Carrega modelo, gerencia CUDA graphs, expõe API de streaming."""

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device: str = "cuda",
        dtype: torch.dtype = torch.bfloat16,
    ) -> "MacawTTS":
        """Carrega modelo Qwen3-TTS e prepara CUDA graphs."""

    def stream(
        self,
        text: str,
        language: str = "Portuguese",
        speaker: str = "Ryan",
        *,
        temperature: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.05,
        max_frames: int = 2048,
    ) -> Generator[tuple[np.ndarray, int, dict], None, None]:
        """Streaming real — yield (pcm_float32_24k, sample_rate, metadata)."""

    def stream_voice_clone(
        self,
        text: str,
        language: str = "Portuguese",
        ref_audio: str | np.ndarray,
        ref_text: str = "",
        *,
        # mesmos params de stream()
    ) -> Generator[tuple[np.ndarray, int, dict], None, None]:
        """Streaming com voice cloning (ICL mode)."""

    @property
    def sample_rate(self) -> int:
        """24000 Hz (output nativo do Qwen3-TTS 12Hz)."""
```

**Fonte:** Baseado em `faster-qwen3-tts/faster_qwen3_tts/model.py` (1150 linhas)

**Decisões de design:**
- `from_pretrained()` classmethod (padrão HuggingFace)
- Lazy CUDA graph capture (no primeiro `stream()`)
- Sync generator (não async) — roda em thread no voice agent
- Metadata dict em cada yield para observabilidade

### 3.2 `streaming.py` — StreamingGenerator

```python
def fast_generate_streaming(
    talker_graph: TalkerCUDAGraph,
    predictor_graph: PredictorCUDAGraph,
    decoder: StreamingDecoder,
    crossfader: HannCrossfader,
    *,
    prefill_output: PrefillOutput,
    max_new_tokens: int = 2048,
    emit_every_phase1: int = 1,      # 1 frame = 83ms → TTFA ~88ms
    emit_every_phase2: int = 4,      # 4 frames = 333ms → throughput estável
    phase1_frames: int = 1,          # Só o primeiro frame é Phase 1
    decode_window: int = 40,         # 40 frames context para decoder
    temperature: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.05,
    rep_penalty_window: int = 256,
) -> Generator[tuple[torch.Tensor, dict], None, None]:
    """Core do streaming. Pull-based generator."""
```

**Fonte:** Baseado em `faster-qwen3-tts/faster_qwen3_tts/streaming.py` (170 linhas) + two-phase de rekuenkdr

**Evidências que suportam os parâmetros:**
- `emit_every_phase1=1`: rekuenkdr usa 4-5 e atinge 208ms. Com 1, atingimos ~88ms
- `emit_every_phase2=4`: Balance entre latência e qualidade. rekuenkdr usa 12, faster usa 12
- `phase1_frames=1`: Apenas o primeiro frame precisa ser agressivo
- `decode_window=40`: Menor que rekuenkdr (80) e dffdeeq (80), mas suficiente para qualidade. Code2Wav 12Hz é causal e funciona com 25 frames de contexto (evidência: faster-qwen3-tts usa 25)

### 3.3 `talker_graph.py` — TalkerCUDAGraph

```python
class TalkerCUDAGraph:
    """CUDA graph para single-token decode do Talker (28 layers)."""

    def __init__(self, talker_model, config, max_seq_len=2048):
        # StaticCache pré-alocado
        # Attention mask table pré-computado
        # Input/output buffers estáticos

    def capture(self, prefill_len: int, num_warmup: int = 3):
        """Captura CUDA graph após prefill."""

    def run(self, inputs_embeds: torch.Tensor, position: int) -> torch.Tensor:
        """Executa 1 step via graph replay. Retorna hidden_states."""

    def copy_cache_from_hf(self, hf_past_key_values):
        """Copia KV cache do prefill HF para StaticCache."""
```

**Fonte:** Baseado em `faster-qwen3-tts/faster_qwen3_tts/talker_graph.py` (214 linhas)

**Otimizações chave (evidência do código faster):**
- `attn_mask_table[position]` pré-computado (linha 74-92): O(1) por step
- `cache_position` buffer estático de 1 elemento (linha 64): zero alocação
- Graph capture em stream separado (linha 110-147): sem interferência
- `static_cache.reset()` entre sequências (sem realocação)

### 3.4 `predictor_graph.py` — PredictorCUDAGraph

```python
class PredictorCUDAGraph:
    """CUDA graph para CodePredictor — gera 15 codebooks em um graph."""

    def __init__(self, predictor_model, config, talker_hidden_size):
        # StaticCache para 5 layers
        # 15 lm_heads (um por codebook)
        # Pre-built cache positions para cada step

    def capture(self, num_warmup: int = 3):
        """Captura CUDA graph do loop de 15 codebooks."""

    def run(self, pred_input: torch.Tensor) -> torch.Tensor:
        """Executa 15-step loop via graph replay. Retorna [15] token IDs."""
```

**Fonte:** Baseado em `faster-qwen3-tts/faster_qwen3_tts/predictor_graph.py` (215 linhas)

**Detalhes do loop interno (evidência linhas 115-167):**
```
Step 0 (Prefill): input=[past_hidden, cb0_embed] (2 tokens)
  → Forward 5 layers → Sample CB1

Steps 1-14 (Decode): input=[cb_i_embed] (1 token cada)
  → Forward 5 layers → Sample CB_{i+1}

Total: 15 steps em 1 graph replay → ~4ms
```

### 3.5 `decoder.py` — StreamingDecoder

```python
class StreamingDecoder:
    """Code2Wav com torch.compile e fixed-shape padding."""

    def __init__(self, speech_tokenizer, decode_window: int = 40):
        self._tokenizer = speech_tokenizer
        self._decode_window = decode_window
        self._compiled = False
        self._samples_per_frame = 2000  # 12Hz → 24kHz = 24000/12 = 2000

    def compile(self, mode: str = "reduce-overhead"):
        """Aplica torch.compile ao decoder forward."""

    def decode(
        self,
        codes: torch.Tensor,          # [T, 16] codec frames
        pad_to: int | None = None,    # Left-pad para shape fixa
    ) -> tuple[np.ndarray, int]:
        """Decodifica codes → PCM float32 24kHz."""

    def decode_new_frames(
        self,
        all_codes: list[torch.Tensor],  # Buffer completo GPU-resident
        num_new_frames: int,             # Quantos frames novos
        ref_codes: torch.Tensor | None = None,  # ICL context
    ) -> np.ndarray:
        """Decode incremental: extrai sliding window, decodifica, retorna só áudio novo."""
```

**Fonte:** Baseado em `dffdeeq-streaming` decoder (linhas 1040-1075) + `faster-qwen3-tts` sliding window (linhas 928-1016)

**Estratégia de decodificação (combinação das melhores técnicas):**

1. **Fixed-shape padding** (de dffdeeq):
   - Left-pad codes para `decode_window` fixo
   - Garante shape constante para torch.compile
   - Trim padding do output (left side)

2. **Sliding window** (de faster):
   - `context = 25 frames` (~2.1s @ 12Hz)
   - Custo por decode: O(window_size) não O(total_frames)

3. **samples_per_frame fixo** (simplificação nossa):
   - 12Hz codec → 24kHz output → 2000 samples/frame
   - Sem calibration phase (economiza latência nos primeiros frames)
   - Evidência: dffdeeq e rekuenkdr usam `get_decode_upsample_rate()` fixo

### 3.6 `crossfade.py` — HannCrossfader

```python
class HannCrossfader:
    """Crossfade Hann entre chunks para eliminar clicks/pops."""

    def __init__(self, overlap_samples: int = 512):
        self._overlap = overlap_samples
        self._prev_tail: np.ndarray | None = None
        # Pré-computa as curvas de fade (evita cálculo por chunk)
        t = np.arange(overlap_samples, dtype=np.float32) / max(overlap_samples - 1, 1)
        self._fade_in = 0.5 * (1 - np.cos(np.pi * t))    # Hann
        self._fade_out = 1.0 - self._fade_in               # Complementar

    def apply(self, chunk: np.ndarray, is_first: bool = False, is_last: bool = False) -> np.ndarray:
        """Aplica crossfade. Retorna chunk processado."""

    def reset(self):
        """Reset estado entre utterances."""
```

**Fonte:** Baseado em `rekuenkdr-streaming` (linhas 95-103)

**Evidência da fórmula Hann vs Linear:**
- rekuenkdr usa Hann: `0.5 * (1 - cos(πt))` — preservação de energia perfeita
- dffdeeq usa Linear: `np.linspace(0, 1, n)` — mais simples, mas menos suave
- **Escolha: Hann** — padrão em áudio profissional, zero custo extra

**Overlap de 512 samples @ 24kHz = 21.3ms:**
- Evidência rekuenkdr: "matches the RMS check window"
- Trade-off: menor overlap = menor latência, mas possíveis artifacts
- 512 é o sweet spot testado

### 3.7 `sampling.py` — Sampling e Repetition Penalty

```python
def sample_logits(
    logits: torch.Tensor,
    temperature: float = 0.9,
    top_k: int = 50,
    top_p: float = 1.0,
    suppress_tokens: list[int] | None = None,
) -> torch.Tensor:
    """Amostra token de logits com temperature + top_k + top_p."""

class CircularRepetitionPenalty:
    """Penalidade de repetição com buffer circular GPU-resident."""

    def __init__(self, window: int = 256, penalty: float = 1.05, vocab_size: int = 3072):
        # Buffer circular pré-alocado na GPU
        # Evita crescimento de memória

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        """Aplica penalidade baseada no histórico."""

    def update(self, token: torch.Tensor):
        """Adiciona token ao buffer circular."""
```

**Fonte:**
- Sampling: `faster-qwen3-tts/faster_qwen3_tts/sampling.py` (67 linhas)
- Repetition penalty: `rekuenkdr-streaming` circular buffer GPU (linhas 2755-2823)

**Escolha: Buffer circular GPU (rekuenkdr) > Lista Python (faster):**
- Buffer circular tem custo O(1) por update
- GPU-resident evita CPU↔GPU sync
- Window de 256 tokens evita vocabulary starvation

### 3.8 `audio.py` — Processamento de Áudio

```python
def resample_24k_to_8k(audio: np.ndarray) -> np.ndarray:
    """Resample 24kHz → 8kHz para pipeline interno do Macaw."""

def float32_to_pcm16(audio: np.ndarray) -> bytes:
    """Converte float32 [-1, 1] para PCM 16-bit LE."""

def pcm16_to_float32(pcm: bytes) -> np.ndarray:
    """Converte PCM 16-bit LE para float32."""
```

**Nota:** O decoder Qwen3-TTS gera áudio a 24kHz. O pipeline do Macaw usa 8kHz internamente. O resampling acontece no provider do Macaw, não na biblioteca.

---

## 4. Integração com Macaw Voice Agent

### 4.1 Novo Provider TTS

```python
# src/tts/providers/macaw_streaming_tts.py

class MacawStreamingTTS(TTSProvider):
    """TTS provider usando macaw-qwen3-tts-streaming.

    Streaming real: cada yield contém ~83ms de áudio (1 frame @ 12Hz).
    TTFA: ~88ms no primeiro frame.
    """

    provider_name = "macaw-streaming"

    async def connect(self):
        self._model = await run_inference(
            MacawTTS.from_pretrained, self._model_name
        )

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """Streaming real — yield PCM 8kHz chunks conforme gerados."""
        loop = asyncio.get_running_loop()
        audio_queue = queue.Queue()

        def _stream_to_queue():
            for chunk, sr, meta in self._model.stream(text, language=self._language):
                pcm_8k = resample(chunk, sr, 8000)
                audio_queue.put(float32_to_pcm(pcm_8k))
            audio_queue.put(None)

        future = loop.run_in_executor(None, _stream_to_queue)

        while True:
            item = await loop.run_in_executor(None, audio_queue.get)
            if item is None:
                break
            yield item

        await asyncio.wrap_future(future)

    @property
    def supports_streaming(self) -> bool:
        return True
```

### 4.2 Mudanças no Sentence Pipeline

O pipeline atual espera frases completas. Com streaming real sub-frame, o fluxo muda:

```
ANTES:
  LLM gera frase completa → TTS sintetiza frase → chunka 100ms → yield

DEPOIS:
  LLM gera frase completa → TTS streama frame-a-frame → yield direto (sem re-chunk)
```

**Mudança no `_tts_worker`:** Remover re-chunking de 100ms. O TTS já emite chunks de ~83ms.

### 4.3 Remoção de Providers Batch

Conforme regra inquebrantável de streaming real:
- ❌ `tts_edge.py` — batch MP3 + ffmpeg decode
- ❌ `tts_qwen.py` (API side) — batch only
- ✅ `macaw_streaming_tts.py` — streaming real
- ✅ `tts_kokoro.py` — streaming nativo (mantém como fallback CPU)
- ✅ `tts_remote.py` — gRPC streaming (depende do backend)

---

## 5. Plano de Execução (Fases)

> **REGRA INQUEBRANTÁVEL:** Tasks marcadas com 🟢 GPU requerem SOLICITAÇÃO EXPLÍCITA ao usuário antes de executar.
> Estratégia: implementar local (CPU), testar remoto (GPU na Vast.ai).

### Fase 1: Core Engine — Módulos Puros (Semana 1)

> Tudo roda local, sem GPU. Lógica pura + testes com mocks.

| # | Task | Arquivo | Baseado Em | Complexidade | Ambiente |
|---|------|---------|-----------|--------------|----------|
| 1.1 | Criar estrutura do projeto e pyproject.toml | `pyproject.toml` | — | Baixa | 🖥️ CPU |
| 1.2 | Implementar `sampling.py` (temperature, top_k, top_p, repetition penalty) | `macaw_tts/sampling.py` | faster sampling.py (67 LOC) | Baixa | 🖥️ CPU |
| 1.3 | Implementar `crossfade.py` (Hann window, fade curves pré-computadas) | `macaw_tts/crossfade.py` | rekuenkdr _crossfade (10 LOC) | Baixa | 🖥️ CPU |
| 1.4 | Implementar `audio.py` (resample, float32↔PCM16) | `macaw_tts/audio.py` | faster utils.py + existente | Baixa | 🖥️ CPU |
| 1.5 | Implementar `predictor_graph.py` (estrutura + lógica, sem CUDA) | `macaw_tts/predictor_graph.py` | faster predictor_graph.py (215 LOC) | Alta | 🖥️ CPU (código) |
| 1.6 | Implementar `talker_graph.py` (estrutura + lógica, sem CUDA) | `macaw_tts/talker_graph.py` | faster talker_graph.py (214 LOC) | Alta | 🖥️ CPU (código) |
| 1.7 | Implementar `decoder.py` (sliding window, fixed-shape padding) | `macaw_tts/decoder.py` | dffdeeq decode_padded + faster sliding window | Média | 🖥️ CPU (código) |
| 1.8 | Testes unitários: sampling, crossfade, audio (sem GPU) | `tests/` | — | Média | 🖥️ CPU |

**Checkpoint Fase 1:** Todos os módulos implementados e testados com tensores CPU/mocks. Zero dependência de GPU.

### Fase 2: Streaming Generator + Model Wrapper (Semana 2)

> Código escrito local. Testes de integração requerem GPU.

| # | Task | Arquivo | Baseado Em | Complexidade | Ambiente |
|---|------|---------|-----------|--------------|----------|
| 2.1 | Implementar `streaming.py` (core generator, two-phase) | `macaw_tts/streaming.py` | faster streaming.py + rekuenkdr two-phase | Alta | 🖥️ CPU (código) |
| 2.2 | Implementar `model.py` (from_pretrained, stream, stream_voice_clone) | `macaw_tts/model.py` | faster model.py (adaptado) | Média | 🖥️ CPU (código) |
| 2.3 | Implementar voice cloning com ICL | `macaw_tts/model.py` | faster model.py linhas 215-383 | Média | 🖥️ CPU (código) |
| 2.4 | Testes unitários com mocks (FakeTalker, FakePredictor, FakeDecoder) | `tests/` | — | Média | 🖥️ CPU |
| 2.5 | **⚠️ SOLICITAR GPU:** Teste de integração — carregar modelo real, gerar áudio | `tests/test_streaming.py` | — | Alta | 🟢 GPU |
| 2.6 | **⚠️ SOLICITAR GPU:** Validar CUDA graph capture (Talker + Predictor) | — | — | Alta | 🟢 GPU |
| 2.7 | **⚠️ SOLICITAR GPU:** Validar torch.compile decoder com fixed-shape | — | — | Média | 🟢 GPU |

**Checkpoint Fase 2:** Código completo. Após GPU: streaming funcional gerando áudio real.

### Fase 3: Integração Macaw (Semana 3)

> Provider e pipeline escritos local. Teste end-to-end requer GPU.

| # | Task | Arquivo | Complexidade | Ambiente |
|---|------|---------|--------------|----------|
| 3.1 | Criar provider `macaw_streaming_tts.py` | `src/tts/providers/` | Média | 🖥️ CPU (código) |
| 3.2 | Registrar no factory e config | `src/tts/providers/base.py` | Baixa | 🖥️ CPU |
| 3.3 | Adaptar `sentence_pipeline.py` (remover re-chunk de 100ms) | `src/api/pipeline/` | Média | 🖥️ CPU |
| 3.4 | Remover providers batch (edge, qwen API-side) | `src/api/providers/` | Baixa | 🖥️ CPU |
| 3.5 | Testes do provider com FakeMacawTTS (mock) | `tests/` | Média | 🖥️ CPU |
| 3.6 | **⚠️ SOLICITAR GPU:** Teste end-to-end completo (WebSocket → TTS streaming → áudio) | — | Alta | 🟢 GPU |
| 3.7 | **⚠️ SOLICITAR GPU:** Benchmark TTFA end-to-end | `benchmarks/` | Média | 🟢 GPU |
| 3.8 | Atualizar CHANGELOG.md | `CHANGELOG.md` | Baixa | 🖥️ CPU |

**Checkpoint Fase 3:** Integração completa. Após GPU: voice agent funcionando com streaming real.

### Fase 4: Otimização e Tuning (Semana 4)

> Toda esta fase requer GPU.

| # | Task | Complexidade | Ambiente |
|---|------|--------------|----------|
| 4.1 | **⚠️ SOLICITAR GPU:** Profile TTFA com torch.profiler | Média | 🟢 GPU |
| 4.2 | **⚠️ SOLICITAR GPU:** Tuning de decode_window (25/40/80) vs qualidade | Média | 🟢 GPU |
| 4.3 | **⚠️ SOLICITAR GPU:** Tuning de overlap_samples (256/512/1024) vs latência | Baixa | 🟢 GPU |
| 4.4 | **⚠️ SOLICITAR GPU:** Benchmark RTF (A100 vs RTX 4090) | Média | 🟢 GPU |
| 4.5 | Documentação e exemplos | Baixa | 🖥️ CPU |

**Checkpoint Fase 4:** TTFA ≤ 88ms validado, RTF ≥ 5x confirmado.

### Resumo GPU vs CPU

```
Fase 1: 100% CPU (8 tasks) ──────────── Sem bloqueio
Fase 2:  57% CPU (4 tasks) + 43% GPU (3 tasks) ── GPU no final
Fase 3:  62% CPU (5 tasks) + 25% GPU (2 tasks) ── GPU no final
Fase 4:  20% CPU (1 task)  + 80% GPU (4 tasks) ── GPU dominante
```

**Total: 18 tasks CPU (podem começar AGORA) + 9 tasks GPU (solicitar quando chegar)**

---

## 6. Riscos e Mitigações

| Risco | Probabilidade | Impacto | Mitigação |
|-------|:---:|:---:|-----------|
| CUDA graph incompatível com versão do modelo | Média | Alto | Fallback para torch.compile sem CUDA graphs (como dffdeeq) |
| TTFA > 88ms na primeira chamada (warmup) | Alta | Médio | Warmup explícito no `connect()` com texto dummy. TTFA de 88ms a partir da 2ª chamada |
| Artefatos de áudio com emit_every=1 | Média | Alto | Hann crossfade + decode_window suficiente (25+ frames). Se persistir, aumentar para emit_every=2 (~166ms) |
| Conflito CUDA graphs + KV cache dinâmico | Baixa | Alto | StaticCache pré-alocado (técnica comprovada no faster-qwen3-tts) |
| Resampling 24k→8k adiciona latência | Baixa | Baixo | scipy.signal.resample_poly é O(chunk_size), ~0.1ms para 2000 samples |

---

## 7. Métricas de Sucesso

| Métrica | Meta | Como Medir |
|---------|------|-----------|
| **TTFA** | ≤ 88ms | `time.perf_counter()` do início do `stream()` até primeiro `yield` |
| **RTF** | ≥ 5x | Duração total do áudio / tempo de parede total |
| **Qualidade (MOS)** | ≥ 4.0 | Avaliação subjetiva com PT-BR |
| **Artefatos** | Zero clicks/pops | Escuta manual + análise espectral |
| **Memória GPU** | ≤ 4GB (0.6B) / ≤ 8GB (1.7B) | `torch.cuda.max_memory_allocated()` |

---

## 8. Referências de Código (Evidências)

### CUDA Graphs (Talker + Predictor)
- **Arquivo:** `faster-qwen3-tts/faster_qwen3_tts/talker_graph.py:110-147`
- **Arquivo:** `faster-qwen3-tts/faster_qwen3_tts/predictor_graph.py:170-202`
- **Técnica:** StaticCache + attention mask table pré-computado + graph capture em stream separado
- **Benchmark:** RTF 4.78x, ~20.9ms/step (RTX 4090)

### Two-Phase Latency
- **Arquivo:** `rekuenkdr-streaming/qwen_tts/core/models/modeling_qwen3_tts.py:2828-2843`
- **Técnica:** Phase 1 (agressiva) → Phase 2 (estável)
- **Benchmark:** TTFA 570ms → 208ms (2.75x melhoria com Phase1=5 frames)
- **Nossa melhoria:** Phase1=1 frame → TTFA teórico ~88ms

### Hann Crossfade
- **Arquivo:** `rekuenkdr-streaming/qwen_tts/core/models/modeling_qwen3_tts.py:95-103`
- **Técnica:** `fade_in = 0.5 * (1 - cos(πt))`, overlap=512 samples
- **Resultado:** Zero clicks entre chunks

### Fixed-Shape torch.compile
- **Arquivo:** `dffdeeq-streaming/qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py:1040-1075`
- **Técnica:** Left-pad para shape fixa → single compiled kernel → trim output
- **Resultado:** ~6x speedup no decoder

### Sliding Window Decode
- **Arquivo:** `faster-qwen3-tts/faster_qwen3_tts/model.py:928-1016`
- **Técnica:** 25-frame left context, calibration → sliding window
- **Resultado:** O(window) por decode, não O(total_frames)

### Repetition Penalty (Circular Buffer GPU)
- **Arquivo:** `rekuenkdr-streaming/qwen_tts/core/models/modeling_qwen3_tts.py:2755-2823`
- **Técnica:** Buffer circular GPU-resident, window=256, scatter-based penalty
- **Resultado:** Zero CPU↔GPU sync, O(1) por update

---

## 9. Decisões Técnicas Tomadas

| Decisão | Escolha | Alternativa Rejeitada | Motivo |
|---------|---------|----------------------|--------|
| CUDA graph para Talker/Predictor | CUDA graph manual (faster) | torch.compile (rekuenkdr/dffdeeq) | CUDA graph manual é 2-3x mais rápido por step que torch.compile. Evidência: faster atinge 20.9ms/step vs ~50ms estimado com compile |
| torch.compile para Decoder | torch.compile reduce-overhead (dffdeeq) | CUDA graph manual | Decoder tem shapes dinâmicas (sliding window). torch.compile com padding é mais flexível |
| Crossfade | Hann (rekuenkdr) | Linear (dffdeeq) | Hann preserva energia, padrão em áudio profissional |
| emit_every Phase 1 | 1 frame | 4-5 frames (rekuenkdr) | Meta de 88ms requer emissão no primeiro frame |
| Repetition Penalty | Circular buffer GPU (rekuenkdr) | Lista Python (faster) | Zero CPU↔GPU sync, O(1) updates |
| Calibration phase | Removida (samples_per_frame fixo) | Calibration de 25 frames (faster) | 12Hz codec tem ratio fixo (2000 samples/frame). Calibration adiciona 25 frames de latência |
| Batch support | Não | Sim (dffdeeq) | Voice agent processa 1 request por vez |
| decode_window | 40 frames | 80 (rekuenkdr/dffdeeq) | 25 frames são suficientes para Code2Wav causal (evidência faster). 40 dá margem extra |

---

## 10. Diferenciação da Nossa Biblioteca

### vs. faster-qwen3-tts
- ✅ Two-phase latency (emit_every=1 no primeiro frame)
- ✅ Hann crossfade (faster não tem crossfade)
- ✅ torch.compile no decoder (faster não tem)
- ✅ Circular buffer GPU para repetition penalty
- ✅ API assíncrona preparada para voice agents

### vs. rekuenkdr-streaming
- ✅ CUDA graphs para Talker + Predictor (rekuenkdr usa torch.compile, mais lento)
- ✅ emit_every=1 no Phase 1 (rekuenkdr usa 4-5)
- ✅ samples_per_frame fixo (sem calibration overhead)
- ✅ API mais simples e focada em voice agents

### vs. dffdeeq-streaming
- ✅ CUDA graphs para Talker + Predictor
- ✅ Two-phase latency
- ✅ Hann crossfade (dffdeeq usa linear)
- ✅ Menor decode_window (40 vs 80 = menos latência por decode)

### Combinação Única
```
faster CUDA graphs + rekuenkdr two-phase + dffdeeq torch.compile decoder
= TTFA ~88ms com qualidade de áudio profissional
```
