<p align="center">
  <h1 align="center">Macaw-Qwen3-TTS-Streaming</h1>
  <p align="center">
    True streaming TTS for Qwen3-TTS with CUDA graphs and two-phase latency.
    <br />
    <strong>~88ms time-to-first-audio on a single GPU.</strong>
  </p>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &bull;
  <a href="#how-it-works">How It Works</a> &bull;
  <a href="#architecture">Architecture</a> &bull;
  <a href="#api-reference">API Reference</a> &bull;
  <a href="#performance">Performance</a>
</p>

---

## Overview

`macaw-qwen3-tts-streaming` wraps [Qwen3-TTS](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-0.6B-Base) with CUDA graph acceleration and a two-phase emission strategy to produce audio **frame-by-frame** as the model generates tokens.

Standard Qwen3-TTS generates the full utterance before returning any audio. This library starts streaming audio after a single codec frame (~83ms of audio), making it suitable for real-time voice applications where latency is critical.

Built for the [Macaw Voice Agent](https://github.com/usemacaw/macaw-voice-agent) voice-to-voice pipeline.

## Quick Start

```python
from macaw_tts import MacawTTS

# Load model and capture CUDA graphs (one-time, ~10s)
tts = MacawTTS.from_pretrained("Qwen/Qwen3-TTS-12Hz-0.6B-Base")

# Stream audio chunk-by-chunk
for audio, sample_rate, meta in tts.stream("Olá, como posso ajudar?", language="Portuguese"):
    play(audio, sample_rate)  # float32 numpy @ 24kHz
    if meta.chunk_index == 0:
        print(f"First audio in {meta.ttfa_ms:.0f}ms")
```

**Voice cloning:**

```python
for audio, sr, meta in tts.stream_voice_clone(
    text="Texto a ser falado.",
    language="Portuguese",
    ref_audio="reference.wav",
    ref_text="Transcrição do áudio de referência.",
):
    play(audio, sr)
```

## Installation

```bash
git clone <repo-url>
cd qwen3-tts-streaming
pip install -e ".[dev]"
```

| Dependency | Version | Notes |
|------------|---------|-------|
| Python | >= 3.10 | |
| PyTorch | >= 2.5.1 | CUDA required |
| transformers | >= 4.45 | For StaticCache and masking utils |
| scipy | >= 1.10 | Polyphase resampling |
| qwen-tts | latest | Qwen3-TTS model package (not on PyPI, install from source) |

## How It Works

Qwen3-TTS generates speech in three sequential stages. This library makes all three stages streamable by capturing them as CUDA graphs and emitting audio incrementally:

```
Text ──> Talker (28 layers) ──> CB0 token
                                   |
                     CodePredictor (5 layers) ──> CB1-CB15 (15 codebook tokens)
                                   |
                     Code2Wav (causal ConvNet) ──> PCM float32 @ 24kHz
```

### Optimization techniques

| Technique | What it does | Latency impact |
|-----------|-------------|----------------|
| **CUDA graph — Talker** | Captures single-token decode as a replayable graph | ~8ms/step (vs ~50ms interpreted) |
| **CUDA graph — CodePredictor** | Captures full 15-codebook loop as one graph | ~4ms per frame (vs ~20ms) |
| **torch.compile — Code2Wav** | Compiles decoder with fixed-shape padding | ~3-5ms per decode |
| **Two-phase emission** | Phase 1: emit after 1 frame; Phase 2: every 4 | TTFA ~88ms without sacrificing throughput |
| **Hann crossfade** | Overlap-add blending at chunk boundaries | Eliminates clicks/pops |
| **Sliding window** | Decodes only the last N frames, not the full sequence | O(window) per decode |
| **GPU-resident repetition penalty** | Circular buffer on GPU, no CPU sync | Zero overhead per token |

### Two-phase emission strategy

The library uses an adaptive emission rate to balance latency and throughput:

```
Frame 1          Frame 2    3    4    5          6    7    8    9
  |                |    |    |    |                |    |    |    |
  +--> emit (P1)   +----+----+----+--> emit (P2)  +----+----+----+--> emit (P2)
  ~88ms TTFA                ~333ms/chunk                ~333ms/chunk
```

- **Phase 1** emits audio after every frame. Gets the first audio to the speaker as fast as possible.
- **Phase 2** batches 4 frames per emission. Amortizes decode overhead for higher throughput.

## Architecture

```
macaw_tts/
├── model.py            MacawTTS facade (from_pretrained, stream, stream_voice_clone)
├── streaming.py        Generation loop, two-phase emission, prefill
├── talker_graph.py     CUDA graph capture for Talker (StaticCache + attention masks)
├── predictor_graph.py  CUDA graph capture for CodePredictor (15-step autoregressive)
├── decoder.py          Code2Wav with torch.compile, fixed-shape padding, sliding window
├── crossfade.py        Hann window overlap-add (stateful, pre-computed curves)
├── sampling.py         Token sampling (top-k/p, temperature) + circular repetition penalty
└── audio.py            Sample rate constants, polyphase resampling, PCM16 conversion
```

**Module dependency graph:**

```
audio ◄── decoder ◄── streaming ◄── model
                         ▲             ▲
crossfade ───────────────┤             │
sampling ──── predictor_graph ─────────┤
              talker_graph ────────────┘
```

No circular dependencies. Leaf modules (`audio`, `crossfade`, `sampling`) have zero internal imports.

## API Reference

### `MacawTTS.from_pretrained()`

```python
tts = MacawTTS.from_pretrained(
    model_name="Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    device="cuda",
    dtype=torch.bfloat16,
    max_seq_len=2048,       # KV cache limit for CUDA graphs
    decode_window=40,       # Fixed window for torch.compile
    context_frames=25,      # Left context for decode quality
    compile_decoder=True,   # torch.compile on Code2Wav
)
```

Loads the model, builds CUDA graph wrappers, and optionally compiles the decoder. CUDA graphs are captured lazily on the first call to `stream()`.

### `MacawTTS.stream()`

```python
for audio, sample_rate, meta in tts.stream(
    text="Hello, world.",
    language="English",      # "Portuguese", "Chinese", "English", etc.
    speaker=None,            # Named speaker from model config, or None
    temperature=0.9,
    top_k=50,
    top_p=1.0,               # 1.0 = disabled
    repetition_penalty=1.05,
    max_frames=2048,
    emit_every_phase1=1,     # Frames per Phase 1 emission
    emit_every_phase2=4,     # Frames per Phase 2 emission
    phase1_frames=1,         # How many frames before switching to Phase 2
    overlap_samples=512,     # Hann crossfade overlap (~21ms @ 24kHz)
):
    ...
```

Returns a generator yielding `(audio_chunk, sample_rate, ChunkMetadata)` tuples.

### `MacawTTS.stream_voice_clone()`

Same interface as `stream()`, plus `ref_audio` (path to WAV), `ref_text` (transcript), and optional pre-built `voice_clone_prompt`.

### `ChunkMetadata`

| Field | Type | Description |
|-------|------|-------------|
| `chunk_index` | `int` | 0-based sequential chunk number |
| `num_frames` | `int` | Codec frames decoded in this chunk |
| `phase` | `int` | `1` = aggressive latency, `2` = stable throughput |
| `is_final` | `bool` | `True` on the last chunk of the utterance |
| `decode_ms` | `float` | Wall-clock time to decode this chunk |
| `total_frames_so_far` | `int` | Cumulative frames emitted |
| `ttfa_ms` | `float` | Time-to-first-audio (non-zero only on first chunk) |

### Audio utilities

```python
from macaw_tts import (
    CODEC_SAMPLE_RATE,       # 24000 — Qwen3-TTS native output rate
    INTERNAL_SAMPLE_RATE,    # 8000  — Macaw Voice Agent telephony rate
    SAMPLES_PER_FRAME,       # 1920  — 12Hz codec @ 24kHz
    resample_to_internal,    # 24kHz float32 -> 8kHz float32
    float32_to_pcm16,        # float32 [-1,1] -> PCM16 LE bytes
    pcm16_to_float32,        # PCM16 LE bytes -> float32 [-1,1]
)
```

## Performance

Benchmarked on RTX 4090 (2026-03-20):

| Configuration | RTF | TTFA | Verdict |
|---------------|-----|------|---------|
| Vanilla Qwen3-TTS (no streaming) | — | full utterance | Baseline |
| flash-attn + dynamic cache | 0.54x | — | Slower than real-time |
| flash-attn + torch.compile | crash | — | InternalTorchDynamoError |
| **SDPA + CUDA graphs (this library)** | **3.49x** | **64ms** | **Production** |

> **Why SDPA instead of flash-attn?** Flash-attention uses dynamic memory patterns incompatible with CUDA graph capture. SDPA with CUDA graphs delivers 6.5x better RTF than flash-attn with dynamic cache.

### Latency budget per frame

| Stage | Time | Method |
|-------|------|--------|
| Talker decode (1 token) | ~8ms | CUDA graph replay |
| CodePredictor (15 codebooks) | ~4ms | CUDA graph replay |
| Code2Wav decode (1 frame) | ~3-5ms | torch.compile |
| Crossfade + emission | ~1ms | NumPy |
| **Total per frame** | **~16-18ms** | |
| + 1 frame of audio | +83ms | 12Hz codec |
| **TTFA (Phase 1)** | **~88ms** | |

## Testing

```bash
pip install -e ".[dev]"
pytest -v
```

73 unit tests, all CPU-only. No GPU required to run the test suite.

| Test file | Module covered | Tests |
|-----------|---------------|-------|
| `test_sampling.py` | `sampling.py` | 12 |
| `test_crossfade.py` | `crossfade.py` | 12 |
| `test_audio.py` | `audio.py` | 16 |
| `test_decoder.py` | `decoder.py` | 11 |
| `test_cuda_graphs.py` | `talker_graph.py`, `predictor_graph.py` | 8 |
| `test_streaming_emitter.py` | `streaming.py` | 14 |

## Project structure

```
qwen3-tts-streaming/
├── macaw_tts/          # Library source (1,945 lines, 8 modules)
├── tests/              # Unit tests (867 lines, 73 tests)
├── pyproject.toml      # Package configuration
├── CHANGELOG.md        # Release history
└── README.md
```

## License

[MIT](https://opensource.org/licenses/MIT)
