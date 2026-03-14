# TTS Server

Standalone gRPC Text-to-Speech microservice. Supports batch and streaming synthesis with pluggable TTS engines.

## Architecture

```
gRPC Client (AI Agent, open-voice-api, etc.)
       │
       │  SynthesizeRequest / stream AudioChunk
       ▼
┌──────────────────────────────────────────┐
│          TTS Server (:50070)             │
│                                          │
│  ┌─────────────────────────────────┐    │
│  │        TTSServicer              │    │
│  │  Synthesize()       — batch     │    │
│  │  SynthesizeStream() — streaming │    │
│  └──────────────┬──────────────────┘    │
│                 │                        │
│                 ▼                        │
│  ┌─────────────────────────────────┐    │
│  │      TTS Provider (ABC)         │    │
│  │  mock | kokoro | faster | qwen  │    │
│  └─────────────────────────────────┘    │
└──────────────────────────────────────────┘
```

## Quick Start

```bash
# CPU / Mock (no GPU needed)
cd tts-server
pip install -r requirements.txt
TTS_PROVIDER=mock python server.py

# Docker (CPU)
docker build -t tts-server -f Dockerfile .
docker run -p 50070:50070 tts-server

# Docker (GPU — Qwen3-TTS via faster-qwen3-tts)
docker build -t tts-server-gpu -f Dockerfile.gpu .
docker run --gpus all -p 50070:50070 tts-server-gpu

# Docker (GPU — Kokoro-ONNX)
docker build -t tts-server-kokoro -f Dockerfile.kokoro-gpu .
docker run --gpus all -p 50070:50070 tts-server-kokoro
```

## gRPC API

Defined in [`proto/tts_service.proto`](../proto/tts_service.proto).

### `Synthesize` (unary)

Send text, receive complete audio.

```
SynthesizeRequest {
  text: string            # Text to synthesize
  language: string        # "pt", "en", etc.
  output_config: optional # sample rate, encoding
  voice_id: string        # Optional voice selection
}

SynthesizeResponse {
  audio_data: bytes       # PCM 8kHz 16-bit mono
  duration_ms: float      # Audio duration
  processing_ms: float    # Synthesis latency
}
```

### `SynthesizeStream` (server streaming)

Send text, receive audio chunks as they're generated.

```
→ SynthesizeRequest { text, language, ... }
← stream AudioChunk { audio_payload, is_last, sequence }
```

- Chunks arrive as the engine generates them (TTFB ~156ms for Qwen3)
- Last chunk: `is_last=true` (may be empty sentinel)
- `sequence` field for chunk ordering

## TTS Engines

| Engine | `TTS_PROVIDER=` | GPU | Streaming | TTFB | Quality |
|--------|-----------------|-----|-----------|------|---------|
| Mock | `mock` | No | No | ~100ms | N/A (tone) |
| Kokoro-ONNX | `kokoro` | Optional | Yes | ~500ms | Good |
| FasterQwen3-TTS | `faster` | Yes | Yes | ~156ms | Excellent |
| Qwen3-TTS | `qwen` | Yes | No | ~800ms | Excellent |

### Recommended for Production

- **CPU-only:** `kokoro` (82M params, native PT-BR via `pf_dora` voice)
- **GPU:** `faster` (Qwen3-TTS-1.7B, CUDA graphs, 156ms TTFB, streaming)

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GRPC_HOST` | `0.0.0.0` | Bind address |
| `GRPC_PORT` | `50070` | Listen port |
| `TTS_PROVIDER` | `mock` | TTS engine |
| `TTS_LANGUAGE` | `pt` | Default language |
| `KOKORO_VOICE` | `pf_dora` | Kokoro voice (PT-BR recommended) |
| `KOKORO_LANG` | `pt-br` | Kokoro language code |
| `KOKORO_SPEED` | `1.0` | Speech speed (0.5-2.0) |
| `QWEN_TTS_MODEL` | `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | Qwen model |
| `QWEN_TTS_SPEAKER` | `Ryan` | Qwen speaker voice |
| `QWEN_TTS_LANGUAGE` | `Portuguese` | Qwen language |
| `FASTER_TTS_CHUNK_SIZE` | `4` | Tokens per streaming chunk |
| `LOG_LEVEL` | `INFO` | Logging level |

## Health Check

gRPC health service registered as `theo.tts.TTSService`. gRPC reflection enabled.

```bash
# Check health
grpcurl -plaintext localhost:50070 grpc.health.v1.Health/Check
```

## Tests

```bash
cd tts-server
PYTHONPATH=../ai-agent:../shared pytest tests/ -v
```

## Audio Format

- **Output:** PCM 8kHz 16-bit mono (all providers resample internally)
- **Provider native rates:** Kokoro 24kHz, Qwen3-TTS 12kHz — resampled to 8kHz
- **Max message size:** 10MB (configurable via `GRPC_MAX_MESSAGE_SIZE`)
