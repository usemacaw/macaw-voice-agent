# STT Server

Standalone gRPC Speech-to-Text microservice. Supports batch and bidirectional streaming transcription with pluggable STT engines.

## Architecture

```
gRPC Client (AI Agent, open-voice-api, etc.)
       │
       │  TranscribeRequest / AudioChunk stream
       ▼
┌──────────────────────────────────────────┐
│          STT Server (:50060)             │
│                                          │
│  ┌─────────────────────────────────┐    │
│  │        STTServicer              │    │
│  │  Transcribe()     — batch       │    │
│  │  TranscribeStream() — streaming │    │
│  └──────────────┬──────────────────┘    │
│                 │                        │
│                 ▼                        │
│  ┌─────────────────────────────────┐    │
│  │      STT Provider (ABC)         │    │
│  │  mock | qwen | qwen-streaming   │    │
│  │  whisper                        │    │
│  └─────────────────────────────────┘    │
└──────────────────────────────────────────┘
```

## Quick Start

```bash
# CPU / Mock (no GPU needed)
cd stt-server
pip install -r requirements.txt
STT_PROVIDER=mock python server.py

# Docker (CPU)
docker build -t stt-server -f Dockerfile .
docker run -p 50060:50060 stt-server

# Docker (GPU — Qwen3-ASR)
docker build -t stt-server-gpu -f Dockerfile.gpu .
docker run --gpus all -p 50060:50060 stt-server-gpu

# Docker (GPU — Faster-Whisper)
docker build -t stt-server-whisper -f Dockerfile.whisper .
docker run --gpus all -p 50060:50060 stt-server-whisper
```

## gRPC API

Defined in [`proto/stt_service.proto`](../proto/stt_service.proto).

### `Transcribe` (unary)

Send complete audio, receive full transcription.

```
TranscribeRequest {
  audio_data: bytes       # PCM 8kHz 16-bit mono
  audio_config: optional  # sample rate, encoding
  language: string        # "pt", "en", etc.
}

TranscribeResponse {
  text: string
  confidence: float
  processing_ms: float
}
```

### `TranscribeStream` (bidirectional streaming)

Send audio chunks incrementally, receive partial and final results.

```
→ AudioChunk { stream_id, audio_payload, audio_config (first only), end_of_stream }
← TranscribeResult { stream_id, text, is_final, confidence }
```

- Partial results: `is_final=false` with intermediate text
- Final result: `is_final=true` after `end_of_stream=true` received

## STT Engines

| Engine | `STT_PROVIDER=` | GPU | Streaming | Notes |
|--------|-----------------|-----|-----------|-------|
| Mock | `mock` | No | No | Returns fixed text. Development only |
| Qwen3-ASR | `qwen` | Yes | No | Batch only, 0.6B-1.7B params |
| Qwen3-ASR Streaming | `qwen-streaming` | Yes | Yes | vLLM backend, 1.7B, real streaming |
| Faster-Whisper | `whisper` | Yes | No | CTranslate2, large-v3-turbo, int8 |

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GRPC_HOST` | `0.0.0.0` | Bind address |
| `GRPC_PORT` | `50060` | Listen port |
| `STT_PROVIDER` | `mock` | STT engine |
| `STT_LANGUAGE` | `pt` | Default language |
| `QWEN_STT_MODEL` | `Qwen/Qwen3-ASR-1.7B` | Model name (qwen/qwen-streaming) |
| `QWEN_STT_GPU_MEM_UTIL` | `0.80` | GPU memory fraction (qwen-streaming) |
| `WHISPER_MODEL` | `large-v3-turbo` | Whisper model size |
| `WHISPER_COMPUTE_TYPE` | `int8` | Quantization (int8, float16, float32) |
| `WHISPER_BEAM_SIZE` | `1` | 1=greedy, >1=beam search |
| `LOG_LEVEL` | `INFO` | Logging level |

## Health Check

gRPC health service registered as `theo.stt.STTService`. gRPC reflection enabled.

```bash
# Check health
grpcurl -plaintext localhost:50060 grpc.health.v1.Health/Check
```

## Tests

```bash
cd stt-server
PYTHONPATH=../ai-agent:../shared pytest tests/ -v
```

## Audio Format

- **Input:** PCM 8kHz 16-bit mono (telephony standard)
- **Internal:** Providers may resample (e.g., Whisper needs 16kHz)
- **Max message size:** 10MB (configurable via `GRPC_MAX_MESSAGE_SIZE`)
