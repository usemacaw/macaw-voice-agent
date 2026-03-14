# TTS Server

Standalone gRPC Text-to-Speech microservice with pluggable providers.

## Commands

```bash
# Run server (from src/)
PYTHONPATH=. python3 -m tts.server

# Run tests (from src/)
PYTHONPATH=. pytest tts/tests/ -v

# Docker build (from src/)
docker build -f tts/Dockerfile.kokoro-gpu -t tts-server-kokoro-gpu .
```

## Architecture

- **Single file server:** `server.py` contains `TTSServicer` (gRPC handlers) and `TTSServer` (lifecycle)
- **Self-contained providers:** `tts/providers/` — base, kokoro_tts, faster_tts, qwen_tts
- **Common modules:** `common/` — config, audio_utils, executor
- **Proto stubs:** `shared/grpc_gen/tts_service_pb2{,_grpc}.py`

## Key Points

- Port: 50070
- Audio format: PCM 8kHz 16-bit mono (providers resample from native rate)
- Streaming: server-streaming gRPC (client sends text, server streams AudioChunk)
- gRPC keepalive: 30s ping, 10s timeout, health + reflection enabled
- Graceful shutdown: 5s grace period, `provider.disconnect()` called
- Empty/whitespace text → `INVALID_ARGUMENT`, provider error → `INTERNAL`

## Dockerfiles

- `Dockerfile` — CPU/mock (`python:3.11-slim`)
- `Dockerfile.kokoro-gpu` — Kokoro-ONNX (`nvidia/cuda:12.4.1`)
