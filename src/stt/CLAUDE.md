# STT Server

Standalone gRPC Speech-to-Text microservice with pluggable providers.

## Commands

```bash
# Run server (from src/)
PYTHONPATH=. python3 -m stt.server

# Run tests (from src/)
PYTHONPATH=. pytest stt/tests/ -v

# Docker build (from src/)
docker build -f stt/Dockerfile.whisper -t stt-server-whisper .
```

## Architecture

- **Single file server:** `server.py` contains `STTServicer` (gRPC handlers) and `STTServer` (lifecycle)
- **Self-contained providers:** `stt/providers/` — base, whisper_stt, qwen_stt
- **Common modules:** `common/` — config, audio_utils, executor
- **Proto stubs:** `shared/grpc_gen/stt_service_pb2{,_grpc}.py`

## Key Points

- Port: 50060
- Audio format: PCM 8kHz 16-bit mono (all providers)
- Streaming: bidirectional gRPC, state isolated by `stream_id`
- gRPC keepalive: 30s ping, 10s timeout, health + reflection enabled
- Graceful shutdown: 5s grace period, `provider.disconnect()` called
- Empty audio → `INVALID_ARGUMENT`, provider error → `INTERNAL`

## Dockerfiles

- `Dockerfile` — CPU/mock (`python:3.11-slim`)
- `Dockerfile.whisper` — Faster-Whisper (`nvidia/cuda:12.4.1`)
