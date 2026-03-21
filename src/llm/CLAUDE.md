# LLM Server

Standalone gRPC Language Model microservice with pluggable providers.

## Commands

```bash
# Run server (from src/)
PYTHONPATH=. python3 -m llm.server

# Run with vLLM backend
LLM_BACKEND_PROVIDER=vllm VLLM_BASE_URL=http://localhost:8000/v1 PYTHONPATH=. python3 -m llm.server

# Docker build (from src/)
docker build -f llm/Dockerfile.qwen -t llm-server-qwen .
```

## Architecture

- **Single file server:** `server.py` contains `LLMServicer` (gRPC handlers) and `LLMServer` (lifecycle)
- **Self-contained providers:** `llm/providers/` — base, vllm_provider
- **Common modules:** `common/` — config, grpc_server
- **Proto stubs:** `shared/grpc_gen/llm_service_pb2{,_grpc}.py`

## Key Points

- Port: 50080 (env: `GRPC_PORT`)
- Protocol: unary-stream gRPC (client sends messages, server streams StreamEvent)
- StreamEvent maps 1:1 to LLMStreamEvent: text_delta, tool_call_start, tool_call_delta, tool_call_end
- Tool calling: tools passed as ToolDefinition protos, server converts to OpenAI format
- Messages: ChatMessage protos converted to OpenAI-format dicts internally
- gRPC keepalive: 30s ping, 10s timeout, health + reflection enabled
- Graceful shutdown: 5s grace period

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_BACKEND_PROVIDER` | `vllm` | Backend provider: vllm, anthropic, openai, mock |
| `VLLM_BASE_URL` | `http://localhost:8000/v1` | vLLM OpenAI-compatible endpoint |
| `LLM_MODEL` | `Qwen/Qwen2.5-7B-Instruct-AWQ` | Model name |
| `LLM_TIMEOUT` | `30.0` | Request timeout in seconds |
| `GRPC_PORT` | `50080` | gRPC listen port |
| `LOG_LEVEL` | `INFO` | Logging level |
