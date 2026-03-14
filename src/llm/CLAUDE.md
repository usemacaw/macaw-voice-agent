# LLM Server

Standalone LLM inference server using vLLM with Qwen2.5-7B-Instruct-AWQ.

## Commands

```bash
# Docker build (from src/)
docker build -f llm/Dockerfile.qwen -t llm-server-qwen .

# Docker run
docker run --gpus all -p 8000:8000 llm-server-qwen

# Test
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen2.5-7B-Instruct-AWQ","messages":[{"role":"user","content":"Olá!"}],"max_tokens":100}'
```

## Architecture

- **Engine:** vLLM with OpenAI-compatible API
- **Model:** Qwen2.5-7B-Instruct-AWQ (4-bit quantized, ~5GB VRAM)
- **Function calling:** Enabled via `--enable-auto-tool-choice --tool-call-parser hermes`
- **Port:** 8000 (OpenAI-compatible `/v1/chat/completions`)

## Key Points

- AWQ 4-bit quantization keeps VRAM under 6GB
- `gpu-memory-utilization=0.85` leaves room for STT+TTS on shared GPU
- Model pre-downloaded at build time to avoid cold-start
- Supports streaming, function calling, system prompts
- PT-BR: Qwen2.5 has strong multilingual support including Portuguese
