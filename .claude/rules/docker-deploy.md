---
paths:
  - "src/**/Dockerfile*"
  - "src/docker-compose*.yml"
  - "src/entrypoint*.sh"
  - "deploy/**"
  - "docs/DEPLOYMENT.md"
  - "docs/GPU_PROVISIONING.md"
---

# Regras Docker e Deploy

## Imagens Base
- GPU: `nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04` (STT/TTS) ou `vllm/vllm-openai:v0.14.0` (Qwen3)
- CPU/Mock: `python:3.11-slim`

## Padrões
- Multi-stage build: base → deps → app
- Modelos pré-baixados no build (evitar cold start)
- `NVIDIA_VISIBLE_DEVICES=all`, `NVIDIA_DRIVER_CAPABILITIES=compute,utility`
- Healthcheck obrigatório em todo container (gRPC health ou HTTP)
- Usuário non-root em containers CPU

## Portas Padrão
- 50060: STT (gRPC)
- 50070: TTS (gRPC)
- 8000: vLLM (HTTP)
- 8765: API WebSocket

## Vast.ai
- Script de deploy: `src/api/scripts/deploy_vastai.sh`
- GPU query: `gpu_ram>=20 num_gpus=1 inet_up>=200 disk_space>=50 reliability>=0.95`
- Documentação completa em `docs/DEPLOYMENT.md`
