# Deployment — Macaw Voice Agent

Guia completo para reproduzir o ambiente de produção do Macaw Voice Agent na Vast.ai.

## Visão Geral da Arquitetura

O sistema roda distribuído em 3 componentes:

```
┌──────────────────────────────┐
│  Máquina Local (sem GPU)     │
│                              │
│  ┌────────────────────────┐  │
│  │  API Server (:8765)    │  │──── WebSocket ────▶ Browser (:5173)
│  │  python main.py        │  │
│  └──────┬───────┬─────────┘  │
│         │       │            │
│         │       │ HTTP :8100 │
│         │       │ (SSH tunnel)
│         │  ┌────▼─────────┐  │
│         │  │ vLLM (A100)  │  │  ◀── Vast.ai Instância #1
│         │  │ Qwen3-8B-AWQ │  │      SSH tunnel localhost:8100 → :8000
│         │  └──────────────┘  │
│         │                    │
│    gRPC │                    │
│         │                    │
└─────────┼────────────────────┘
          │
          ▼
┌──────────────────────────────┐
│  Vast.ai Instância #2 (GPU) │
│                              │
│  ┌──────────────────┐        │
│  │ STT Server       │ :10473 │  ◀── gRPC (público)
│  │ Whisper large-v3 │        │
│  └──────────────────┘        │
│                              │
│  ┌──────────────────┐        │
│  │ TTS Server       │ :10584 │  ◀── gRPC (público)
│  │ Kokoro v1.0      │        │
│  └──────────────────┘        │
└──────────────────────────────┘
```

## Configuração Atual em Produção

### Instância 1 — LLM (vLLM + Qwen3-8B-AWQ)

| Parâmetro | Valor |
|-----------|-------|
| **Provedor** | Vast.ai |
| **GPU** | A100 |
| **Modelo** | `Qwen/Qwen3-8B-AWQ` (4-bit AWQ, ~5GB VRAM) |
| **Max model length** | 8192 tokens |
| **GPU memory utilization** | 90% |
| **Tool calling** | Habilitado (Hermes parser) |
| **Porta interna** | 8000 |
| **Porta SSH** | 21068 |
| **SSH host** | `ssh5.vast.ai` |

**Comando de inicialização do vLLM:**

```bash
nohup python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B-AWQ \
  --quantization awq \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --port 8000 \
  --host 0.0.0.0 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  > /tmp/vllm.log 2>&1 &
```

**SSH tunnel (máquina local):**

```bash
ssh -N -L 8100:localhost:8000 -p 21068 root@ssh5.vast.ai
```

Isso expõe o vLLM em `localhost:8100` na máquina local.

### Instância 2 — STT + TTS (Whisper + Kokoro)

| Parâmetro | Valor |
|-----------|-------|
| **Provedor** | Vast.ai |
| **IP público** | `142.127.68.223` |
| **STT porta** | 10473 (gRPC) |
| **TTS porta** | 10584 (gRPC) |

**STT — Faster-Whisper:**

| Parâmetro | Valor |
|-----------|-------|
| Modelo | `large-v3-turbo` (~1.7GB VRAM) |
| Device | CUDA |
| Compute type | int8 |
| Beam size | 1 |
| VAD filter | Habilitado |
| Idioma | pt |
| Streaming | Habilitado |

**TTS — Kokoro-ONNX:**

| Parâmetro | Valor |
|-----------|-------|
| Modelo | `kokoro-v1.0.onnx` + `voices-v1.0.bin` (~0.5GB VRAM) |
| ONNX provider | CUDAExecutionProvider |
| Voice | `pf_dora` (português feminino) |
| Speed | 1.0 |
| Language | pt-br |

### API Server (máquina local)

**Arquivo:** `src/api/.env`

```bash
# --- WebSocket Server ---
WS_HOST=0.0.0.0
WS_PORT=8765
WS_PATH=/v1/realtime
REALTIME_API_KEY=
MAX_CONNECTIONS=10

# --- ASR (gRPC remoto → Vast.ai) ---
ASR_PROVIDER=remote
ASR_REMOTE_TARGET=142.127.68.223:10473
ASR_REMOTE_TIMEOUT=30.0
ASR_REMOTE_STREAMING=true
ASR_LANGUAGE=pt

# --- TTS (gRPC remoto → Vast.ai) ---
TTS_PROVIDER=remote
TTS_REMOTE_TARGET=142.127.68.223:10584
TTS_REMOTE_TIMEOUT=60.0
TTS_LANGUAGE=pt
TTS_VOICE=alloy

# --- LLM (vLLM via SSH tunnel) ---
LLM_PROVIDER=vllm
LLM_MODEL=Qwen/Qwen3-8B-AWQ
LLM_MAX_TOKENS=80
LLM_TEMPERATURE=0.7
LLM_TIMEOUT=30.0
VLLM_BASE_URL=http://localhost:8100/v1

# --- VAD ---
VAD_AGGRESSIVENESS=2
VAD_SILENCE_MS=150
VAD_PREFIX_PADDING_MS=300
VAD_MIN_SPEECH_MS=250

# --- Pipeline ---
PIPELINE_SENTENCE_QUEUE_SIZE=6
PIPELINE_TTS_PREFETCH_SIZE=4
PIPELINE_MAX_SENTENCE_CHARS=150
PIPELINE_TTS_TIMEOUT=15.0
PIPELINE_SENTENCE_TIMEOUT=15.0

# --- Tools ---
TOOL_ENABLE_MOCK=false
TOOL_ENABLE_WEB_SEARCH=true
TOOL_TIMEOUT=10.0
TOOL_MAX_ROUNDS=1
TOOL_DEFAULT_FILLER=Um momento, por favor.

# --- Logging ---
LOG_LEVEL=INFO
```

**Comando de inicialização:**

```bash
cd src/api && nohup python3 main.py > /tmp/macaw-api.log 2>&1 &
```

## Portas e Conexões

| Serviço | Host | Porta | Protocolo | Descrição |
|---------|------|-------|-----------|-----------|
| API Server | localhost | 8765 | WebSocket | Ponto de entrada para clientes |
| Vite Dev Server | localhost | 5173 | HTTP | Frontend React (desenvolvimento) |
| vLLM API | localhost | 8100 | HTTP | SSH tunnel → Vast.ai :8000 |
| STT Server | 142.127.68.223 | 10473 | gRPC | Faster-Whisper (Vast.ai) |
| TTS Server | 142.127.68.223 | 10584 | gRPC | Kokoro-ONNX (Vast.ai) |

## Reproduzindo o Ambiente

### Pré-requisitos

- Python 3.11+
- Node.js 18+
- Conta na [Vast.ai](https://vast.ai) com créditos
- `vastai` CLI instalado (`pip install vastai`)

### Passo 1 — Provisionar GPU para LLM (vLLM)

```bash
# Instalar CLI
pip install vastai
vastai set api-key <SUA_API_KEY>

# Buscar oferta com A100 ou RTX 4090
vastai search offers \
  'gpu_ram>=20 num_gpus=1 inet_up>=200 disk_space>=50 reliability>=0.95' \
  -o 'dph'

# Criar instância (substitua <OFFER_ID>)
vastai create instance <OFFER_ID> \
  --image vllm/vllm-openai:latest \
  --disk 50 \
  --ssh --direct
```

Após a instância iniciar:

```bash
# SSH na instância
ssh -p <PORT> root@<SSH_ADDR>

# Iniciar vLLM
nohup python3 -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-8B-AWQ \
  --quantization awq \
  --max-model-len 8192 \
  --gpu-memory-utilization 0.90 \
  --port 8000 \
  --host 0.0.0.0 \
  --trust-remote-code \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  > /tmp/vllm.log 2>&1 &

# Verificar que está rodando
curl http://localhost:8000/v1/models
```

Na máquina local, criar o SSH tunnel:

```bash
ssh -N -L 8100:localhost:8000 -p <PORT> root@<SSH_ADDR>
```

### Passo 2 — Provisionar GPU para STT + TTS

Pode usar o script automatizado ou provisionar manualmente.

#### Opção A: Script automatizado

```bash
cd src/api/scripts
./deploy_vastai.sh
```

O script:
1. Busca 2 ofertas GPU (1 para STT, 1 para TTS)
2. Provisiona as instâncias com as env vars corretas
3. Aguarda ambas ficarem running (timeout: 10 min)
4. Imprime os endpoints gRPC para configurar no `.env`

```bash
# Ver status das instâncias
./deploy_vastai.sh --status

# Destruir instâncias
./deploy_vastai.sh --destroy
```

#### Opção B: Docker Compose (GPU única)

Se preferir rodar STT + TTS na mesma GPU:

```bash
cd src
docker compose -f docker-compose.gpu.yml up -d
```

Isso inicia:
- STT (Whisper large-v3-turbo) em `localhost:50060`
- TTS (Kokoro-ONNX pf_dora) em `localhost:50070`

#### Opção C: Container combo (GPU única)

```bash
cd src
docker build -f Dockerfile.combo -t macaw-combo .
docker run -d --gpus all \
  -p 50060:50060 \
  -p 50070:50070 \
  --name macaw-stt-tts \
  macaw-combo
```

### Passo 3 — API Server (local)

```bash
cd src/api
pip install -e ".[dev,vad]"
cp .env.example .env
# Editar .env com os endpoints corretos (ver seção "Configuração Atual")
python main.py
```

### Passo 4 — Frontend

```bash
cd src/web
npm install
npm run dev
# Abrir http://localhost:5173
```

### Passo 5 — Validação

```bash
# Health check
curl http://localhost:8765/health

# Verificar logs do API server
tail -f /tmp/macaw-api.log

# Verificar vLLM
curl http://localhost:8100/v1/models

# Testar STT via gRPC (requer grpcurl)
grpcurl -plaintext <STT_HOST>:<STT_PORT> grpc.health.v1.Health/Check

# Testar TTS via gRPC
grpcurl -plaintext <TTS_HOST>:<TTS_PORT> grpc.health.v1.Health/Check
```

## Configurações Alternativas de Providers

### Setup com Providers Qwen3 (maior latência, self-hosted completo)

Para STT e TTS baseados em Qwen3 ao invés de Whisper/Kokoro:

**STT — Qwen3-ASR-1.7B:**
```bash
STT_PROVIDER=qwen-streaming
QWEN_STT_MODEL=Qwen/Qwen3-ASR-1.7B
QWEN_STT_GPU_MEM_UTIL=0.80
QWEN_STT_ENFORCE_EAGER=true
```

**TTS — FasterQwen3TTS:**
```bash
TTS_PROVIDER=faster
QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice
QWEN_TTS_SPEAKER=aiden
FASTER_TTS_CHUNK_SIZE=4
```

Docker images: `src/stt/Dockerfile.gpu` e `src/tts/Dockerfile.gpu` (base `vllm/vllm-openai:v0.14.0`).

### Setup com LLM Cloud (sem GPU para LLM)

```bash
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-20250514
ANTHROPIC_API_KEY=sk-ant-xxx
# Remover VLLM_BASE_URL
```

ou

```bash
LLM_PROVIDER=openai
LLM_MODEL=gpt-4o
OPENAI_API_KEY=sk-xxx
```

## VRAM Estimado por Modelo

| Modelo | VRAM | Uso |
|--------|------|-----|
| Qwen3-8B-AWQ (4-bit) | ~5GB | LLM (vLLM) |
| Whisper large-v3-turbo | ~1.7GB | STT |
| Kokoro-ONNX v1.0 | ~0.5GB | TTS |
| Qwen3-ASR-1.7B | ~4GB | STT (alternativo) |
| Qwen3-TTS-1.7B | ~4GB | TTS (alternativo) |

**GPU mínima recomendada:**
- LLM separado: qualquer GPU com 8GB+ VRAM (RTX 3070, A10G, A100)
- STT + TTS juntos (Whisper + Kokoro): qualquer GPU com 4GB+ VRAM
- STT + TTS juntos (Qwen3): GPU com 12GB+ VRAM

## Benchmarks Medidos (Vast.ai, 2x RTX 4090)

| Métrica | Valor |
|---------|-------|
| TTS TTFB (streaming) | 143–213ms |
| TTS batch (23 chars) | 636ms |
| TTS batch (62 chars) | 1569ms |
| STT batch (2s áudio) | 87ms |
| TTS RTF (warmup) | 0.56 (19.1ms/step) |

## Custos Estimados (Vast.ai, mensal 24/7)

| Configuração | GPUs | Custo/mês |
|-------------|------|-----------|
| 2x RTX 4090 (STT+TTS separados) | 2 | ~$430 |
| 1x RTX 4090 (STT+TTS juntos) | 1 | ~$220 |
| 1x A100 (LLM vLLM) | 1 | ~$360 |

> Para comparação com AWS, GCP e Azure, veja [`docs/GPU_PROVISIONING.md`](GPU_PROVISIONING.md).

## Troubleshooting

### vLLM não responde
```bash
# Verificar se está rodando
ssh -p <PORT> root@<SSH_ADDR> "ps aux | grep vllm"

# Verificar logs
ssh -p <PORT> root@<SSH_ADDR> "tail -50 /tmp/vllm.log"

# Restart
ssh -p <PORT> root@<SSH_ADDR> "kill \$(pgrep -f vllm); sleep 2; nohup python3 -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-8B-AWQ --quantization awq --max-model-len 8192 --gpu-memory-utilization 0.90 --port 8000 --host 0.0.0.0 --trust-remote-code --enable-auto-tool-choice --tool-call-parser hermes > /tmp/vllm.log 2>&1 &"
```

### SSH tunnel caiu
```bash
# Reconectar
ssh -N -L 8100:localhost:8000 -p <PORT> root@<SSH_ADDR>

# Para tunnel persistente, usar autossh
autossh -M 0 -N -L 8100:localhost:8000 -p <PORT> root@<SSH_ADDR> \
  -o "ServerAliveInterval=30" -o "ServerAliveCountMax=3"
```

### STT/TTS gRPC não conecta
```bash
# Verificar conectividade
grpcurl -plaintext <IP>:<PORT> grpc.health.v1.Health/Check

# Se timeout: verificar se a instância Vast.ai ainda está rodando
vastai show instances

# Verificar logs do container
vastai logs <INSTANCE_ID>
```

### API server não emite métricas
```bash
# Verificar se a sessão WebSocket está usando código atualizado
# (sempre reiniciar o server e reconectar o browser após mudanças)
kill $(pgrep -f "python3 main.py")
cd src/api && nohup python3 main.py > /tmp/macaw-api.log 2>&1 &
# Recarregar o frontend no browser
```
