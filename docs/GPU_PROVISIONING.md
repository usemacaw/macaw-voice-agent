# Provisionamento GPU — Theo AI Voice Agent

Guia de deploy do sistema Voice2Voice (STT + TTS + AI Agent) em diferentes provedores de GPU.

## Requisitos do Sistema

| Componente | Modelo | VRAM | GPU |
|---|---|---|---|
| STT | Qwen3-ASR-1.7B (vLLM) | ~4GB + KV cache | CUDA, Ampere+ |
| TTS | Qwen3-TTS-1.7B (faster-qwen3-tts) | ~4GB + CUDA graphs | CUDA, Ampere+ |
| AI Agent | CPU only (gRPC, LLM via API) | — | — |

**Imagem Docker:** `paulohenriquevn/theo-split-gpu:latest` (~15GB)

**Arquiteturas suportadas:**
- **2 GPUs (split):** STT na GPU 0, TTS na GPU 1 — menor contenção, melhor latência
- **1 GPU (monolítico):** STT + TTS compartilhando a mesma GPU — menor custo

---

## Opção 1: Vast.ai

Marketplace de GPUs sob demanda. Melhor custo-benefício para desenvolvimento e testes.

### Vantagens
- GPUs de consumo (RTX 4090) a preço baixo
- Sem compromisso mínimo, cobrança por hora
- Deploy via CLI ou web

### Instâncias recomendadas

| Config | GPU | VRAM | Custo/hr | Custo/mês (24/7) |
|---|---|---|---|---|
| 2x RTX 4090 | 2 GPUs | 24GB x2 | $0.50–0.70 | ~$360–500 |
| 1x RTX 4090 | 1 GPU | 24GB | $0.25–0.40 | ~$180–290 |
| 2x A100 40GB | 2 GPUs | 40GB x2 | $0.80–1.20 | ~$576–864 |

> Preços variam conforme disponibilidade e região. Consulte `vastai search offers`.

### Deploy (2 GPUs)

```bash
# 1. Instalar CLI
pip install vastai
vastai set api-key <SUA_API_KEY>

# 2. Buscar melhor oferta (2x RTX 4090, >95% reliability, >500Mbps download)
vastai search offers \
  'gpu_name=RTX_4090 num_gpus=2 reliability>0.95 inet_down>500 disk_space>=80' \
  -o 'dph'

# 3. Criar instância (substitua <OFFER_ID> pelo ID da oferta escolhida)
vastai create instance <OFFER_ID> \
  --image paulohenriquevn/theo-split-gpu:latest \
  --disk 80 \
  --onstart-cmd "bash /app/start.sh" \
  --ssh --direct \
  --env "ANTHROPIC_API_KEY=<SUA_KEY>"

# 4. Verificar status
vastai show instances

# 5. Acessar via SSH (use o SSH addr e port do output anterior)
ssh -p <PORT> root@<SSH_ADDR>

# 6. Testar
ssh -p <PORT> root@<SSH_ADDR> \
  "cd /app && PYTHONPATH=/app:/app/shared python3 test-e2e.py"

# 7. Destruir quando terminar
vastai destroy instance <INSTANCE_ID>
```

### Variáveis de ambiente (opcionais)

Passadas via `--env` no `vastai create instance`:

```bash
--env "ANTHROPIC_API_KEY=sk-..."        # Obrigatório para LLM
--env "TTS_PROVIDER=faster"             # Default: faster (faster-qwen3-tts)
--env "STT_PROVIDER=qwen-streaming"     # Default: qwen-streaming
--env "FASTER_TTS_CHUNK_SIZE=4"         # Tokens por chunk TTS (menor = menor TTFB)
--env "PIPELINE_TTS_PREFETCH_SIZE=4"    # Sentenças pre-sintetizadas
--env "PIPELINE_MAX_SENTENCE_CHARS=50"  # Max chars por sentença para TTS
--env "LOG_LEVEL=INFO"                  # DEBUG para troubleshooting
```

---

## Opção 2: AWS

Provedores de GPU via EC2. Melhor para produção com SLAs e integração com ecossistema AWS.

### Instâncias recomendadas

#### 2 GPUs (arquitetura split, sem alterações)

| Instância | GPUs | VRAM | vCPUs | RAM | On-Demand/hr | Spot/hr | Nota |
|---|---|---|---|---|---|---|---|
| **g5.12xlarge** | 4x A10G | 24GB x4 | 48 | 192GB | $5.67 | ~$1.70 | Menor multi-GPU g5. Usa 2 das 4 GPUs |
| **g5.24xlarge** | 4x A10G | 24GB x4 | 96 | 384GB | $8.14 | ~$2.44 | Mais CPU/RAM que g5.12xlarge |
| **g5.48xlarge** | 8x A10G | 24GB x8 | 192 | 768GB | $16.29 | ~$4.89 | Overkill, só se precisar de muita CPU |
| **p4d.24xlarge** | 8x A100 | 40GB x8 | 96 | 1152GB | $32.77 | ~$9.83 | Overkill. Só para cargas muito pesadas |

> **Nota:** Não existe instância g5 com exatamente 2 GPUs. A g5.12xlarge (4 GPUs) é a menor opção multi-GPU.

#### 1 GPU (custo otimizado, requer modo monolítico)

| Instância | GPU | VRAM | vCPUs | RAM | On-Demand/hr | Spot/hr | Nota |
|---|---|---|---|---|---|---|---|
| **g5.xlarge** | 1x A10G | 24GB | 4 | 16GB | $1.006 | ~$0.35 | Melhor custo. STT+TTS na mesma GPU |
| **g5.2xlarge** | 1x A10G | 24GB | 8 | 32GB | $1.212 | ~$0.42 | Mais CPU/RAM para o AI Agent |
| **g5.4xlarge** | 1x A10G | 24GB | 16 | 64GB | $1.624 | ~$0.56 | Se precisar de mais CPU |
| **g6e.xlarge** | 1x L40S | 48GB | 4 | 16GB | $1.86 | ~$0.65 | 48GB VRAM, folga total |

### Deploy (2 GPUs — g5.12xlarge)

```bash
# 1. Criar instância EC2 (via AWS CLI)
aws ec2 run-instances \
  --image-id ami-0a5c3558529277641 \
  --instance-type g5.12xlarge \
  --key-name <SUA_KEY_PAIR> \
  --security-group-ids <SG_ID> \
  --subnet-id <SUBNET_ID> \
  --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
  --tag-specifications 'ResourceType=instance,Tags=[{Key=Name,Value=theo-voice-agent}]'

# Nota: Use uma AMI com NVIDIA drivers pré-instalados:
#   - Amazon Linux 2 com GPU: ami-0a5c3558529277641 (us-east-1)
#   - Deep Learning AMI (Ubuntu): procure "Deep Learning AMI GPU PyTorch" no marketplace
#   - Ou instale drivers manualmente: https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/install-nvidia-driver.html

# 2. Conectar via SSH
ssh -i <key.pem> ec2-user@<PUBLIC_IP>

# 3. Instalar Docker + NVIDIA Container Toolkit (se não veio na AMI)
sudo yum install -y docker
sudo systemctl start docker
# Veja: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html

# 4. Executar container
sudo docker run -d --gpus all \
  -p 50051:50051 \
  -e ANTHROPIC_API_KEY=<SUA_KEY> \
  --name theo-voice-agent \
  --restart unless-stopped \
  paulohenriquevn/theo-split-gpu:latest

# 5. Verificar logs
sudo docker logs -f theo-voice-agent

# 6. Testar
sudo docker exec theo-voice-agent \
  bash -c "cd /app && PYTHONPATH=/app:/app/shared python3 test-e2e.py"
```

### Deploy com Spot Instance (economia de ~70%)

```bash
aws ec2 request-spot-instances \
  --spot-price "2.00" \
  --instance-count 1 \
  --type "one-time" \
  --launch-specification '{
    "ImageId": "ami-0a5c3558529277641",
    "InstanceType": "g5.12xlarge",
    "KeyName": "<SUA_KEY_PAIR>",
    "SecurityGroupIds": ["<SG_ID>"],
    "BlockDeviceMappings": [
      {"DeviceName": "/dev/sda1", "Ebs": {"VolumeSize": 100, "VolumeType": "gp3"}}
    ]
  }'
```

> **Atenção:** Spot instances podem ser interrompidas pela AWS com 2 minutos de aviso. Use para testes e desenvolvimento. Para produção, prefira on-demand ou reserved instances.

### Deploy com ECS/Fargate (produção)

Para produção com auto-scaling e alta disponibilidade, considere ECS com GPU:

```bash
# Task Definition (resumo)
{
  "family": "theo-voice-agent",
  "requiresCompatibilities": ["EC2"],
  "containerDefinitions": [{
    "name": "theo-voice-agent",
    "image": "paulohenriquevn/theo-split-gpu:latest",
    "resourceRequirements": [
      {"type": "GPU", "value": "2"}
    ],
    "portMappings": [
      {"containerPort": 50051, "protocol": "tcp"}
    ],
    "environment": [
      {"name": "ANTHROPIC_API_KEY", "value": "<SUA_KEY>"}
    ]
  }]
}
```

---

## Opção 3: GCP (Google Cloud)

### Instâncias recomendadas

| Instância | GPU | VRAM | On-Demand/hr | Nota |
|---|---|---|---|---|
| **g2-standard-8** | 1x L4 | 24GB | ~$0.84 | Custo otimizado, 1 GPU |
| **g2-standard-24** | 2x L4 | 24GB x2 | ~$1.68 | 2 GPUs, arquitetura split |
| **a2-highgpu-2g** | 2x A100 | 40GB x2 | ~$7.35 | Alta performance |

### Deploy rápido

```bash
# Criar VM com 2x L4
gcloud compute instances create theo-voice-agent \
  --zone=us-central1-a \
  --machine-type=g2-standard-24 \
  --accelerator=count=2,type=nvidia-l4 \
  --boot-disk-size=100GB \
  --image-family=common-gpu \
  --image-project=deeplearning-platform-release \
  --maintenance-policy=TERMINATE

# SSH e executar container
gcloud compute ssh theo-voice-agent
sudo docker run -d --gpus all \
  -p 50051:50051 \
  -e ANTHROPIC_API_KEY=<SUA_KEY> \
  paulohenriquevn/theo-split-gpu:latest
```

---

## Opção 4: Azure

### Instâncias recomendadas

| Instância | GPU | VRAM | On-Demand/hr | Nota |
|---|---|---|---|---|
| **NC4as_T4_v3** | 1x T4 | 16GB | ~$0.53 | Barata, mas 16GB é apertado |
| **NC24ads_A100_v4** | 1x A100 | 80GB | ~$3.67 | VRAM de sobra |
| **NC48ads_A100_v4** | 2x A100 | 80GB x2 | ~$7.35 | 2 GPUs, arquitetura split |

### Deploy rápido

```bash
# Criar VM com GPU
az vm create \
  --resource-group theo-rg \
  --name theo-voice-agent \
  --size Standard_NC24ads_A100_v4 \
  --image Canonical:0001-com-ubuntu-server-jammy:22_04-lts-gen2:latest \
  --admin-username azureuser \
  --generate-ssh-keys

# Instalar NVIDIA drivers + Docker, depois:
sudo docker run -d --gpus all \
  -p 50051:50051 \
  -e ANTHROPIC_API_KEY=<SUA_KEY> \
  paulohenriquevn/theo-split-gpu:latest
```

---

## Comparação de custos (mensal, 24/7)

| Provedor | Config | GPUs | Custo/mês |
|---|---|---|---|
| **Vast.ai** | 2x RTX 4090 | 2 | **~$430** |
| **Vast.ai** | 1x RTX 4090 | 1 | **~$220** |
| **AWS Spot** | g5.xlarge (1x A10G) | 1 | **~$252** |
| **AWS Spot** | g5.12xlarge (4x A10G) | 4 (usa 2) | **~$1,224** |
| **AWS On-Demand** | g5.xlarge | 1 | **~$724** |
| **AWS On-Demand** | g5.12xlarge | 4 (usa 2) | **~$4,082** |
| **AWS Reserved 1yr** | g5.xlarge | 1 | **~$460** |
| **GCP** | g2-standard-24 (2x L4) | 2 | **~$1,210** |
| **GCP Spot** | g2-standard-24 (2x L4) | 2 | **~$400** |
| **Azure** | NC24ads_A100_v4 | 1 | **~$2,642** |

---

## Benchmarks medidos (2x RTX 4090, Vast.ai)

| Métrica | Valor |
|---|---|
| TTS TTFB (streaming) | **143–213ms** |
| TTS batch (23 chars) | 636ms |
| TTS batch (62 chars) | 1569ms |
| STT batch (2s áudio) | 87ms |
| TTS RTF (warmup) | 0.56 (19.1ms/step) |
| E2E pipeline | 8/8 testes passando |

---

## Checklist de validação pós-deploy

```bash
# 1. Verificar que os 3 serviços estão rodando
docker exec <container> ps aux | grep python

# 2. Verificar TTS provider correto
docker exec <container> bash -c \
  'cat /proc/$(pgrep -f tts-server | head -1)/environ | tr "\0" "\n" | grep TTS_PROVIDER'
# Esperado: TTS_PROVIDER=faster

# 3. Verificar CUDA graphs capturados nos logs
docker logs <container> 2>&1 | grep "CUDA graphs captured"
# Esperado: "CUDA graphs captured and ready"

# 4. Rodar teste E2E
docker exec <container> bash -c \
  "cd /app && PYTHONPATH=/app:/app/shared python3 test-e2e.py"
# Esperado: 8 passed, 0 failed

# 5. Benchmark TTS TTFB (opcional)
docker exec <container> python3 -c "
import grpc, time, sys
sys.path.insert(0, '/app/shared')
from grpc_gen import tts_service_pb2, tts_service_pb2_grpc
stub = tts_service_pb2_grpc.TTSServiceStub(grpc.insecure_channel('localhost:50070'))
t0 = time.perf_counter()
first = None
for c in stub.SynthesizeStream(tts_service_pb2.SynthesizeRequest(text='Olá, tudo bem?')):
    if not first: first = time.perf_counter()
print(f'TTS TTFB: {(first-t0)*1000:.0f}ms')
"
# Esperado: < 250ms
```
