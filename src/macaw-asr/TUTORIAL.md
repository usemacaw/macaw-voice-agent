# macaw-asr — Tutorial

Guia completo para o time. Cobre instalacao, todos os modelos, API, multi-GPU e troubleshooting.

## Indice

1. [Conceito](#conceito)
2. [Instalacao](#instalacao)
3. [CLI](#cli)
4. [Modelos](#modelos)
   - [Faster-Whisper (recomendado)](#faster-whisper-recomendado)
   - [Whisper (HuggingFace)](#whisper-huggingface)
   - [Qwen3-ASR](#qwen3-asr)
   - [Parakeet (NVIDIA NeMo)](#parakeet-nvidia-nemo)
   - [FastConformer PT-BR](#fastconformer-pt-br)
   - [Canary 1B v2](#canary-1b-v2)
5. [Multi-GPU](#multi-gpu)
6. [API Server](#api-server)
7. [Usando com OpenAI SDK](#usando-com-openai-sdk)
8. [Formatos de resposta](#formatos-de-resposta)
9. [Compatibilidade de versoes](#compatibilidade-de-versoes)
10. [Troubleshooting](#troubleshooting)

---

## Conceito

macaw-asr e um servidor ASR (speech-to-text) que:

- Suporta **multiplos modelos** com uma unica interface
- Expoe a **mesma API do OpenAI** (`/v1/audio/transcriptions`)
- Funciona como o **Ollama, mas para ASR**: `pull`, `serve`, `list`
- Cada modelo instala **apenas suas proprias dependencias**
- Suporta **multi-GPU** com replicacao e round-robin

```
pip install macaw-asr               # Core leve (~5MB)
pip install macaw-asr[faster-whisper]  # + CTranslate2 (~200MB)
macaw-asr pull faster-whisper-small
macaw-asr serve
```

---

## Instalacao

### Requisitos

- Python >= 3.10
- GPU NVIDIA com CUDA (para modelos reais)
- ffmpeg (opcional, para decodificar mp3/webm/ogg)

### Core (sem nenhum modelo)

```bash
pip install macaw-asr
```

Isso instala apenas: numpy, pydantic, fastapi, uvicorn. Nenhum modelo funciona ainda — voce escolhe qual instalar.

### Instalar por familia de modelo

```bash
pip install macaw-asr[faster-whisper]    # CTranslate2 (~200MB, sem PyTorch)
pip install macaw-asr[whisper]           # PyTorch + transformers (~5GB)
pip install macaw-asr[qwen]             # PyTorch + transformers + qwen-asr (~5GB)
pip install macaw-asr[parakeet]         # PyTorch + NeMo (~5GB)
pip install macaw-asr[all]              # Tudo
```

### Verificar modelos disponiveis

```bash
macaw-asr list --all
```

Saida:

```
NAME                      FAMILY           SIZE     DEPS
----------------------------------------------------------------------
qwen                      qwen             0.6B     missing (pip install "macaw-asr[qwen]")
whisper-tiny              whisper          39M      missing (pip install "macaw-asr[whisper]")
faster-whisper-small      faster-whisper   244M     ok
parakeet                  parakeet         0.6B     missing (pip install "macaw-asr[parakeet]")
fastconformer-pt          parakeet         115M     missing (pip install "macaw-asr[parakeet]")
canary                    parakeet         1.0B     missing (pip install "macaw-asr[parakeet]")
...
```

A coluna DEPS mostra `ok` ou o comando exato para instalar.

---

## CLI

```bash
macaw-asr pull <modelo>        # Baixa pesos do modelo
macaw-asr serve                # Inicia servidor HTTP na porta 8766
macaw-asr serve --port 9000   # Porta customizada
macaw-asr serve --devices 2   # Multi-GPU (2 replicas)
macaw-asr transcribe <arquivo> # Transcreve direto (sem servidor)
macaw-asr list                 # Lista modelos baixados localmente
macaw-asr list --all           # Lista TODOS os modelos disponiveis
macaw-asr remove <modelo>      # Remove modelo local
```

O `pull` aceita nomes curtos:

```bash
macaw-asr pull faster-whisper-small
# Resolved: faster-whisper-small -> openai/whisper-small (family: faster-whisper)
# Downloading openai/whisper-small...
# Done.
```

Se faltar dependencia, ele avisa antes de baixar:

```bash
macaw-asr pull canary
# Resolved: canary -> nvidia/canary-1b-v2 (family: parakeet)
# Missing dependencies: torch, nemo
# Install with: pip install "macaw-asr[parakeet]"
```

---

## Modelos

### Faster-Whisper (recomendado)

**O que e:** Whisper rodando no CTranslate2 (runtime C++ otimizado). 4x mais rapido que o Whisper original, sem depender de PyTorch.

**Quando usar:** Primeira escolha para producao. Leve, rapido, confiavel.

**Variantes:**

| Modelo | Params | VRAM | Qualidade PT-BR |
|--------|--------|------|-----------------|
| `faster-whisper-tiny` | 39M | ~1GB | Basica |
| `faster-whisper-small` | 244M | ~2GB | Boa |
| `faster-whisper-medium` | 769M | ~5GB | Muito boa |
| `faster-whisper-large` | 1.5B | ~10GB | Excelente |

**Instalacao e uso:**

```bash
# 1. Instalar dependencias (~200MB, SEM PyTorch)
pip install macaw-asr[faster-whisper]

# 2. Baixar modelo
macaw-asr pull faster-whisper-small

# 3. Servir
MACAW_ASR_MODEL=faster-whisper-small macaw-asr serve

# 4. Testar
curl -F file=@audio.wav -F model=whisper-1 http://localhost:8766/v1/audio/transcriptions
```

**Dependencias instaladas:**

```
faster-whisper >= 1.0.0    # Wrapper Python do CTranslate2
ctranslate2                # Runtime C++ (instalado automaticamente)
huggingface_hub            # Para download de modelos
```

**Performance (RTX 4090, audio 1s):**

```
Start (cold): ~2-5s (download CTranslate2 model na primeira vez)
Start (warm): ~500ms
Inferencia:   ~2ms
```

**Detalhes tecnicos:**
- Backend CTranslate2 com quantizacao int8/float16
- Na primeira carga, converte automaticamente de HuggingFace para formato CTranslate2
- Nao precisa de PyTorch em runtime
- Suporta: `compute_type` float16, int8, int8_float16, float32

---

### Whisper (HuggingFace)

**O que e:** Whisper original da OpenAI via HuggingFace transformers + PyTorch.

**Quando usar:** Quando precisar de acesso direto ao modelo PyTorch (pesquisa, fine-tuning, torch.compile).

**Variantes:**

| Modelo | Params | VRAM | Qualidade PT-BR |
|--------|--------|------|-----------------|
| `whisper-tiny` | 39M | ~1GB | Basica |
| `whisper-small` | 244M | ~2GB | Boa |
| `whisper-medium` | 769M | ~5GB | Muito boa |
| `whisper-large` | 1.5B | ~10GB | Excelente |

**Instalacao e uso:**

```bash
# 1. Instalar dependencias (~5GB com PyTorch)
pip install macaw-asr[whisper]

# 2. Baixar modelo
macaw-asr pull whisper-small

# 3. Servir
MACAW_ASR_MODEL=whisper-small macaw-asr serve

# 4. Testar
curl -F file=@audio.wav -F model=whisper-1 http://localhost:8766/v1/audio/transcriptions
```

**Dependencias instaladas:**

```
torch >= 2.0.0             # PyTorch (GPU)
transformers >= 4.40.0     # HuggingFace (WhisperForConditionalGeneration)
huggingface_hub            # Para download de modelos
```

**Performance (RTX 4090, audio 1s):**

```
Start (cold): ~4-7s
Inferencia:   ~22ms
```

**Detalhes tecnicos:**
- Usa `model.generate()` do HuggingFace (encoder-decoder)
- Nao tem streaming token-by-token (batch only)
- Suporta `torch.compile` no encoder para acelerar
- O dtype padrao e float16

---

### Qwen3-ASR

**O que e:** Modelo ASR da Alibaba. Autoregressive com streaming token-by-token. SOTA entre modelos open-source de 0.6B.

**Quando usar:** Quando precisar de streaming real (token-by-token via SSE) ou melhor qualidade em PT-BR com modelo pequeno.

**Variantes:**

| Modelo | Params | VRAM | Qualidade PT-BR |
|--------|--------|------|-----------------|
| `qwen` | 0.6B | ~2GB | Muito boa |

**Instalacao e uso:**

```bash
# 1. Instalar dependencias (~5GB com PyTorch)
pip install macaw-asr[qwen]

# 2. Baixar modelo
macaw-asr pull qwen

# 3. Servir
MACAW_ASR_MODEL=qwen macaw-asr serve

# 4. Testar (batch)
curl -F file=@audio.wav -F model=whisper-1 http://localhost:8766/v1/audio/transcriptions

# 5. Testar (streaming SSE)
curl -F file=@audio.wav -F model=whisper-1 -F stream=true http://localhost:8766/v1/audio/transcriptions
# data: {"type":"transcript.text.delta","delta":"Ola"}
# data: {"type":"transcript.text.delta","delta":", como"}
# data: {"type":"transcript.text.done","text":"Ola, como voce esta?","usage":{...}}
```

**Dependencias instaladas:**

```
torch >= 2.0.0             # PyTorch (GPU)
transformers >= 4.40.0     # HuggingFace (AutoModel)
qwen-asr                   # Qwen3ASRConfig, Qwen3ASRProcessor
huggingface_hub            # Para download de modelos
```

**Performance (RTX 4090, audio 1s):**

```
Start (cold): ~6-23s (download modelo + warmup)
Inferencia:   ~80ms
```

**Detalhes tecnicos:**
- Unico modelo com streaming token-by-token real (SSE)
- Usa GPU mel spectrogram (~2ms) com fallback CPU (~15ms)
- Decode manual com KV cache (nao usa HF generate)
- Suporta `fast_finish_inputs` para otimizar streaming sessions
- Suporta `torch.compile` e CUDA graphs

**Nota sobre GPU mel:**
Em versoes torch >= 2.11, o GPU mel (`_torch_extract_fbank_features`) pode falhar com erro de STFT kernel. O sistema faz fallback automatico para CPU — funciona perfeitamente, apenas ~13ms mais lento por request. Com torch 2.4.x nao ha problema.

---

### Parakeet (NVIDIA NeMo)

**O que e:** Modelo ASR da NVIDIA baseado em FastConformer + TDT decoder. Top 1 no Open ASR Leaderboard. Melhor qualidade PT-BR entre todos os modelos.

**Quando usar:** Quando qualidade maxima em PT-BR e prioridade e latencia nao e critica.

**Variantes:**

| Modelo | Params | VRAM | Qualidade PT-BR |
|--------|--------|------|-----------------|
| `parakeet` / `parakeet-tdt` | 0.6B | ~3GB | Excelente (melhor) |
| `parakeet-ctc` | 1.1B | ~5GB | Excelente |

**Instalacao e uso:**

```bash
# 1. Instalar dependencias (~5GB com PyTorch + NeMo)
pip install macaw-asr[parakeet]

# 2. Baixar modelo
macaw-asr pull parakeet

# 3. Servir
MACAW_ASR_MODEL=parakeet macaw-asr serve

# 4. Testar
curl -F file=@audio.wav -F model=whisper-1 http://localhost:8766/v1/audio/transcriptions
```

**Dependencias instaladas:**

```
torch >= 2.0.0             # PyTorch (GPU)
nemo_toolkit[asr] >= 2.0.0 # NVIDIA NeMo framework
huggingface_hub            # Para download de modelos
```

**Performance (RTX 4090, audio 1s):**

```
Start (cold): ~39s (NeMo e pesado para carregar)
Inferencia:   ~30ms
```

**Detalhes tecnicos:**
- NeMo usa `.nemo` checkpoint format (nao safetensors)
- Batch only (sem streaming token-by-token)
- NeMo faz preprocessing interno (mel extraction)
- Nao suporta torch.compile
- Startup lento (~39s) por causa do framework NeMo

**Compatibilidade critica:**
- NeMo 2.1.x: compativel com torch 2.4.x
- NeMo 2.7.x: requer torch >= 2.6.0
- Em ambientes com torch 2.4.x, usar: `pip install "nemo_toolkit[asr]>=2.0.0,<2.2.0"`

---

### FastConformer PT-BR

**O que e:** Modelo NVIDIA especializado em portugues brasileiro. FastConformer Hybrid (Transducer+CTC) com pontuacao e capitalizacao automatica. 115M params — leve e rapido.

**Quando usar:** Quando precisa de ASR **dedicado para PT-BR** com pontuacao, e quer um modelo leve.

| Modelo | Params | VRAM | Qualidade PT-BR |
|--------|--------|------|-----------------|
| `fastconformer-pt` | 115M | ~1GB | Excelente (WER 12% MCV) |

**Instalacao e uso:**

```bash
# 1. Instalar dependencias (mesma do Parakeet)
pip install macaw-asr[parakeet]

# 2. Baixar modelo
macaw-asr pull fastconformer-pt

# 3. Servir
MACAW_ASR_MODEL=fastconformer-pt macaw-asr serve

# 4. Testar
curl -F file=@audio.wav -F model=whisper-1 http://localhost:8766/v1/audio/transcriptions
```

**Performance (RTX 4090, audio 1s):**

```
Start (cold): ~19s (download + NeMo init)
Inferencia:   ~30ms
```

**Detalhes tecnicos:**
- HuggingFace ID: `nvidia/stt_pt_fastconformer_hybrid_large_pc`
- Arquitetura: FastConformer encoder + Hybrid Transducer/CTC decoder
- Pontuacao automatica: ponto, virgula, interrogacao
- Capitalizacao automatica
- Tokenizer: SentencePiece (128 tokens)
- Licenca: CC-BY-NC-4.0 (uso nao-comercial)

---

### Canary 1B v2

**O que e:** Modelo multilingual da NVIDIA com 978M params. Suporta 25 idiomas + traducao entre idiomas. O mais capaz do lineup.

**Quando usar:** Quando precisa de ASR **multilingual**, traducao de fala, ou qualidade maxima com pontuacao.

| Modelo | Params | VRAM | Idiomas |
|--------|--------|------|---------|
| `canary` | 1.0B | ~6GB | 25 (PT, EN, ES, FR, DE, ...) |

**Instalacao e uso:**

```bash
# 1. Instalar dependencias (mesma do Parakeet)
pip install macaw-asr[parakeet]

# 2. Baixar modelo
macaw-asr pull canary

# 3. Servir
MACAW_ASR_MODEL=canary macaw-asr serve

# 4. Testar
curl -F file=@audio.wav -F model=whisper-1 http://localhost:8766/v1/audio/transcriptions
```

**Performance (RTX 4090, audio 1s):**

```
Start (cold): ~74s (modelo grande, primeiro download)
Start (warm): ~19s
Inferencia:   ~460ms
```

**Detalhes tecnicos:**
- HuggingFace ID: `nvidia/canary-1b-v2`
- Arquitetura: FastConformer encoder (32 layers) + Transformer decoder (8 layers)
- 25 idiomas europeus (incluindo PT — portugues europeu)
- Traducao de fala: qualquer idioma suportado para/de ingles
- Pontuacao e capitalizacao automatica
- Suporta timestamps por palavra e por segmento
- Licenca: CC-BY-4.0 (uso comercial permitido)

**Nota:** O Canary usa tokens especiais internos (`<|en|>`, `<|pnc|>`, etc.) que sao automaticamente limpos pelo macaw-asr antes de retornar o texto.

---

## Multi-GPU

O macaw-asr suporta replicacao de modelo em multiplas GPUs com distribuicao round-robin automatica. Cada GPU roda uma copia independente do modelo, e as requests sao distribuidas entre elas.

### Uso

```bash
# 2 GPUs
macaw-asr serve --devices 2

# 4 GPUs
macaw-asr serve --devices 4

# Via variavel de ambiente
MACAW_ASR_DEVICES=2 macaw-asr serve

# Lista explicita (power users)
MACAW_ASR_DEVICES=cuda:0,cuda:2 macaw-asr serve
```

### Como funciona

```
Request 1 → cuda:0
Request 2 → cuda:1
Request 3 → cuda:0
Request 4 → cuda:1
...
```

O scheduler cria N engines (uma por GPU), e distribui requests via round-robin. Nao muda nada na API — as requests sao distribuidas automaticamente.

### Startup com multi-GPU

```
macaw-asr server starting on http://0.0.0.0:8766
  Model: faster-whisper-small
  GPUs:  cuda:0, cuda:1 (2 replicas)
  Docs:  http://0.0.0.0:8766/docs
```

### Quando usar

- **1 GPU:** Suficiente para a maioria dos casos. Modelos ASR sao pequenos (< 2GB VRAM).
- **2+ GPUs:** Quando throughput e importante (muitas requests simultaneas). Cada GPU processa requests independentemente.

### Nota

Multi-GPU e sobre **throughput** (mais requests por segundo), nao sobre velocidade de uma unica request. Um request individual nao fica mais rapido com mais GPUs.

---

## API Server

### Iniciar

```bash
# Modelo padrao (definido por MACAW_ASR_MODEL)
macaw-asr serve

# Modelo especifico
MACAW_ASR_MODEL=faster-whisper-small macaw-asr serve

# Porta, host e multi-GPU
macaw-asr serve --host 0.0.0.0 --port 9000 --devices 2
```

### Endpoints

| Metodo | Rota | Descricao |
|--------|------|-----------|
| `POST` | `/v1/audio/transcriptions` | Transcreve audio (OpenAI-compatible) |
| `POST` | `/v1/audio/translations` | Traduz audio para ingles |
| `GET` | `/v1/models` | Lista modelos disponiveis |
| `POST` | `/api/show` | Detalhes de um modelo |
| `GET` | `/api/ps` | Modelos carregados em memoria |
| `POST` | `/api/pull` | Baixa um modelo |
| `DELETE` | `/api/delete` | Remove um modelo |
| `GET` | `/api/version` | Versao do servidor |
| `GET` | `/` | Health check |

### Transcrever audio

```bash
# JSON (padrao)
curl -F file=@audio.wav -F model=whisper-1 \
  http://localhost:8766/v1/audio/transcriptions
# {"text": "Ola, como voce esta?", "usage": {"type": "duration", "seconds": 3}}

# Texto puro
curl -F file=@audio.wav -F model=whisper-1 -F response_format=text \
  http://localhost:8766/v1/audio/transcriptions
# Ola, como voce esta?

# Streaming SSE (apenas Qwen)
curl -F file=@audio.wav -F model=whisper-1 -F stream=true \
  http://localhost:8766/v1/audio/transcriptions
```

### Variaveis de ambiente

| Variavel | Padrao | Descricao |
|----------|--------|-----------|
| `MACAW_ASR_MODEL` | `qwen` | Nome do modelo no registry |
| `MACAW_ASR_MODEL_ID` | *(auto)* | ID HuggingFace (resolvido do registry se omitido) |
| `MACAW_ASR_DEVICE` | `cuda:0` | Device de inferencia |
| `MACAW_ASR_DEVICES` | *(vazio)* | Multi-GPU: numero de GPUs ou lista (ex: `2` ou `cuda:0,cuda:1`) |
| `MACAW_ASR_DTYPE` | *(auto)* | Dtype do modelo (resolvido do registry se omitido) |
| `MACAW_ASR_LANGUAGE` | `pt` | Idioma padrao |
| `MACAW_ASR_HOME` | `~/.macaw-asr` | Diretorio de modelos locais |

**Nota:** `MACAW_ASR_MODEL_ID` e `MACAW_ASR_DTYPE` sao resolvidos automaticamente a partir do registry quando voce define apenas `MACAW_ASR_MODEL`. Por exemplo, `MACAW_ASR_MODEL=faster-whisper-small` automaticamente resolve para `model_id=openai/whisper-small` e `dtype=float16`.

---

## Usando com OpenAI SDK

O macaw-asr e **drop-in replacement** da API da OpenAI. Qualquer codigo que use a OpenAI SDK funciona sem modificacao:

```python
from openai import OpenAI

# Apontar para o macaw-asr local
client = OpenAI(base_url="http://localhost:8766/v1", api_key="unused")

# Transcrever
result = client.audio.transcriptions.create(
    model="whisper-1",      # aceita qualquer alias OpenAI
    file=open("audio.wav", "rb"),
    language="pt",
)
print(result.text)

# Verbose JSON (com timestamps)
result = client.audio.transcriptions.create(
    model="whisper-1",
    file=open("audio.wav", "rb"),
    response_format="verbose_json",
)
print(f"Duracao: {result.duration}s")
print(f"Idioma: {result.language}")

# Listar modelos
for model in client.models.list().data:
    print(model.id)
```

Os aliases `whisper-1`, `gpt-4o-transcribe`, `gpt-4o-mini-transcribe` sao mapeados automaticamente para o modelo configurado no servidor.

---

## Formatos de resposta

| Formato | Flag | Content-Type |
|---------|------|--------------|
| JSON | `response_format=json` (padrao) | application/json |
| Texto | `response_format=text` | text/plain |
| Verbose JSON | `response_format=verbose_json` | application/json |
| SRT | `response_format=srt` | text/plain |
| VTT | `response_format=vtt` | text/vtt |
| SSE Streaming | `stream=true` | text/event-stream |

### Exemplos

```bash
# SRT (legendas)
curl -F file=@audio.wav -F model=whisper-1 -F response_format=srt \
  http://localhost:8766/v1/audio/transcriptions
# 1
# 00:00:00,000 --> 00:00:03,500
# Ola, como voce esta?

# VTT (legendas web)
curl -F file=@audio.wav -F model=whisper-1 -F response_format=vtt \
  http://localhost:8766/v1/audio/transcriptions
# WEBVTT
#
# 00:00:00.000 --> 00:00:03.500
# Ola, como voce esta?
```

---

## Compatibilidade de versoes

### Matriz de compatibilidade

| Modelo | torch | transformers | Extras |
|--------|-------|--------------|--------|
| **faster-whisper** | Nao precisa | Nao precisa | `faster-whisper >= 1.0` + `ctranslate2` |
| **whisper** | >= 2.0 | >= 4.40 | - |
| **qwen** | >= 2.0 | >= 4.40 | `qwen-asr` |
| **parakeet** | 2.4.x - 2.5.x | - | `nemo_toolkit[asr] >= 2.0, < 2.2` |
| **fastconformer-pt** | 2.4.x - 2.5.x | - | `nemo_toolkit[asr] >= 2.0, < 2.2` |
| **canary** | 2.4.x - 2.5.x | - | `nemo_toolkit[asr] >= 2.0, < 2.2` |

### Combinacoes testadas

| Ambiente | torch | NeMo | Modelos testados | Status |
|----------|-------|------|------------------|--------|
| RTX 4090, torch 2.4.1 | 2.4.1+cu124 | 2.1.0 | faster-whisper, whisper, qwen, parakeet, fastconformer-pt, canary | Todos PASS |
| RTX 4090, torch 2.11.0 | 2.11.0+cu130 | - | faster-whisper, whisper, qwen | 3/3 PASS (NeMo incompativel) |
| RTX 3090, torch 2.4.1 | 2.4.1+cu124 | 2.1.0 | Todos | OK (producao) |

### Recomendacao de ambiente

Para rodar **todos** os modelos sem conflito:

```bash
# Usar torch 2.4.x (mais compativel)
pip install "torch==2.4.1" --index-url https://download.pytorch.org/whl/cu124
pip install macaw-asr[all]
pip install "nemo_toolkit[asr]>=2.0.0,<2.2.0"  # Downgrade NeMo se necessario
```

Para rodar **apenas faster-whisper** (sem PyTorch):

```bash
pip install macaw-asr[faster-whisper]
# Pronto. Sem torch, sem conflitos.
```

---

## Troubleshooting

### "Missing dependencies: torch, transformers"

```
$ macaw-asr pull qwen
Missing dependencies: torch, transformers, qwen_asr
Install with: pip install "macaw-asr[qwen]"
```

Instale o extra correspondente ao modelo.

### "Unable to open file 'model.bin'"

Faster-whisper precisa de modelos no formato CTranslate2. Na primeira carga, ele converte automaticamente. Se esse erro aparecer, o modelo foi baixado no formato HuggingFace raw. Solucao: remova e re-pull:

```bash
macaw-asr remove openai/whisper-small
macaw-asr pull faster-whisper-small
```

### "GPU mel failed, falling back to CPU"

Qwen3-ASR com torch >= 2.11 tem um bug no `_torch_extract_fbank_features`. O sistema faz fallback automatico para CPU. Funciona perfeitamente, apenas ~13ms mais lento por request. Nao e um erro critico.

### "module 'torch.nn' has no attribute 'Buffer'" (NeMo)

NeMo 2.7+ requer torch >= 2.6. Se voce esta com torch 2.4.x:

```bash
pip install "nemo_toolkit[asr]>=2.0.0,<2.2.0"
```

### "INTERNAL ASSERT FAILED" / "CUDA error" (NeMo)

NeMo 2.7+ com torch 2.11 causa CUDA errors. Use torch 2.4.x-2.5.x com NeMo < 2.2:

```bash
pip install "torch==2.4.1" --index-url https://download.pytorch.org/whl/cu124
pip install "nemo_toolkit[asr]>=2.0.0,<2.2.0"
```

### "cuDNN error: CUDNN_STATUS_NOT_INITIALIZED"

Problema de hardware/drivers na maquina GPU. Modelos com RNN (FastConformer Hybrid, Canary) precisam de cuDNN funcional. Teste com:

```python
import torch
rnn = torch.nn.LSTM(10, 20, 1).cuda()
x = torch.randn(5, 3, 10).cuda()
out, _ = rnn(x)
print("cuDNN OK")
```

Se falhar, troque de maquina/instancia. O macaw-asr carrega modelos NeMo na CPU primeiro e depois move pra GPU para minimizar esse problema.

### "torchvision has no attribute 'extension'"

Conflito de versao torchvision/torch. Atualize:

```bash
pip install torchvision torchaudio --upgrade
```

### Servidor nao inicia ("address already in use")

Outra instancia ja esta rodando na mesma porta:

```bash
# Verificar
lsof -i :8766
# Matar
kill $(lsof -t -i :8766)
# Re-iniciar
macaw-asr serve
```

### Modelos locais: onde ficam?

```
~/.macaw-asr/models/
├── openai--whisper-small/       # macaw-asr pull whisper-small
├── openai--whisper-tiny/        # macaw-asr pull whisper-tiny
├── Qwen--Qwen3-ASR-0.6B/       # macaw-asr pull qwen
└── ...
```

Para mudar o diretorio: `export MACAW_ASR_HOME=/outro/caminho`

---

## Resumo: qual modelo escolher?

| Cenario | Modelo recomendado | Motivo |
|---------|--------------------|--------|
| **Producao (latencia baixa)** | `faster-whisper-small` | 2ms inferencia, sem PyTorch, ~200MB install |
| **Producao (qualidade maxima)** | `faster-whisper-large` | Melhor Whisper, sem PyTorch |
| **PT-BR (melhor qualidade)** | `parakeet` | Top 1 Open ASR Leaderboard |
| **PT-BR (leve + pontuacao)** | `fastconformer-pt` | 115M params, pontuacao automatica, WER 12% |
| **Multilingual (25 idiomas)** | `canary` | 978M params, traducao, pontuacao |
| **Streaming real-time** | `qwen` | Unico com SSE token-by-token |
| **Ambiente sem GPU** | `faster-whisper-tiny` | Roda em CPU (lento mas funciona) |
| **Pesquisa / fine-tuning** | `whisper-small` | Acesso direto ao modelo PyTorch |
| **Install mais leve** | `faster-whisper-*` | ~200MB total (sem PyTorch) |
| **Alto throughput** | qualquer + `--devices N` | Multi-GPU round-robin |
