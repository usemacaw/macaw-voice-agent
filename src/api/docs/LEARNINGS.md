# Aprendizados — OpenVoiceAPI Voice Pipeline

> Documento consolidando todos os aprendizados da construção do pipeline de voz self-hosted.
> Data: 2026-03-12

---

## 1. TTS — Text-to-Speech

### Comparativo de Providers Testados

| Provider | Qualidade PT-BR | Latência | Self-hosted | Resultado |
|---|---|---|---|---|
| **Qwen3-TTS** (speaker Ryan) | Ruim — sotaque chinês carregado | ~1.5s (GPU) | Sim | Descartado |
| **Edge TTS** (Microsoft) | Excelente — vozes nativas PT-BR | ~0.8s | Não (cloud API) | Descartado (requisito self-hosted) |
| **Kokoro-ONNX** (pf_dora) | Bom — PT-BR natural sem sotaque | ~2.1s CPU / ~1.0s GPU | Sim | **Escolhido** |

### Kokoro-ONNX — Detalhes

- **Package**: `kokoro-onnx` (não confundir com `kokoro` que é PyTorch)
- **Modelo**: `kokoro-v1.0.onnx` (~311MB) + `voices-v1.0.bin` (~27MB)
- **Voz PT-BR**: `pf_dora` (portuguesa feminina) com lang `pt-br`
- **Output**: 24kHz float32 — precisa resample para 8kHz no pipeline
- **Streaming**: Nativo via `kokoro.create_stream()` (async generator)
- **54 vozes disponíveis** em múltiplos idiomas

### Kokoro GPU — Configuração Crítica

1. **`kokoro-onnx` instala `onnxruntime` CPU por padrão**. Para GPU:
   ```dockerfile
   # Instalar kokoro-onnx primeiro (traz onnxruntime CPU)
   RUN pip install kokoro-onnx
   # Depois forçar reinstalação do GPU
   RUN pip install --no-cache-dir --force-reinstall "onnxruntime-gpu>=1.18.0"
   ```

2. **ONNX provider não é auto-detectado**. O kokoro-onnx verifica a env var:
   ```
   ONNX_PROVIDER=CUDAExecutionProvider
   ```
   Sem isso, usa CPU mesmo com `onnxruntime-gpu` instalado.

3. **Verificar providers disponíveis**:
   ```python
   import onnxruntime
   print(onnxruntime.get_available_providers())
   # Deve mostrar: ['CUDAExecutionProvider', 'CPUExecutionProvider']
   ```

---

## 2. ASR — Automatic Speech Recognition

### Faster-Whisper Local (CPU) — Não Recomendado para PT-BR

| Modelo | Params | Qualidade PT-BR | Latência CPU |
|---|---|---|---|
| `base` | 74M | Péssima — transcrições sem sentido | ~1.5s |
| `small` | 244M | Ruim — erros frequentes | ~3s |
| `large-v3-turbo` | 809M | Excelente | Inviável em CPU |

**Conclusão**: Para PT-BR com qualidade, `large-v3-turbo` em GPU é obrigatório. Modelos menores não têm vocabulário suficiente para português.

### Configuração Ideal

- **Provider**: Remote gRPC apontando para `stt-server` com GPU
- **Modelo**: `large-v3-turbo` (melhor custo-benefício qualidade/velocidade)
- **GPU recomendada**: RTX 4070+ (12GB VRAM suficiente)

---

## 3. VAD — Voice Activity Detection

### webrtcvad vs Silero VAD

| Aspecto | webrtcvad | Silero VAD |
|---|---|---|
| Tipo | Rule-based (GMM) | ML-based (ONNX, ~2MB) |
| Output | Binário (speech/não) | Probabilidade 0.0-1.0 |
| Ruído/digitação | Muitos falsos positivos | Excelente discriminação |
| Eco/reverberação | Sensível | Robusto |
| Latência | <1ms | ~1-2ms |
| Usado por | Projetos legados | LiveKit, Pipecat, Daily |

### Silero VAD — Configuração

```python
from silero_vad import load_silero_vad

model = load_silero_vad(onnx=True)  # ~2MB, roda em CPU

# Processa chunks de 256 samples @ 8kHz (32ms)
samples = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / 32768.0
audio_tensor = torch.from_numpy(samples)
prob = model(audio_tensor, 8000).item()

# prob >= 0.5 = speech detectado
```

**Importante**: Chamar `model.reset_states()` ao resetar a sessão (modelo tem estado interno LSTM).

---

## 4. Latência — Análise do Pipeline

### Breakdown por turno de conversa

```
Usuário fala → [VAD detecta fim] → ASR → LLM → TTS → Usuário ouve
                    ~200ms          ~300ms  ~500ms  ~800ms
                                    (GPU)   (API)   (CPU)
```

**Total típico**: ~1.8s (end-to-end)

### Onde otimizar (em ordem de impacto)

1. **TTS em GPU** — Maior ganho: ~800ms CPU → ~400ms GPU (Kokoro ONNX)
2. **LLM streaming + sentence pipeline** — Não esperar resposta completa; sintetizar por sentença
3. **ASR em GPU** — large-v3-turbo em GPU: ~300ms vs inviável em CPU
4. **VAD silence_duration_ms** — Reduzir de 500ms para 200ms (resposta mais rápida, risco de cortar)

### Sentence Pipeline

O `SentencePipeline` é crítico para latência percebida:
- LLM faz streaming de tokens
- Ao detectar fim de sentença (`.!?`), dispara TTS imediatamente
- Primeira sentença chega ao usuário ~500ms antes da resposta completa

---

## 5. Deploy — Vast.ai

### Armadilhas Descobertas

1. **`vastai update` + `reboot` NÃO puxa nova imagem Docker**
   - A instância continua rodando a imagem antiga
   - **Solução**: `vastai destroy instance` + `vastai create instance` com nova imagem

2. **WORKDIR do Dockerfile é ignorado pelo `onstart`**
   - Vast.ai executa `onstart` com CWD=/root, não /app
   - **Solução**: Usar caminhos absolutos em env vars e scripts
   ```
   KOKORO_MODEL_DIR=/app/models/kokoro  # NÃO usar: models/kokoro
   ```

3. **SSH requer `openssh-server` no Dockerfile**
   - Container precisa rodar como root para SSH funcionar
   - Incluir `mkdir -p /var/run/sshd` no Dockerfile

4. **Matar processo principal desconecta SSH**
   - `kill $(pgrep -f server.py)` mata o entrypoint do container
   - Para debug, usar screen/tmux ou rebuild da imagem

### Template de Deploy Vast.ai

```bash
vastai create instance <GPU_ID> \
  --image paulohenriquevn/tts-server-kokoro-gpu:latest \
  --disk 20 \
  --onstart-cmd "cd /app && python3 server.py" \
  --env "GRPC_PORT=50070 TTS_PROVIDER=kokoro KOKORO_MODEL_DIR=/app/models/kokoro ONNX_PROVIDER=CUDAExecutionProvider"
```

---

## 6. Configuração Final Recomendada

### `.env` para melhor qualidade + latência

```env
# ASR — GPU remoto (large-v3-turbo)
ASR_PROVIDER=remote
ASR_REMOTE_TARGET=<stt-server-ip>:50060

# TTS — Kokoro GPU remoto
TTS_PROVIDER=remote
TTS_REMOTE_TARGET=<tts-server-ip>:50070

# LLM — Claude Haiku (mais rápido para conversação)
LLM_PROVIDER=anthropic
LLM_MODEL=claude-haiku-4-5-20251001

# VAD — Silero (ML-based)
# Configurado via TurnDetection no protocol
```

### Portas

| Serviço | Porta | GPU necessária |
|---|---|---|
| OpenVoiceAPI (WebSocket) | 8765 | Não |
| STT Server (gRPC) | 50060 | Sim (RTX 4070+) |
| TTS Server (gRPC) | 50070 | Sim (RTX 2080 Ti+) |

---

## 7. Lições Gerais

1. **Qualidade de ASR depende dramaticamente do tamanho do modelo para idiomas não-inglês**. Modelos "base" e "small" do Whisper são praticamente inutilizáveis para PT-BR.

2. **TTS self-hosted evoluiu muito**. Kokoro-ONNX oferece qualidade comparável a APIs cloud para PT-BR, com latência aceitável mesmo em CPU.

3. **VAD baseado em ML é essencial para agentes de voz**. webrtcvad gera muitos falsos positivos com ruído ambiente, digitação e eco — inaceitável para produção.

4. **Streaming por sentença é o maior ganho de latência percebida**. A diferença entre esperar 2s pela resposta completa vs ouvir a primeira sentença em 0.8s é transformadora para UX.

5. **GPU hosting (Vast.ai) tem armadilhas não-documentadas**. Sempre usar caminhos absolutos, sempre destruir/recriar instâncias para trocar imagens, sempre incluir SSH no Dockerfile.

6. **ONNX Runtime GPU precisa de configuração explícita**. Instalar `onnxruntime-gpu` não é suficiente — o provider CUDA precisa ser selecionado via env var ou código.
