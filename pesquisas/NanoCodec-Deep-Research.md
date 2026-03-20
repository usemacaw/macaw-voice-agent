# NanoCodec: Pesquisa Profunda

> **Paper:** *NanoCodec: Towards High-Quality Ultra Fast Speech LLM Inference*
> **Autores:** Edresson Casanova, Paarth Neekhara, Ryan Langman, Shehzeen Hussain, et al.
> **Instituição:** NVIDIA Corporation
> **Data:** Agosto 2025
> **arXiv:** [2508.05835v1](https://arxiv.org/abs/2508.05835v1)
> **Licença:** NVIDIA Open Model License Agreement (Junho 2024) — uso comercial e não-comercial permitido

---

## 1. O Que É o NanoCodec

NanoCodec é um **codec neural de áudio** (NAC) que comprime waveforms de fala em tokens discretos e os reconstrói de volta em áudio. Ele **não é um TTS** — é a camada de tokenização que permite Speech LLMs (como Koel-TTS ou Magpie TTS) tratar áudio como uma sequência de tokens, da mesma forma que LLMs de texto tratam palavras como tokens.

### Papel no Pipeline TTS

```
[Texto] → Speech LLM (gera tokens de áudio) → NanoCodec Decoder (tokens → waveform) → [Áudio]
                                                      ↑
                                              É aqui que o NanoCodec atua
```

Para **treinar** o Speech LLM:
```
[Áudio ground truth] → NanoCodec Encoder (waveform → tokens) → [Tokens target para o LLM aprender]
```

---

## 2. Arquitetura Completa

### 2.1 Visão Geral

```
                        NanoCodec (62M params total)
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│  Encoder (30.4M)          FSQ              Decoder (31.6M)       │
│  ┌──────────────┐   ┌────────────┐   ┌──────────────────────┐   │
│  │ 5 Res Blocks │→→→│ 8-13 CBs   │→→→│ HiFi-GAN Causal      │   │
│  │ Non-causal   │   │ 4D per CB  │   │ Snake Activation      │   │
│  │ MRF Modules  │   │ FSQ Levels │   │ Upsampling reverso    │   │
│  └──────────────┘   └────────────┘   └──────────────────────┘   │
│                                                                  │
│  3 Discriminadores:                                              │
│  • Multi-Period (HiFi-GAN)                                      │
│  • Multi-Band Multi-Scale STFT (DAC)                             │
│  • WavLM-based                                                   │
│                                                                  │
│  Losses: Squared-GAN + Feature-matching + Speaker Consistency    │
└──────────────────────────────────────────────────────────────────┘
```

### 2.2 Encoder (Non-Causal, 30.4M params)

- **5 blocos residuais**, cada um com 3 camadas residuais usando módulos MRF (Multi-Receptive Field Fusion)
- **Dilation rates:** 1, 3, 5 (ao invés de 1 como no LFSC original)
- **Downsampling:** 1D convolutional layer após cada bloco residual
- **Canais iniciais:** 24, dobrados após cada downsampling
- **Strides por frame rate:**
  - 21.5 FPS: [2, 2, 4, 8, 8]
  - 12.5 FPS: [2, 3, 3, 7, 7] → configuração recomendada
  - 6.25 FPS: [2, 3, 6, 7, 7]
- **Non-causal:** usa convoluções padrão (não-causais) — para TTS isso é OK pois o encoder só é usado no treinamento, não na inferência

### 2.3 Decoder (Causal, 31.6M params)

- **Baseado no HiFi-GAN** vocoder com upsampling rates reversos do encoder
- **864 canais iniciais**, reduzidos pela metade após cada upsampling
- **Snake activation** substitui Leaky ReLU (inspirado em BigVGAN)
- **CAUSAL:** convoluções causais substituem as não-causais — **crítico para streaming**
  - Elimina lookahead de 232.5ms que o LFSC non-causal requer
  - Permite decodificação frame-a-frame sem buffer
- **45% menos parâmetros** que o LFSC original graças às convoluções causais

### 2.4 Quantização: Finite Scalar Quantization (FSQ)

FSQ é a alternativa moderna ao RVQ (Residual Vector Quantization) usado em EnCodec/SoundStream:

| Aspecto | RVQ (EnCodec) | FSQ (NanoCodec) |
|---------|:--:|:--:|
| Codebook | Aprendido via gradient | Implícito (produto cartesiano de níveis) |
| Codebook collapse | Problema frequente | **Não sofre** |
| Machinery extra | Commitment loss, reseeding, splitting, entropy penalties | **Nenhuma** |
| Performance | Boa | **Competitiva ou superior** |

**Como o FSQ funciona:**
- Projeta a representação contínua para poucas dimensões (4 dimensões por codebook)
- Cada dimensão é quantizada para um número fixo de níveis
- O codebook é o produto cartesiano dos níveis

**Exemplo:** Níveis [8, 7, 6, 6] → 8 × 7 × 6 × 6 = **2.016 codes** por codebook

---

## 3. Variantes Disponíveis

### 3.1 Checkpoints no Hugging Face

| Variante | Frame Rate | Bitrate | Codebooks | Codes/CB | Embed Dim | FSQ Levels | Uso Recomendado |
|----------|:--:|:--:|:--:|:--:|:--:|:--:|:--|
| **1.78kbps-12.5fps** | 12.5 | 1.78 kbps | 13 | 2.016 | 52 | [8,7,6,6] | **Uso geral, baixa latência** |
| **1.89kbps-21.5fps** | 21.5 | 1.89 kbps | 8 | 2.016 | 32 | [8,7,6,6] | **Magpie TTS** |
| **0.6kbps-12.5fps** | 12.5 | 0.6 kbps | 4 | 4.032 | 16 | [9,8,8,7] | **Fine-tuning com poucos speakers** |

### 3.2 Qual Variante Escolher?

- **Para TTS streaming de baixa latência:** `1.78kbps-12.5fps` — 40% menos tokens por segundo de áudio
- **Para Magpie TTS (NVIDIA):** `1.89kbps-21.5fps` — recomendado explicitamente pela NVIDIA
- **Para fine-tuning com dados limitados / duplex S2S:** `0.6kbps-12.5fps` — NÃO usar como codec genérico

### 3.3 Trade-off Frame Rate vs Qualidade

| Frame Rate | Tokens/s de áudio | SQMOS (MLS) | CER (MLS) | Latência relativa |
|:--:|:--:|:--:|:--:|:--:|
| 25 FPS | 200 | 4.423 | 3.614 | Alta |
| **12.5 FPS** | **100** | **4.441** | **2.423** | **Baixa** |
| 6.25 FPS | 50 | 4.001 | 3.395 | Muito baixa |

**Insight crítico:** 12.5 FPS é o sweet spot. A 6.25 FPS, o codec comprime dois fonemas num frame — degradação significativa para inglês (~10-12 fonemas/s). A 12.5 FPS, cada frame contém ~1 fonema.

---

## 4. Dados de Treinamento

### 4.1 Dataset do Codec (pré-treino)

| Dataset | Horas | Utterances | Speakers | Idiomas |
|---------|:--:|:--:|:--:|:--:|
| Common Voice 11.0 | 3.200h | 2.7M | ~100K | 105 |
| MLS English | 25.500h | 6.2M | 4.329 | 1 (EN) |
| **Total** | **28.700h** | **~8.9M** | **~104K** | **105** |

- Áudio a **22.05 kHz**
- Excertos de **1.1 segundo** durante treinamento
- Batch size acumulado: **1.536** (batch 32 × 48 GPUs)

### 4.2 Dataset do Speech LLM TTS (Koel-TTS)

| Dataset | Horas | Tipo |
|---------|:--:|:--|
| LibriTTS (train-clean 100 + 360) | ~460h | Audiobook, multi-speaker |
| HiFiTTS | ~300h | Audiobook, alta qualidade |
| LibriVox MLS (subset 17K hours) | ~17.000h | Audiobook, multi-speaker |
| **Total** | **~18.000h** | Inglês |

### 4.3 Dataset do Magpie TTS (produção NVIDIA)

| Métrica | Valor |
|---------|:--:|
| **Total** | **60.000h** |
| Idiomas | 9 (EN, ES, DE, FR, IT, VI, ZH, HI, JA) |
| Speakers | ≥2 por idioma (1 masc + 1 fem) |

---

## 5. Configuração de Treinamento

### 5.1 Treinamento do Codec

| Parâmetro | Valor |
|-----------|:--:|
| GPUs | 48× A100 |
| Steps | ~196.000 |
| Batch size | 32 (por GPU) |
| Batch acumulado | 1.536 |
| Tamanho do excerto | 1.1s |
| Amostras processadas | ~301M |
| Optimizer | Adam |
| β₁, β₂ | 0.8, 0.99 |
| Learning rate inicial | 2e-4 |
| LR decay | Exponencial (γ = 0.998) |
| Losses | Squared-GAN + Feature-matching + **Speaker Consistency Loss (α=0.1)** |

### 5.2 Treinamento do Koel-TTS (Speech LLM com NanoCodec)

| Parâmetro | Valor |
|-----------|:--:|
| GPUs | 32× A100 |
| Steps | 200.000 |
| Batch size | 8 (total 256) |
| Optimizer | AdamW |
| Learning rate | 1e-4 |
| Contexto de speaker | 5s (fixo, de outro utterance do mesmo speaker) |
| Inference: Top-k | 80 |
| Inference: CFG scale | 2.5 |
| Inference: Temperature | 0.6 |

### 5.3 Treinamento do Magpie TTS (Produção NVIDIA)

| Parâmetro | Valor |
|-----------|:--:|
| Arquitetura | Causal Transformer Encoder-Decoder |
| Encoder layers | 6 |
| Decoder layers | 12 |
| Params total | 357M (241M treináveis) |
| Codebooks | 8 (multi-codebook prediction) |
| Positional encoding | Learnable (length 2048) |
| Dados | 60.000h, 9 idiomas |
| Codec usado | NanoCodec 1.89kbps-21.5fps |
| Técnicas | Attention priors + CFG + GRPO (Group Relative Policy Optimization) |

---

## 6. Performance: Benchmarks Detalhados

### 6.1 Reconstrução do Codec (encode → decode)

**NanoCodec vs SOTA codecs (mesma faixa de bitrate):**

| Codec | Bitrate | SQMOS↑ | PESQ↑ | Mel Dist↓ | SECS↓ | CER↓ |
|-------|:--:|:--:|:--:|:--:|:--:|:--:|
| **NanoCodec 1.78kbps** | 1.78 | **4.44** | 2.76 | **0.143** | **0.862** | **2.42** |
| LFSC (baseline) | 1.89 | 4.43 | **2.83** | 0.146 | 0.841 | 2.53 |
| Mimi | 1.1 | 4.33 | 2.13 | 0.238 | 0.876 | 7.22 |
| WavTokenizer | 0.9 | 4.00 | 1.87 | 0.207 | 0.576 | 15.37 |
| TS3-Codec | 0.85 | 4.67 | 2.77 | **0.167** | 0.733 | 1.52 |
| TAAE | 0.7 | 4.37 | 2.05 | 0.320 | 0.323 | 18.55 |

**Destaque:** NanoCodec tem o **melhor CER** (inteligibilidade) entre todos os codecs comparados, com qualidade perceptual (SQMOS) competitiva e **melhor speaker similarity** (SECS).

### 6.2 Streaming: NanoCodec vs LFSC

| Modo | NanoCodec PESQ | LFSC PESQ | NanoCodec Lookahead | LFSC Lookahead |
|------|:--:|:--:|:--:|:--:|
| Offline | 2.955 | ~2.95 | 0 frames | 0 frames |
| Streaming | **2.955** | Degradado | **0 frames** | Precisa ≥5 frames (232.5ms) |

**NanoCodec atinge qualidade idêntica offline e streaming sem nenhum lookahead.** LFSC precisa bufferizar 5 frames (232.5ms) para qualidade aceitável em streaming.

### 6.3 Zero-Shot TTS (Koel-TTS + NanoCodec)

| Configuração | FPS | CER↓ | MOS↑ | SECS↓ | RTF↓ | TTFA↓ |
|-------------|:--:|:--:|:--:|:--:|:--:|:--:|
| LFSC 21.5 FPS | 21.5 | 4.17 ± 0.04 | 4.16 ± 0.04 | 0.719 | 2.48 | 2.55 |
| NanoCodec 21.5 FPS | 21.5 | 4.16 ± 0.04 | **4.16 ± 0.04** | 0.638 | 2.47 | 1.14 |
| **NanoCodec 12.5 FPS** | **12.5** | **4.01 ± 0.04** | **4.01 ± 0.04** | 0.655 | **1.06** | **1.01** |
| NanoCodec 12.5 FPS + 10s ctx | 12.5 | **3.84 ± 0.04** | 0.55 ± 0.07 | **0.691** | 1.06 | 1.01 |

**Resultados chave:**
- **RTF 2.33× mais rápido** (1.06 vs 2.48) — gera áudio mais rápido que tempo real
- **TTFA 2.5× mais rápido** (1.01 vs 2.55) — primeiro áudio quase instantâneo
- Speaker similarity menor (0.655 vs 0.719) — melhora para 0.691 com contexto de 10s
- CER melhor a 12.5 FPS do que 21.5 FPS — fonemas e tokens em taxa similar ajuda o alinhamento

---

## 7. Prosódia e Expressividade

### 7.1 O Que o NanoCodec Preserva

O NanoCodec como **codec** preserva:
- **Pitch (F0):** reconstruído fielmente pelo decoder HiFi-GAN
- **Ritmo/timing:** cada frame de 80ms captura a envelope temporal
- **Intensidade/volume:** preservado na reconstrução
- **Qualidade timbral:** Speaker Consistency Loss (SCL) garante preservação de identidade

O paper **não avalia prosódia diretamente** — métricas focam em inteligibilidade (CER), qualidade perceptual (MOS/PESQ), e similaridade de speaker (SECS). Não há métricas de F0 RMSE, voiced/unvoiced accuracy, ou expressividade emocional.

### 7.2 Limitações de Prosódia

- **Bitrate muito baixo (≤0.6 kbps):** CER degrada 2.69× — indica perda de informação prosódica fina
- **Frame rate baixo (6.25 FPS):** comprime 2 fonemas por frame — perde nuances temporais
- **12.5 FPS é o limite seguro:** ~1 fonema por frame preserva timing prosódico adequado

### 7.3 Prosódia no TTS Final

A prosódia e expressividade do TTS final **dependem do Speech LLM**, não do codec:

| Componente | Papel na Prosódia |
|------------|:--|
| **NanoCodec** | Preserva prosódia do áudio original (encode/decode) |
| **Speech LLM** (Koel-TTS, Magpie) | **Gera** prosódia — é aqui que expressividade é controlada |
| **CFG (Classifier-Free Guidance)** | Melhora aderência ao texto e ao speaker de referência |
| **GRPO (Magpie)** | Otimiza naturalidade e alinhamento prosódico |
| **Contexto de speaker** | 5-10s de áudio de referência condiciona estilo prosódico |

O Magpie TTS usa "attention priors, CFG, e GRPO" para expressividade — e modo "long-form" com "rolling window" para manter continuidade prosódica entre sentenças.

---

## 8. É Possível Criar um TTS SOTA com NanoCodec?

### 8.1 Resposta Curta

**Sim, absolutamente.** A NVIDIA já fez isso duas vezes:
1. **Koel-TTS** — SOTA em zero-shot TTS com NanoCodec (paper acadêmico)
2. **Magpie TTS** — modelo de produção da NVIDIA, 9 idiomas, 60K horas, usa NanoCodec 21.5fps

### 8.2 Receita para TTS SOTA com NanoCodec

```
Passo 1: Escolher variante do codec
         ├── 12.5 FPS → menor latência, ideal para streaming
         └── 21.5 FPS → melhor speaker similarity, mais tokens para o LLM trabalhar

Passo 2: Tokenizar dataset de áudio
         └── NanoCodec.encode(audio) → tokens discretos

Passo 3: Treinar Speech LLM
         ├── Input: fonemas/texto + tokens de contexto (speaker)
         ├── Output: tokens de áudio (autoregressivos)
         ├── Arquitetura: Transformer Decoder (Koel-TTS) ou Encoder-Decoder (Magpie)
         └── Técnicas: CFG + Preference Optimization (DPO/GRPO)

Passo 4: Inferência streaming
         ├── Speech LLM gera tokens autoregressivamente
         ├── NanoCodec.decode(tokens) → áudio frame a frame
         └── Decoder causal = zero lookahead = streaming real
```

### 8.3 Requisitos de Dados para TTS SOTA

| Nível | Horas | Idiomas | Qualidade Esperada |
|-------|:--:|:--:|:--|
| Mínimo viável | ~500h | 1 | Inteligível, speaker similarity limitada |
| Competitivo | ~5.000h | 1-3 | Boa qualidade, bom zero-shot |
| **SOTA (Koel-TTS)** | **~18.000h** | **1 (EN)** | **SOTA em CER, MOS, SECS** |
| **Produção (Magpie)** | **~60.000h** | **9** | **Produção multilíngue** |

### 8.4 Requisitos de Compute

| Configuração | GPUs | Steps | Tempo estimado |
|-------------|:--:|:--:|:--:|
| Codec (pré-treino) | 48× A100 | 196K | ~dias |
| Codec (fine-tune) | 4-8× A100 | ~50K | ~horas |
| Speech LLM (Koel-TTS scale) | 32× A100 | 200K | ~dias |
| Speech LLM (Magpie scale) | Não divulgado | Não divulgado | ~semanas |

### 8.5 Vantagens do NanoCodec para TTS SOTA

1. **Menor latência:** 12.5 FPS = 40% menos tokens autoregressivos que LFSC (21.5 FPS)
2. **Streaming real:** decoder causal sem lookahead → primeiro áudio sem buffer
3. **Melhor inteligibilidade:** CER consistentemente melhor que codecs concorrentes
4. **Treinamento mais rápido:** menos tokens = menos steps de LLM = convergência mais rápida
5. **FSQ estável:** sem codebook collapse = treinamento robusto
6. **Speaker identity:** SCL preserva identidade do speaker na reconstrução

### 8.6 Limitações e Riscos

| Limitação | Impacto | Mitigação |
|-----------|:--|:--|
| Speaker similarity menor a 12.5 FPS | 0.655 vs 0.719 (21.5 FPS) | Aumentar contexto para 10s (→ 0.691) |
| Apenas inglês validado para TTS | Koel-TTS só treinou em EN | Magpie prova viabilidade multilíngue |
| Codec treinado majoritariamente em EN | 25.5K das 28.7K horas são EN | Fine-tune com dados do idioma alvo |
| 62M params (codec relativamente grande) | Mais lento que Piper/Kokoro em CPU | Projetado para GPU |
| Sem prosódia emocional explícita | Codec preserva, mas não controla | Depende do Speech LLM |

---

## 9. Integração Prática

### 9.1 API Python

```python
from nemo.collections.tts.models import AudioCodecModel
import librosa, torch, soundfile as sf

# Carregar modelo
codec = AudioCodecModel.from_pretrained(
    "nvidia/nemo-nano-codec-22khz-1.78kbps-12.5fps"
).eval().cuda()

# ENCODE: áudio → tokens
audio, _ = librosa.load("input.wav", sr=codec.sample_rate)
audio_t = torch.from_numpy(audio).unsqueeze(0).cuda()
audio_len = torch.tensor([audio_t.shape[-1]]).cuda()
tokens, tokens_len = codec.encode(audio=audio_t, audio_len=audio_len)
# tokens shape: [batch, num_codebooks, num_frames]

# DECODE: tokens → áudio
reconstructed, _ = codec.decode(tokens=tokens, tokens_len=tokens_len)
sf.write("output.wav", reconstructed.cpu().numpy().squeeze(), codec.sample_rate)
```

### 9.2 Métodos Internos

| Método | Input | Output | Uso |
|--------|:--|:--|:--|
| `encode(audio, audio_len)` | Waveform PCM | Tokens discretos [B, CB, T] | Tokenizar para treino |
| `decode(tokens, tokens_len)` | Tokens discretos | Waveform PCM | Gerar áudio na inferência |
| `encode_audio(audio)` | Waveform | Representação contínua | Debug/análise |
| `quantize(encoded)` | Contínua | Tokens discretos | Debug/análise |
| `dequantize(tokens)` | Tokens discretos | Representação contínua | Debug/análise |
| `decode_audio(continuous)` | Contínua | Waveform | Debug/análise |

### 9.3 Fine-tuning

```bash
# Config: audio_codec_low_frame_rate_22050.yaml
# Pretrained: audio_codec_low_frame_rate_22khz

python audio_codec.py \
  --config-path=<CONFIG_DIR> \
  --config-name=audio_codec_low_frame_rate_22050 \
  max_epochs=100 \
  batch_size=32 \
  +init_from_nemo_model=<CHECKPOINT_PATH> \
  +train_ds_meta.my_data.manifest_path=<MANIFEST_JSONL> \
  +train_ds_meta.my_data.audio_dir=<AUDIO_DIR>
```

**Formato do manifest:**
```json
{"audio_filepath": "speaker01/utterance_001.flac", "duration": 3.45}
{"audio_filepath": "speaker01/utterance_002.flac", "duration": 2.10}
```

---

## 10. Comparação com Outros Codecs

| Codec | FPS | Bitrate | Causal? | Quantização | Params | Streaming sem Lookahead? |
|-------|:--:|:--:|:--:|:--:|:--:|:--:|
| **NanoCodec** | **12.5** | **1.78 kbps** | **Parcial (enc NC, dec C)** | **FSQ** | **62M** | **Sim** |
| LFSC | 21.5 | 1.89 kbps | Não | FSQ | ~113M | Não (232.5ms) |
| Mimi | 12.5 | 1.1 kbps | Sim | RVQ | N/A | Sim |
| EnCodec | 75 | 6 kbps | Sim | RVQ | ~15M | Sim |
| SoundStream | 50 | 6 kbps | Sim | RVQ | N/A | Sim |
| DAC | 86 | 8 kbps | Não | RVQ | N/A | Não |
| WavTokenizer | 75 | 0.9 kbps | N/A | RVQ | N/A | N/A |
| SNAC | 21.5-86 | Variável | Sim | RVQ | N/A | Sim |

---

## 11. Ecossistema NVIDIA: Peças do Puzzle

```
┌─────────────────────────────────────────────────────────┐
│                    NVIDIA Speech Stack                    │
│                                                          │
│  ┌──────────┐   ┌──────────┐   ┌───────────────────┐   │
│  │ NanoCodec │   │ Koel-TTS │   │ Magpie TTS        │   │
│  │ (Codec)  │──→│ (Research)│   │ (Produção, 357M)  │   │
│  │ 62M      │   │ ZS-TTS   │   │ 9 idiomas, 60Kh   │   │
│  └──────────┘   └──────────┘   │ CFG + GRPO         │   │
│       │                         │ NanoCodec 21.5fps  │   │
│       │                         └───────────────────┘   │
│       │                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────────────┐    │
│  │ Parakeet │   │ Canary   │   │ NeMo Framework   │    │
│  │ (ASR)    │   │ (S2S)    │   │ Apache 2.0       │    │
│  └──────────┘   └──────────┘   └──────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

---

## 12. Conclusões e Recomendações

### Para o Macaw Voice Agent

1. **NanoCodec não substitui o Kokoro diretamente** — Kokoro é um TTS end-to-end (texto → áudio). NanoCodec é um codec (áudio ↔ tokens).

2. **NanoCodec seria útil se migrarmos para um Speech LLM** como arquitetura de TTS (ex: Orpheus, Koel-TTS, ou treinar nosso próprio). Nesse cenário, NanoCodec seria a camada de tokenização.

3. **Para TTS SOTA com NanoCodec, precisaríamos:**
   - NanoCodec (codec) — disponível, open model license
   - Um Speech LLM treinado com os tokens do NanoCodec — precisaria treinar ou usar Magpie
   - Dataset substancial (≥5.000h para qualidade competitiva, ≥18.000h para SOTA)
   - Compute significativo (32+ A100s para training)

4. **Alternativa mais pragmática para o Macaw hoje:** usar Qwen3-TTS ou Orpheus (TTS completos com streaming real) e considerar NanoCodec como investimento futuro para um pipeline custom.

### Quando Faz Sentido Usar NanoCodec

| Cenário | Recomendado? |
|---------|:--:|
| Preciso de TTS agora, com pouco compute | **Não** — use Qwen3-TTS, Orpheus, ou Kokoro |
| Vou treinar meu próprio Speech LLM | **Sim** — NanoCodec é o melhor codec disponível |
| Quero usar Magpie TTS da NVIDIA | **Sim** — Magpie já usa NanoCodec 21.5fps |
| Preciso de streaming com latência mínima | **Sim** — decoder causal sem lookahead |
| Meu target é português brasileiro | **Parcialmente** — codec é multilíngue, mas TTS precisa de dados PT-BR |
