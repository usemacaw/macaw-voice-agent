# Koel-TTS & Magpie TTS: Pesquisa Profunda

> **Koel-TTS Paper:** *Enhancing LLM based Speech Generation with Preference Alignment and Classifier Free Guidance*
> **Autores:** Shehzeen Hussain, Paarth Neekhara, Xuesong Yang, Edresson Casanova, et al. (NVIDIA)
> **Publicado:** EMNLP 2025 | arXiv [2502.05236](https://arxiv.org/abs/2502.05236)
> **Magpie TTS:** Versão de produção do Koel-TTS, disponível no [NeMo](https://github.com/NVIDIA/NeMo) e [NVIDIA NIM](https://build.nvidia.com/nvidia/magpie-tts-multilingual)

---

## 1. Relação entre os Modelos

```
T5-TTS (2406.17957)          Koel-TTS (2502.05236)          Magpie TTS (produção)
  │ Arquitetura base            │ + CFG + DPO/RPO               │ + GRPO + NanoCodec
  │ Attention priors            │ Preference alignment           │ + Long-form + Streaming
  │ CTC alignment loss          │ 380M EN / 1.1B multilingual    │ 357M, 9 idiomas
  └─────────────────────────────┴────────────────────────────────┘
```

| Aspecto | Koel-TTS (paper) | Magpie TTS (produção) |
|---------|:--|:--|
| Params | 380M (EN), 1.1B (multilingual) | 357M (241M treináveis + 116M codec frozen) |
| Idiomas | 6 (EN, DE, NL, ES, FR, IT) | 9 (EN, ES, DE, FR, IT, VI, ZH, HI, JA) |
| Dados treino | ~21K horas | ~38K horas (de 60K coletadas) |
| Codec | LFSC (21.5 FPS, 1.89 kbps) | NanoCodec (21.5 FPS, 1.89 kbps) |
| Preference optim. | DPO + RPO (offline) | DPO + **GRPO** (online, recomendado) |
| Long-form | Não | Sim (sentence-level chunking) |
| Streaming | Não | Sim (via Riva NIM) |
| Weights abertos | Não | Sim (HuggingFace + NeMo) |

---

## 2. Arquitetura Completa

### 2.1 Visão Geral

```
┌─────────────────────────────────────────────────────────────┐
│                    Koel-TTS / Magpie TTS                     │
│                                                              │
│  ┌────────────────┐         ┌───────────────────────────┐   │
│  │  Text Encoder   │         │   AR Decoder              │   │
│  │  6 layers NAR   │──xattn──│   12 layers Causal        │   │
│  │  d=768, h=12    │         │   d=768, h=12             │   │
│  │  FFN=3072       │         │   xattn h=1, d_head=128   │   │
│  │  Causal Conv k=3│         │   Causal Conv FFN k=1     │   │
│  └────────────────┘         │                           │   │
│         ↑                    │   ↑ self-attn              │   │
│     [Fonemas/Chars]          │   [Context Audio Tokens]   │   │
│                              │   + [Target Audio Tokens]  │   │
│                              │         ↓                  │   │
│                              │   Linear → N×2^m logits    │   │
│                              │   (8 codebooks parallel)   │   │
│                              └───────────────────────────┘   │
│                                        ↓                     │
│  ┌──────────────────┐    ┌──────────────────────────────┐   │
│  │ Local Transformer │    │ NanoCodec Decoder (frozen)    │   │
│  │ 1 layer, h=1     │────│ 31.6M params, HiFi-GAN       │   │
│  │ d=256             │    │ Tokens → Waveform 22.05kHz   │   │
│  └──────────────────┘    └──────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Text Encoder (6 layers, Non-Causal)

| Parâmetro | Valor |
|-----------|:--:|
| Layers | 6 |
| Hidden dim (d_model) | 768 |
| FFN dim | 3072 |
| Self-attention heads | 12 |
| Causal convolution kernel | 3 |
| Dropout | 0.1 |
| Positional encoding | Learnable (max 2048) |
| Layer norm | Output |
| Cross-attention | Não (encoder não tem xattn) |

**Input:** fonemas IPA (EN, DE, ES) ou caracteres byte-level (todos os idiomas). O modelo multilíngue usa um **tokenizer agregado** com embeddings separados por idioma.

**Descoberta importante:** tokenização por **caracteres** funciona tão bem quanto fonemas, simplificando extensão multilíngue.

### 2.3 AR Decoder (12 layers, Causal)

| Parâmetro | Valor |
|-----------|:--:|
| Layers | 12 |
| Hidden dim (d_model) | 768 |
| FFN dim | 3072 |
| Self-attention heads | 12 |
| **Cross-attention heads** | **1** |
| Cross-attention head dim | 128 |
| Cross-attention memory dim | 768 |
| Kernel (FFN) | 1 |
| Dropout | 0.1 |
| Positional encoding | Learnable (max 2048) |
| Causal masking | Sim |

**Peculiaridade arquitetural:** o FFN sublayer usa **convolução causal** (kernel 3 no encoder, kernel 1 no decoder) em vez do MLP padrão. Isso adiciona contexto local sem custo computacional significativo.

### 2.4 Context Encoder (Speaker Conditioning)

O Magpie TTS usa o modo **`decoder_ce`** (decoder context with context encoder):

| Parâmetro | Valor |
|-----------|:--:|
| Layers | 1 (non-causal) |
| Hidden dim | 768 |
| FFN dim | 3072 |
| Self-attention heads | 12 |

O context encoder processa tokens de áudio de referência (5s do speaker) e alimenta o decoder via self-attention.

### 2.5 Local Transformer (Refinamento)

| Parâmetro | Valor |
|-----------|:--:|
| Type | Autoregressive (ou MaskGit) |
| Layers | 1 |
| Heads | 1 |
| Hidden dim | 256 |
| Loss scale | 1.0 |

**Propósito:** quando frame stacking está habilitado (fator > 1), o decoder base processa múltiplos frames por forward pass (otimização de velocidade). O local transformer expande frames agrupados em predições frame-a-frame individuais. Com frame stacking=1, ainda refina predições de codebook.

### 2.6 Predição Multi-Codebook

**Todos os 8 codebooks são preditos em paralelo** a cada timestep. Não precisa de delay pattern porque FSQ tem codebooks **independentes** (diferente de RVQ onde codebooks são residuais/hierárquicos).

```
Timestep t:
  Input:  e_t = Σ Embed_i(C[t-1, i])  para i=1..8  (soma dos embeddings do passo anterior)
  Output: Linear(h_t) → vetor de tamanho N × 2^m  (8 × 2016 = 16128 logits)
  Loss:   Cross-entropy sobre cada codebook independentemente
```

### 2.7 Três Variantes de Condicionamento de Speaker (Koel-TTS paper)

| Variante | Como funciona | Zero-shot (unseen) | Seen speakers |
|----------|:--|:--:|:--:|
| **Decoder Context** | Tokens de contexto prepended aos tokens target no decoder | **Melhor** (SSIM 0.637) | Bom |
| SV Conditioned | TitaNet embedding → projetado + somado ao output do text encoder | Razoável (0.619) | Bom |
| Multi Encoder | 3 layers separados para contexto, cross-attn alternada | Pior (0.601) | **Melhor** (overfits) |

**Conclusão:** Decoder Context generaliza melhor para speakers não vistos no treinamento — adotado como padrão.

---

## 3. Mecanismos de Alinhamento

### 3.1 Attention Priors (Beta-Binomial)

Distribuição beta-binomial 2D aplicada element-wise nas matrizes de cross-attention para **forçar alinhamento monofônico** entre texto e fala (previne hallucinations: pular palavras, repetições, desalinhamento).

| Parâmetro | Valor |
|-----------|:--:|
| Prior scaling factor | 0.5 |
| Prior ativo até step | 12.000 |
| Annealing linear começa | 8.000 |
| Annealing para | Matriz uniforme (ones) |

### 3.2 CTC Alignment Loss

Loss CTC aplicada nas matrizes de cross-attention para reforçar sequências monofônicas.

```
L_total = L_token + α × L_align
```

| Parâmetro | Valor |
|-----------|:--:|
| α (alignment_loss_scale) | 0.002 |
| Aplicada sobre | Todas as cross-attention heads e layers |

---

## 4. Classifier-Free Guidance (CFG)

### 4.1 Treinamento

Durante treinamento, **drop out aleatório de ambas** as condições (texto E áudio de contexto) com probabilidade **10%**. Isso treina o modelo a produzir outputs condicionais e incondicionais.

### 4.2 Inferência

```
logits_cfg = γ × logits_condicionais + (1 - γ) × logits_incondicionais
```

| Parâmetro | Valor |
|-----------|:--:|
| γ (CFG scale) ótimo | **2.5** |
| Sweep testado | 1.0 a 3.0 (intervalo 0.2) |
| Custo computacional | **2× forward passes** por step |

### 4.3 Resultado: CFG é Complementar a DPO/RPO

CFG e preference alignment **se somam multiplicativamente** — não são redundantes:

| Configuração | CER↓ | SSIM↑ | MOS↑ |
|-------------|:--:|:--:|:--:|
| Baseline | 2.68 | 0.637 | 4.35 |
| + DPO | 0.89 | 0.667 | 4.40 |
| + CFG | 0.57 | 0.720 | 4.42 |
| **+ DPO + CFG** | **0.55** | **0.729** | **4.41** |
| **+ RPO + CFG** | **0.55** | **0.729** | **4.42** |

---

## 5. Preference Alignment

### 5.1 DPO/RPO (Koel-TTS — offline)

#### Construção de Dados de Preferência

```
1. 800 textos difíceis (gerados por Llama-8B: repetições, aliterações, sequências fonéticas complexas)
2. 50.000 transcrições regulares do dataset de treino
3. Para cada: gerar P=6 amostras via top-k sampling (k=80, temp=0.7)
4. Avaliar cada amostra:
   - CER via Parakeet-TDT 1.1B (EN) ou Whisper-large-v3 (multilingual)
   - SSIM via TitaNet-Large (cosine similarity)
5. Ranking Pareto-optimal: fronts não-dominados, priorizar CER baixo
6. chosen = melhor ranking, rejected = pior ranking
7. Descartar pares onde chosen é pior em QUALQUER métrica que rejected
```

**Total:** 58.000 pares texto-contexto (EN) + 10K/idioma (multilingual)

#### Objetivo DPO

```
L_DPO = E[β × log(π(y_c|x) / π_ref(y_c|x)) − β × log(π(y_l|x) / π_ref(y_l|x))]
```

#### Objetivo RPO (Reward-aware)

```
L_RPO = D[β × log(π(y_c|x)/π_ref(y_c|x)) − β × log(π(y_l|x)/π_ref(y_l|x)) || η × (r*(y_c) − r*(y_l))]
```

Onde D[a||b] é a divergência KL reversa, e o reward gap usa CDF da normal padrão sobre CER e SSIM normalizados.

#### Hyperparâmetros DPO/RPO

| Parâmetro | Valor |
|-----------|:--:|
| Max iterations | 4.000 |
| Batch size | 64 pares |
| Optimizer | Adam |
| Learning rate | 2e-7 (fixo) |
| β (DPO) | 0.01 |
| β (RPO) | 0.01 |
| η (RPO) | 1.0 |

**Insight:** RPO é **menos sensível** a hyperparameters que DPO. DPO é mais sensível ao β.

**Armadilha SpeechAlign:** usar áudio ground-truth como chosen (estilo SpeechAlign) causa degeneração — loss cai para ~zero em centenas de iterations, CER explode para >90%. GT tokens têm distribuição fundamentalmente diferente de tokens gerados pelo modelo.

### 5.2 GRPO (Magpie TTS — online, recomendado)

GRPO (Group Relative Policy Optimization) gera candidatos **on-the-fly** durante treino. Mais simples e eficaz que DPO offline.

#### Como Funciona

```
Para cada exemplo (texto + contexto):
  1. Gerar 12 candidatos de áudio (num_generations_per_item=12)
  2. Para cada candidato, computar 3 rewards:
     - CER reward (peso 0.33): via Parakeet-TDT 1.1B ou Whisper
     - SSIM reward (peso 0.33): via TitaNet-Large
     - PESQ reward (peso 0.33): qualidade perceptual de fala
  3. Normalizar rewards dentro do grupo (vantagens relativas, dividir por std)
  4. Otimizar via policy gradient para maximizar rewards
```

#### Hyperparâmetros GRPO

| Parâmetro | Valor |
|-----------|:--:|
| Learning rate | 1e-7 |
| Batch size | 2 (× 12 gerações = 24 forward passes) |
| Temperature (inferência) | 0.8 |
| Top-k (inferência) | 2016 (efetivamente desabilitado) |
| reference_free | true (sem KL divergence) |
| scale_rewards | true (normaliza por std dev) |
| Gradient clipping | 2.5 |
| Precisão | 32-bit |
| Decoder dropout | **0.0** (CRÍTICO: deve ser desabilitado) |
| Validation interval | 50 steps |

**Crítico:** durante GRPO, dropout, attention priors e CTC loss devem ser **todos desabilitados** para KL divergence estável.

---

## 6. Dados de Treinamento

### 6.1 Koel-TTS (Paper)

**Inglês (18K horas):**

| Dataset | Horas |
|---------|:--:|
| LibriTTS train-clean-360 | ~360 |
| LibriTTS train-clean-100 | ~100 |
| HiFiTTS | ~292 |
| LibriVox MLS subset | ~17.000 |
| Proprietário (2 speakers) | 63 |
| **Total EN** | **~18.000** |

**Multilingual (adicional ~3K horas):**

| Idioma | Dataset | Horas |
|--------|---------|:--:|
| Alemão | CML-TTS | 1.562 |
| Holandês | CML-TTS | 642 |
| Espanhol | CML-TTS + interno | 518 |
| Francês | CML-TTS | 283 |
| Italiano | CML-TTS | 131 |
| **Total multilingual** | | **~21.000** |

### 6.2 Magpie TTS (Produção)

**60.000 horas coletadas → ~38.000 horas após filtragem**, 9 idiomas:

| Idioma | Datasets |
|--------|:--|
| English | HiFi-TTS, HiFi-TTS-2, LibriTTS, 17K LibriVox/MLS, interno |
| Spanish | CML-TTS Es, interno |
| French | CML-TTS Fr, interno |
| German | CML-TTS De |
| Italian | CML-TTS It |
| Vietnamese | LSVSC, InfoRe-1, InfoRe-2, interno |
| Mandarin Chinese | Interno |
| Hindi | AI4Bharat Kathbath |
| Japanese | Emilia YODAS, Common Voice 17 Ja, interno |

**Compute:** ~1.62 × 10²¹ FLOPs, ~2.230 kWh energia (~0.72 tCO2e)

---

## 7. Configuração de Treinamento

### 7.1 Base Model (Koel-TTS)

| Parâmetro | 380M (EN) | 1.1B (multilingual) |
|-----------|:--:|:--:|
| GPUs | 16× A100 | 32× A100 |
| Steps | ~200K | ~150K |
| Batch size global | 256 | 256 |
| Optimizer | Adam | Adam |
| LR inicial | 1e-4 | 1e-4 |
| LR schedule | Exponential decay (γ=0.998) a cada 1000 steps | Idem |
| Contexto de speaker | 5s (random slice, mesma pessoa, outro utterance) | 5s |
| Loss | Cross-entropy + CTC alignment (α=0.002) | Idem |

### 7.2 Magpie TTS (Config YAML)

| Parâmetro | Valor |
|-----------|:--:|
| Optimizer | AdamW |
| LR | 2e-4 |
| LR scheduler | ExponentialLR (γ=0.998) |
| Batch size | 16 |
| Gradient clipping | 2.5 |
| Precisão | 32-bit |
| Dropout | 0.1 |
| Max audio duration | 20.0s |
| Min audio duration | 0.2s |
| Contexto speaker | 5.0s (fixo) |
| CFG dropout prob | 0.1 (10%) |

### 7.3 Koel-TTS 1.1B Multilingual (Arquitetura)

| Componente | Valor |
|-----------|:--:|
| Decoder layers | 16 (vs 12 no 380M) |
| Hidden dim | 1536 (vs 768) |
| FFN dim | 6144 (vs 3072) |
| Encoder layers | 6 (mesmo) |
| Total params | 1.1B |

---

## 8. Inferência

### 8.1 Parâmetros Padrão

| Parâmetro | Valor |
|-----------|:--:|
| Sampling | Multinomial top-k |
| Top-k | 80 |
| Temperature | 0.6 |
| CFG scale (γ) | 2.5 |
| Contexto speaker | 5s slice |
| Output sample rate | 22.050 Hz |
| Output format | PCM16 mono WAV |
| Max geração | 30s (standard) |

### 8.2 Long-Form Mode (Magpie TTS)

Sentence-level chunking com preservação de estado:

```
1. Texto split em sentenças (pontuação: . ? ! ...)
2. LongformChunkState rastreia: tokens de histórico, contexto encoder, posições de attention
3. Para cada sentença:
   a. Prepend contexto anterior
   b. Aplicar attention prior aprendido
   c. Gerar tokens de áudio autoregressivamente
   d. Atualizar estado
4. Concatenar tokens de áudio de todos os chunks
5. Decodificar pelo NanoCodec
```

**Thresholds por idioma (~20s de áudio):** EN=45 words, ES=73, FR=69, etc.
Max decoder steps: 50.000.

### 8.3 Latência de Streaming (via Riva NIM)

**Magpie TTS Multilingual em A100:**

| Streams Simultâneos | Latência 1º Áudio (avg/p99 ms) | Latência Inter-chunk (avg/p99 ms) | Throughput (RTFX) |
|:--:|:--:|:--:|:--:|
| 1 | **77 / 91** | 15 / 17 | 10.43× |
| 4 | 88 / 108 | 17 / 19 | 35.49× |
| 32 | 192 / 229 | 49 / 56 | 123.64× |
| 64 | 305 / 418 | 99 / 121 | 152.84× |

**Em H100:** 1 stream = 96ms primeiro áudio, até 170.57× RTFX com 64 streams.

**Magpie TTS Zeroshot em A100:** 373ms primeiro áudio (1 stream), 1.04× RTFX — **significativamente mais lento** que o modelo fixo.

---

## 9. Resultados: Benchmarks Completos

### 9.1 SOTA Comparison (test-clean LibriTTS, unseen speakers)

| Modelo | CER↓ (%) | WER↓ (%) | SSIM↑ | MOS↑ | SMOS↑ |
|--------|:--:|:--:|:--:|:--:|:--:|
| Ground Truth | 0.80 | 1.83 | 0.771 | 3.94 | — |
| VALLE-X | 6.65 | 11.28 | 0.675 | 3.53 | 3.72 |
| YourTTS | 2.44 | 5.19 | 0.583 | 3.24 | 3.23 |
| T5-TTS | 1.66 | 3.28 | 0.459 | 3.53 | 3.37 |
| XTTS-v2 | 0.99 | 2.09 | 0.680 | 3.72 | 3.43 |
| E2-TTS | 1.29 | 2.66 | **0.848** | 3.89 | 3.80 |
| F5-TTS | 1.23 | 2.55 | 0.834 | 3.93 | 3.79 |
| StyleTTS-2 | 0.75 | 1.52 | 0.579 | 4.05 | 3.79 |
| **Koel-TTS 380M** | **0.55** | **1.41** | 0.729 | **4.05** | **3.83** |
| **Koel-TTS 1.1B** | 0.63 | 1.42 | 0.740 | 4.06 | 3.85 |

**Destaques:**
- **Melhor CER/WER** de todos — até melhor que ground truth em WER!
- **Maior MOS e SMOS** — humanos preferem Koel-TTS em naturalidade e similaridade
- SSIM abaixo de E2-TTS/F5-TTS que usam **100K+ horas** (vs 21K do Koel)
- Testes CMOS: humanos preferem Koel-TTS sobre **todos** os sistemas concorrentes

### 9.2 Magpie TTS Multilingual (v2602)

| Idioma | CER↓ (%) | SV-SSIM↑ |
|--------|:--:|:--:|
| English (US) | **0.34** | **83.49** |
| Spanish | 1.14 | 71.53 |
| German | 0.66 | 62.59 |
| French | 2.70 | 70.34 |
| Italian | 4.00 | 66.71 |
| Vietnamese | 0.60 | 72.35 |
| Hindi | 0.86 | 75.59 |
| Japanese | 1.12 | 74.82 |
| Mandarin | 4.24 | — |

**CER 0.34% em inglês** — melhor que os resultados do paper (0.55%).

### 9.3 Ablation: Efeito de Cada Técnica

| Técnica | CER↓ | SSIM↑ | MOS↑ | Δ CER vs Baseline |
|---------|:--:|:--:|:--:|:--:|
| Baseline | 2.68 | 0.637 | 4.35 | — |
| + DPO | 0.89 | 0.667 | 4.40 | -67% |
| + RPO | 1.17 | 0.681 | 4.40 | -56% |
| + CFG | 0.57 | 0.720 | 4.42 | -79% |
| + DPO + CFG | **0.55** | **0.729** | 4.41 | **-79%** |
| + RPO + CFG | **0.55** | **0.729** | **4.42** | **-79%** |

**CFG sozinho** já reduz CER em 79%. DPO/RPO adicionam ganho marginal em CER mas **significativo em SSIM** (+0.009 sobre CFG sozinho).

---

## 10. Voice Cloning

### 10.1 Koel-TTS (Paper)

- Condicionamento via 5s de áudio de referência (outro utterance do mesmo speaker)
- Decoder Context: tokens prepended ao input do decoder → melhor generalização zero-shot
- Não disponível publicamente como modelo standalone

### 10.2 Magpie TTS (Produção)

Três modelos separados:

| Modelo | Voice Cloning | Speakers | Latência 1ª Áudio |
|--------|:--:|:--:|:--:|
| **Multilingual** (357M) | Não (5 speakers fixos) | Sofia, Aria, Jason, Leo, John Van Stan | **77ms** (A100) |
| **Zeroshot** | Sim (5s referência) | Qualquer | 373ms (A100) |
| **Flow** (450M) | Sim (5s + transcrição) | Qualquer | Mais lento |

**Segurança:** zero-shot foi removido de algumas distribuições públicas. NVIDIA colabora com Pindrop para deepfake detection.

---

## 11. Inovações Técnicas Chave

### 11.1 Lista de Contribuições Originais

1. **Predição multi-codebook paralela** — habilitada pela independência dos codebooks FSQ. Sem delay pattern.

2. **Preference alignment via ranking Pareto-optimal** — usa CER (ASR) + SSIM (speaker verification) como dual rewards. Primeiro sistema TTS com DPO/RPO.

3. **CFG para predição autoregressiva de tokens** — dropout de texto E áudio (10%), interpolação na inferência com γ=2.5. Ganho de 79% em CER.

4. **Ganhos complementares** — DPO/RPO + CFG stackam multiplicativamente. Não são redundantes.

5. **Convolução causal no FFN** — kernel 3 no encoder, kernel 1 no decoder. Adiciona contexto local.

6. **Attention prior (beta-binomial) + CTC alignment loss** — força alinhamento monofônico texto-fala. Previne hallucinations.

7. **Tokenização por caracteres** funciona tão bem quanto fonemas — simplifica extensão multilíngue.

8. **GRPO (Magpie)** — preference optimization online, sem necessidade de pré-gerar dados de preferência.

### 11.2 Por Que Funciona Tão Bem

```
CER low        ← CFG força aderência ao texto
                ← Attention prior + CTC forçam alinhamento monofônico
                ← DPO/GRPO otimizam contra ASR reward

SSIM high      ← Decoder Context generaliza para unseen speakers
                ← CFG melhora condicionamento de speaker
                ← DPO/GRPO otimizam contra speaker verification reward

MOS high       ← NanoCodec preserva qualidade na reconstrução
                ← GRPO otimiza contra PESQ reward
                ← Dados de 38K+ horas diversificados
```

---

## 12. Reproduzindo: O Que Seria Necessário

### 12.1 Para Replicar Koel-TTS (Paper Scale)

| Recurso | Quantidade |
|---------|:--:|
| GPUs | 16-32× A100 80GB |
| Dados | ~18-21K horas (LibriTTS, HiFiTTS, MLS são públicos) |
| Codec | NanoCodec (download de HuggingFace) |
| Framework | NeMo 2.0+ |
| Steps base | ~200K (batch 256) |
| Steps DPO | ~4K (batch 64) |
| Tempo estimado | ~1-2 semanas (base) + ~1 dia (DPO) |

### 12.2 Para Replicar Magpie TTS (Production Scale)

| Recurso | Quantidade |
|---------|:--:|
| GPUs | 32-64× A100 80GB (estimado) |
| Dados | ~38-60K horas (inclui datasets proprietários) |
| GRPO | 12 gerações por exemplo → 24 forward passes por step |
| Tempo estimado | ~semanas |

### 12.3 Para Português Brasileiro (Caminho Mínimo)

```
1. Fine-tune NanoCodec com dados PT-BR (se necessário — o codec já é multilíngue 105 idiomas)
2. Coletar ~1-5K horas de áudio PT-BR (Common Voice PT = ~150h, MLS PT = ~3.6K horas)
3. Treinar Magpie TTS adicionando PT-BR ao tokenizer agregado
4. GRPO com ASR PT-BR (Whisper) + TitaNet para speaker similarity
5. Estimar: 8-16× A100, ~100K steps, ~3-5 dias
```

---

## 13. Referências Completas

### Papers
- **T5-TTS (arquitetura base):** [arxiv 2406.17957](https://arxiv.org/abs/2406.17957) — Monotonic alignment
- **Koel-TTS:** [arxiv 2502.05236](https://arxiv.org/abs/2502.05236) — CFG + DPO/RPO, EMNLP 2025
- **NanoCodec:** [arxiv 2508.05835](https://arxiv.org/abs/2508.05835) — Low frame-rate FSQ codec
- **LFSC (predecessor):** [arxiv 2409.12117](https://arxiv.org/abs/2409.12117)

### Modelos e Código
- **Magpie TTS HuggingFace:** [nvidia/magpie_tts_multilingual_357m](https://huggingface.co/nvidia/magpie_tts_multilingual_357m)
- **NanoCodec HuggingFace:** [nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps](https://huggingface.co/nvidia/nemo-nano-codec-22khz-1.89kbps-21.5fps)
- **NVIDIA NIM (API):** [build.nvidia.com/nvidia/magpie-tts-multilingual](https://build.nvidia.com/nvidia/magpie-tts-multilingual)
- **NeMo GitHub:** [github.com/NVIDIA/NeMo](https://github.com/NVIDIA/NeMo)
- **Demo Koel-TTS:** [koeltts.github.io](https://koeltts.github.io/)
- **Demo NanoCodec:** [edresson.github.io/NanoCodec](https://edresson.github.io/NanoCodec/)

### Benchmarks
- **Riva NIM Performance:** [docs.nvidia.com/nim/riva/tts/latest/performance.html](https://docs.nvidia.com/nim/riva/tts/latest/performance.html)
- **Magpie TTS Docs:** [docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tts/magpietts.html](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/tts/magpietts.html)
