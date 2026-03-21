# Changelog

## [Unreleased]

### Added
- Biblioteca `macaw-qwen3-tts-streaming` para streaming real de Qwen3-TTS
- `sampling.py` — Token sampling com temperature, top-k, top-p e repetition penalty circular GPU
- `crossfade.py` — Hann crossfade entre chunks de áudio (512 samples overlap)
- `audio.py` — Resampling 24kHz↔8kHz e conversão float32↔PCM16
- `talker_graph.py` — CUDA graph para Talker single-token decode (~8ms/step)
- `predictor_graph.py` — CUDA graph para CodePredictor 15-step loop (~4ms/step)
- `decoder.py` — Streaming Code2Wav com torch.compile e sliding window (25-frame context)
- `streaming.py` — Core streaming generator com two-phase latency (Phase 1: 1 frame, Phase 2: 4 frames)
- `model.py` — MacawTTS wrapper com `stream()` e `stream_voice_clone()` APIs
- 73 testes unitários cobrindo 7/8 módulos — 100% CPU, sem GPU
- Plano executável documentado em `PLANO_EXECUTAVEL.md` com evidências dos 4 projetos estudados

### Added
- Validação de `ref_audio` path em voice cloning para prevenir path traversal e acesso a arquivos não-áudio
- Guards de captura em `TalkerCUDAGraph.run()` e `PredictorCUDAGraph.run()` — RuntimeError se captura não foi feita
- Error recovery nos métodos `capture()` — reset de estado GPU em caso de falha durante captura
- `__repr__` em `MacawTTS`, `TalkerCUDAGraph`, `PredictorCUDAGraph` para debugging em produção
- 12 testes novos: guards de captura, repr, path validation, PCM precision, sampling tensor ops (total: 85)

### Changed
- `_build_talker_inputs()`: parâmetro `m` renomeado para `qwen_model` para clareza
- `stream()` e `stream_voice_clone()` refatorados para usar `_stream_impl()` (elimina duplicação)
- `__init__.py` exporta apenas tipos realmente utilizados
- `rope_deltas = None` documentado com comentário explicativo
- Attention masks agora são lazy (computadas sob demanda e cacheadas) em vez de pré-construir 2048 masks no init — reduz consumo de VRAM de O(max_seq_len) para O(posições usadas)
- `_build_suppress_mask()` usa operações tensoriais em vez de loop Python (1024 iterações eliminadas)
- `transformers` pinado em `>=4.45,<4.50` para evitar breakage de APIs internas (`masking_utils`, `StaticCache`)
- Removido `pytest-asyncio` de dev deps (não havia testes assíncronos)

### Fixed
- **CRÍTICO:** `_sync_cuda()` recursão infinita corrigida (chamava a si mesma em vez de `torch.cuda.synchronize()`)
- **CRÍTICO:** `CircularRepetitionPenalty.update()` eliminado sync CPU↔GPU — `.item()` substituído por operação tensorial pura
- **CRÍTICO:** `TalkerCUDAGraph` pré-alocava 2048 attention masks consumindo até 448MB VRAM — migrado para lazy computation
- `stream_voice_clone()` chamava `m._build_assistant_text` no inner model em vez de `self._model`
- Duplicate `_warmup()` call removida em `stream_voice_clone()`
- `decoder.py` docstring corrigido: samples_per_frame = 1920 (não 2000)
- `pcm16_to_float32()` divisor corrigido de 32767 para 32768 — valor mais negativo (-32768) agora mapeia corretamente para [-1, 1]
- Documentado `position_ids` shape `[3, 1, 1]` — 3 tipos de posição para Qwen3-TTS RoPE
- Documentada limitação: PredictorCUDAGraph sampling params são fixos após captura do CUDA graph

### Removed
- `streaming_generate_dynamic()` — dead code (140 linhas, nunca chamado)
- `compile_talker_for_streaming()` — dead code (33 linhas, nunca chamado)
- `StreamingMetrics` dataclass — nunca instanciado
- `SAMPLE_WIDTH` constante — definida mas nunca referenciada
- `StreamingDecoder._get_device_params()` — método nunca chamado
- Dead code branch em `from_pretrained()` (check `attn_implementation != "sdpa"` com variável hardcoded)
- Diretório vazio `benchmarks/`
