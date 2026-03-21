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

### Changed
- `_build_talker_inputs()`: parâmetro `m` renomeado para `qwen_model` para clareza
- `stream()` e `stream_voice_clone()` refatorados para usar `_stream_impl()` (elimina duplicação)
- `__init__.py` exporta apenas tipos realmente utilizados
- `rope_deltas = None` documentado com comentário explicativo

### Fixed
- **CRÍTICO:** `_sync_cuda()` recursão infinita corrigida (chamava a si mesma em vez de `torch.cuda.synchronize()`)
- `stream_voice_clone()` chamava `m._build_assistant_text` no inner model em vez de `self._model`
- Duplicate `_warmup()` call removida em `stream_voice_clone()`
- `decoder.py` docstring corrigido: samples_per_frame = 1920 (não 2000)

### Removed
- `streaming_generate_dynamic()` — dead code (140 linhas, nunca chamado)
- `compile_talker_for_streaming()` — dead code (33 linhas, nunca chamado)
- `StreamingMetrics` dataclass — nunca instanciado
- `SAMPLE_WIDTH` constante — definida mas nunca referenciada
- `StreamingDecoder._get_device_params()` — método nunca chamado
- Dead code branch em `from_pretrained()` (check `attn_implementation != "sdpa"` com variável hardcoded)
- Diretório vazio `benchmarks/`
