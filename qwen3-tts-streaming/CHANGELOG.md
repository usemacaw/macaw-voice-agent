# Changelog

## [Unreleased]

### Added
- `_log_vram()` helper em `streaming.py` — logging estruturado de GPU VRAM (allocated/reserved MB) após prefill e geração
- `_GPU_EXECUTOR` em `model.py` — ThreadPoolExecutor dedicado (max_workers=1) para GPU em async wrappers, evita thread starvation do asyncio default pool
- `MacawTTS.LOCK_TIMEOUT_S` — timeout configurável (30s default) para lock de geração, com mensagem de erro diagnóstica
- `_MAX_REF_AUDIO_SIZE` (50MB) e validação de tamanho/vazio em `_validate_audio_path()` — previne upload de arquivos maliciosamente grandes
- `_num_position_types` atributo em `TalkerCUDAGraph` — documenta e parametriza shape de position_ids (3 para Qwen3-TTS RoPE)
- `_mask_pos_buf` em `TalkerCUDAGraph` — buffer pré-alocado para posição de mask, elimina `torch.tensor()` allocation por step
- `try/finally` com `crossfader.reset()` em `streaming_generate()` — garante cleanup do crossfader em barge-in/cancelamento
- `gen.close()` em `astream()`/`astream_voice_clone()` — cleanup explícito do sync generator em cancelamento async
- `_check_upstream_compat()` em `MacawTTS` — verifica na warmup que APIs upstream (generate_speaker_prompt, generate_icl_prompt, talker, speech_tokenizer) ainda existem, loga warning se divergir
- `pad_token_id` parâmetro em `StreamingDecoder` — permite configurar token de padding para compiled decode (default: codec_pad_id do modelo)
- `MacawTTS.MIN_FREE_VRAM_MB` (256MB) e `_check_vram_available()` — circuit breaker que verifica VRAM livre antes de iniciar geração, previne OOM mid-generation
- 16 testes novos: VRAM logging, mask pos buf, position_ids shape, lock timeout, GPU executor, path validation, crossfader cleanup, concurrent lock, decoder pad token, upstream compat, VRAM circuit breaker, predictor reset ordering (total: 115)

### Changed
- **PERF:** EOS check movido do início para o final do loop de geração — `.item()` sync agora se sobrepõe com GPU work já enfileirado
- **PERF:** `_MASK_CACHE_MAX_SIZE` reduzido de 64 para 16 entries — VRAM peak de ~1GB para ~256MB sem impacto em desempenho (acesso sequencial, cada posição usada uma vez)
- **PERF:** `astream()`/`astream_voice_clone()` usam `_GPU_EXECUTOR` dedicado em vez do default asyncio executor — previne thread starvation com múltiplas conexões
- Lock de geração mudou de reject imediato (`blocking=False`) para espera com timeout (30s) — requests concorrentes são enfileirados em vez de rejeitados
- f-string no logger de `decoder.py` substituída por `%`-style para lazy formatting
- f-strings no logger de `streaming.py` (timeout, max_seq_len) substituídas por `%`-style
- Comentário documentando por que `output_tokens.clone()` é mandatório no `PredictorCUDAGraph.run()`
- Documentação expandida do `_BLOCKED_PATH_PREFIXES` explicando por que `/etc` não é bloqueado
- Documentação de segurança em `PredictorCUDAGraph.run()` e `capture()` — explica por que `StaticCache.reset()` DEVE estar fora do CUDA graph (Python-side state `cumulative_length` em `StaticSlidingWindowLayer` não é capturado)
- `StreamingDecoder._decode_padded()` usa `codec_pad_id` em vez de zeros para left-padding — reduz artefatos de áudio por bleed-through do decoder neural
- `transformers` pin documentação expandida com checklist de 5 passos para upgrade seguro
- `from_pretrained()` passa `codec_pad_id` do config do modelo ao StreamingDecoder

### Fixed
- **CRÍTICO:** Crossfader retinha ~21ms de áudio sem cleanup em barge-in (stream interruption) — `finally` block garante `crossfader.reset()` em qualquer path de saída
- **CRÍTICO:** Async generator em `astream()` não chamava `gen.close()` em cancelamento — sync generator ficava pendente até GC
- **CRÍTICO:** `gen.close()` em `astream()` executava no asyncio thread em vez do GPU executor thread — race condition com `threading.Lock` no `_stream_impl`. Movido para `run_in_executor(_GPU_EXECUTOR, gen.close)`
- **CRÍTICO:** Early return path no prefill EOS check não estava coberto pelo `try/finally` — crossfader não era resetado se modelo produzia EOS imediatamente. `try` block agora engloba o early return
- **PERF:** `_compute_mask()` alocava novo `torch.tensor([position])` a cada chamada no hot loop — substituído por buffer pré-alocado `_mask_pos_buf`

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
- `max_wall_time_s` parameter em `streaming_generate()`, `stream()` e `stream_voice_clone()` — previne gerações presas se o modelo nunca produzir EOS (default: 60s)
- Bloqueio de paths de sistema (`/proc`, `/sys`, `/dev`) na validação de `ref_audio` para voice cloning
- Referência de commit upstream na documentação de `_build_talker_inputs()` para rastrear divergências
- `HannCrossfader.drain()` — retorna tail pendente com fade-out para end-of-stream (previne perda de ~21ms de áudio)
- `HannCrossfader.has_pending_tail` property para checar se há samples retidos
- Threading lock em `MacawTTS` — previne acesso concorrente a CUDA graph static buffers (RuntimeError se duas streams simultâneas)
- `MacawTTS.astream()` e `astream_voice_clone()` — async wrappers via `run_in_executor` para uso com asyncio sem bloquear o event loop
- 14 testes novos: LRU mask cache, path validation de diretórios de sistema, crossfade drain/tail, flush drains tail, _step tensor, ChunkMetadata.num_frames (total: 99)
- Validação de `ref_audio` path em voice cloning para prevenir path traversal e acesso a arquivos não-áudio
- Guards de captura em `TalkerCUDAGraph.run()` e `PredictorCUDAGraph.run()` — RuntimeError se captura não foi feita
- Error recovery nos métodos `capture()` — reset de estado GPU em caso de falha durante captura
- `__repr__` em `MacawTTS`, `TalkerCUDAGraph`, `PredictorCUDAGraph` para debugging em produção
- 12 testes novos: guards de captura, repr, path validation, PCM precision, sampling tensor ops (total: 85)

### Changed
- Attention mask cache em `TalkerCUDAGraph` migrado para LRU com limite de 64 entries — previne OOM em gerações longas (antes: cache unbounded podia atingir ~32GB VRAM)
- EOS check no hot loop usa `token.eq(eos_tensor)` com tensor pré-alocado no device
- `past_hidden` no loop de geração usa buffer pré-alocado com `.copy_()` em vez de `.clone()` por step — elimina allocation churn (~4MB/geração)
- `ChunkMetadata.num_frames` agora reporta frames reais emitidos (antes: reportava o target `emit_every`)
- `_validate_audio_path()` verifica existência do arquivo antes da extensão (fail-fast) e bloqueia diretórios de sistema
- Documentação de segurança do `StaticCache.reset()` no `PredictorCUDAGraph.run()` — explica por que reset antes de replay é correto
- `CircularRepetitionPenalty._step` migrado de Python int para tensor no device — elimina potencial CPU→GPU sync na indexação do buffer circular
- `float32_to_pcm16()` agora usa `* 32768` com clipping simétrico com `pcm16_to_float32()` — `-1.0` mapeia para `-32768` (exato) em vez de `-32767`
- `flush()` do `_StreamingEmitter` agora drena o tail pendente do crossfader quando todos os frames já foram emitidos — previne perda de ~21ms de áudio no final de cada utterance
- Imports de `transformers` (StaticCache, masking_utils) e `scipy` movidos para module-level — elimina ~50-200ms de latência na primeira chamada
- `set_generation_state()` em `TalkerCUDAGraph` tensorizado — substitui Python loop por operações tensoriais (`arange >= pad_counts`)
- Logging estruturado com `extra={}` dict nos pontos críticos de métricas (TTFA, prefill, generation summary) — compatível com Datadog/Loki/JSON formatters
- `transformers` pin documentado em `pyproject.toml` com instruções de upgrade
- Duplicação de `_Predictor` class nos testes extraída para `_make_predictor()` factory
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
- `_StreamingEmitter.label` parameter — dead code após migração para logging estruturado (nunca passado nos call sites)
- `streaming_generate_dynamic()` — dead code (140 linhas, nunca chamado)
- `compile_talker_for_streaming()` — dead code (33 linhas, nunca chamado)
- `StreamingMetrics` dataclass — nunca instanciado
- `SAMPLE_WIDTH` constante — definida mas nunca referenciada
- `StreamingDecoder._get_device_params()` — método nunca chamado
- Dead code branch em `from_pretrained()` (check `attn_implementation != "sdpa"` com variável hardcoded)
- Diretório vazio `benchmarks/`
