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
- Provider `macaw_streaming_tts.py` para integração com microserviço TTS do Macaw Voice Agent
- 40 testes unitários (sampling, crossfade, audio) — 100% CPU, sem GPU
- Plano executável documentado em `PLANO_EXECUTAVEL.md` com evidências dos 4 projetos estudados
