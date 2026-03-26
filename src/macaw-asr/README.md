# macaw-asr

Self-contained ASR engine with pluggable models. Drop-in replacement for the OpenAI Audio API.

```bash
pip install macaw-asr[faster-whisper]
macaw-asr pull faster-whisper-small
macaw-asr serve
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8766/v1", api_key="unused")
result = client.audio.transcriptions.create(
    model="whisper-1",
    file=open("audio.wav", "rb"),
    language="pt",
)
print(result.text)
```

## Models

| Model | ID | Family | Backend | Params | PT-BR | Streaming |
|-------|-----|--------|---------|--------|-------|-----------|
| Faster-Whisper Tiny | `faster-whisper-tiny` | faster-whisper | CTranslate2 | 39M | Yes | No (batch) |
| Faster-Whisper Small | `faster-whisper-small` | faster-whisper | CTranslate2 | 244M | Yes | No (batch) |
| Faster-Whisper Medium | `faster-whisper-medium` | faster-whisper | CTranslate2 | 769M | Yes | No (batch) |
| Faster-Whisper Large | `faster-whisper-large` | faster-whisper | CTranslate2 | 1.5B | Yes | No (batch) |
| Qwen3-ASR | `qwen` | qwen | PyTorch | 0.6B | Yes | Yes (token-by-token SSE) |
| Whisper Tiny | `whisper-tiny` | whisper | PyTorch | 39M | Yes | No (batch) |
| Whisper Small | `whisper-small` | whisper | PyTorch | 244M | Yes | No (batch) |
| Whisper Medium | `whisper-medium` | whisper | PyTorch | 769M | Yes | No (batch) |
| Whisper Large | `whisper-large` | whisper | PyTorch | 1.5B | Yes | No (batch) |
| Parakeet TDT | `parakeet` | parakeet | NeMo | 0.6B | Yes (best) | No (batch) |
| Parakeet CTC | `parakeet-ctc` | parakeet | NeMo | 1.1B | Yes | No (batch) |

## Install

The core package is lightweight (~5MB). Each model family installs only its own dependencies:

```bash
pip install macaw-asr                    # Core only (API server, CLI)
pip install macaw-asr[faster-whisper]    # Faster-Whisper (CTranslate2, ~200MB, no PyTorch)
pip install macaw-asr[whisper]           # Whisper (PyTorch + transformers)
pip install macaw-asr[qwen]             # Qwen3-ASR (PyTorch + transformers + qwen-asr)
pip install macaw-asr[parakeet]         # Parakeet (PyTorch + NeMo)
pip install macaw-asr[all]              # Everything
```

## Quick Start

```bash
# Pull a model (resolves short names automatically)
macaw-asr pull faster-whisper-small

# Start server
macaw-asr serve

# Or specify model via env
MACAW_ASR_MODEL=faster-whisper-small macaw-asr serve

# Transcribe a file directly (no server needed)
macaw-asr transcribe audio.wav --model faster-whisper-small

# List all available models and dependency status
macaw-asr list --all
```

If dependencies are missing, the CLI tells you exactly what to install:

```
$ macaw-asr pull parakeet
Resolved: parakeet -> nvidia/parakeet-tdt-0.6b-v3 (family: parakeet)
Missing dependencies: nemo
Install with: pip install "macaw-asr[parakeet]"
```

## CLI

```bash
macaw-asr pull <model>         # Download model (short name or HuggingFace ID)
macaw-asr serve                # Start HTTP server (:8766)
macaw-asr transcribe <file>    # Transcribe audio file
macaw-asr list                 # List downloaded models
macaw-asr list --all           # List all available models with dep status
macaw-asr remove <model>       # Remove a downloaded model
```

## API Endpoints

### OpenAI-Compatible (drop-in replacement)

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/v1/audio/transcriptions` | Transcribe audio (multipart file upload) |
| `POST` | `/v1/audio/translations` | Translate audio to English |
| `GET` | `/v1/models` | List available models |

### Operational

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/show` | Show model details |
| `GET` | `/api/ps` | List running/loaded models |
| `POST` | `/api/pull` | Download a model |
| `DELETE` | `/api/delete` | Remove a model |
| `GET` | `/api/version` | Server version |
| `GET` | `/` | Health check |

### Response Formats

```bash
# JSON (default)
curl -F file=@audio.wav -F model=whisper-1 http://localhost:8766/v1/audio/transcriptions
# {"text": "Hello world", "usage": {"type": "duration", "seconds": 3}}

# Plain text
curl -F file=@audio.wav -F model=whisper-1 -F response_format=text http://localhost:8766/v1/audio/transcriptions
# Hello world

# Verbose JSON (with timestamps)
curl -F file=@audio.wav -F model=whisper-1 -F response_format=verbose_json http://localhost:8766/v1/audio/transcriptions

# SRT subtitles
curl -F file=@audio.wav -F model=whisper-1 -F response_format=srt http://localhost:8766/v1/audio/transcriptions

# VTT subtitles
curl -F file=@audio.wav -F model=whisper-1 -F response_format=vtt http://localhost:8766/v1/audio/transcriptions

# SSE Streaming (token-by-token, Qwen only)
curl -F file=@audio.wav -F model=whisper-1 -F stream=true http://localhost:8766/v1/audio/transcriptions
# data: {"type":"transcript.text.delta","delta":"Hello"}
# data: {"type":"transcript.text.delta","delta":" world"}
# data: {"type":"transcript.text.done","text":"Hello world","usage":{"type":"duration","seconds":3}}
```

### Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8766/v1", api_key="unused")

# Basic transcription
result = client.audio.transcriptions.create(
    model="whisper-1", file=open("audio.wav", "rb"),
)
print(result.text)

# Verbose JSON
result = client.audio.transcriptions.create(
    model="whisper-1", file=open("audio.wav", "rb"),
    response_format="verbose_json",
)
print(f"Duration: {result.duration}s, Language: {result.language}")

# List models
for model in client.models.list().data:
    print(f"  {model.id}")
```

## Adding a New Model

Adding a model requires **only 2 steps** — no changes to server, routes, engine, or tests.

### Step 1: Register metadata in `models/registry.py`

```python
register(ModelMeta(
    name="my-model",
    model_id="org/my-model-v1",
    module="macaw_asr.models.my_model",
    family="my-family",
    param_size="1.0B",
    dtype="float16",
))
```

### Step 2: Create the model module

```
src/models/my_model/
├── __init__.py          # register_model("my-model", MyASRModel)
├── loader.py            # MyModelLoader(IModelLoader) — load/unload weights
├── preprocessor.py      # MyPreprocessor(IPreprocessor) — audio → tensors
├── decoder.py           # MyDecoder(IDecoder, IStreamDecoder) — tensors → text
└── model.py             # MyASRModel(IASRModel) — facade composing above
```

Each component implements a rigid interface:

```python
# loader.py
class MyModelLoader(IModelLoader):
    def load(self, config: EngineConfig) -> None: ...
    def unload(self) -> None: ...
    def is_loaded(self) -> bool: ...

# preprocessor.py
class MyPreprocessor(IPreprocessor):
    def prepare_inputs(self, audio: np.ndarray, prefix: str = "") -> Any: ...

# decoder.py
class MyDecoder(IDecoder, IStreamDecoder):
    def decode(self, inputs: Any, strategy: DecodeStrategy | None = None) -> ModelOutput: ...
    def decode_stream(self, inputs, strategy) -> Generator[tuple[str, bool, ModelOutput | None]]: ...

# model.py — facade
class MyASRModel(IASRModel):
    def load(self, config): ...
    def unload(self): ...
    def prepare_inputs(self, audio, prefix=""): ...
    def generate(self, inputs, strategy=None): ...
    def generate_stream(self, inputs, strategy=None): ...
    # Properties: eos_token_id, supports_streaming, supports_cuda_graphs
```

That's it. `macaw-asr serve` with `MACAW_ASR_MODEL=my-model` will load and serve it via the OpenAI API.

## Architecture

```
src/
├── models/                          # Model layer (SOLID + Design Patterns)
│   ├── contracts.py                 # IModelLoader, IPreprocessor, IDecoder, IStreamDecoder, IASRModel
│   ├── registry.py                  # Model metadata + FamilyDeps (dependency management)
│   ├── factory.py                   # ModelFactory (Factory Pattern, lazy loading)
│   ├── types.py                     # ModelOutput, InputsWrapper, timing constants
│   ├── qwen/                        # Qwen3-ASR (autoregressive, streaming)
│   │   ├── loader.py                # Weight lifecycle
│   │   ├── preprocessor.py          # GPU mel + CPU fallback
│   │   ├── decoder.py               # Manual decode loop with KV cache (DRY)
│   │   ├── prompt.py                # Chat template builder
│   │   └── model.py                 # Facade (composes all above)
│   ├── whisper/                     # OpenAI Whisper (PyTorch, encoder-decoder, batch)
│   │   ├── loader.py, preprocessor.py, decoder.py, model.py
│   ├── faster_whisper/              # Faster-Whisper (CTranslate2, no PyTorch)
│   │   ├── loader.py, preprocessor.py, decoder.py, model.py
│   ├── parakeet/                    # NVIDIA Parakeet TDT (NeMo, batch)
│   │   ├── loader.py, preprocessor.py, decoder.py, model.py
│   └── mock/                        # Mock for testing
│
├── runner/                          # Inference orchestration
│   ├── contracts.py                 # IEngine, ISession
│   ├── engine.py                    # ASREngine — lifecycle + dispatch
│   └── session.py                   # StreamingSession — per-connection state
│
├── server/                          # HTTP layer (FastAPI)
│   ├── contracts.py                 # IScheduler
│   ├── scheduler.py                 # Model caching, eviction
│   ├── app.py                       # Thin composition (routers + middleware)
│   └── routes/
│       ├── audio.py                 # /v1/audio/* (OpenAI compat)
│       ├── models.py                # /v1/models, /api/show,ps,pull,delete
│       └── system.py                # /, /api/version
│
├── audio/                           # Audio processing (stateless)
│   ├── preprocessing.py             # PCM, resample, AudioPreprocessor
│   ├── accumulator.py               # Chunk accumulation for streaming
│   └── decode.py                    # File decoder (wav, mp3, webm via ffmpeg)
│
├── decode/                          # Decode strategies (Strategy Pattern)
│   ├── strategies.py                # GreedyWithEarlyStopping
│   └── postprocess.py               # Text cleanup
│
├── manifest/                        # Model storage (Repository Pattern)
│   ├── contracts.py                 # IModelPaths, IModelRegistry
│   ├── paths.py                     # ~/.macaw-asr/models/ layout
│   └── registry.py                  # Download, cache, resolve (short name support)
│
├── api/                             # Wire types + client
│   ├── types.py                     # Pydantic models (OpenAI format)
│   └── client.py                    # HTTP client (dogfooding)
│
├── cmd/cli.py                       # CLI: serve, pull, list, transcribe
├── config.py                        # Frozen dataclass configs
└── _executor.py                     # Thread pool for async/sync boundary
```

### Design Patterns

| Pattern | Where | Purpose |
|---------|-------|---------|
| Factory | `ModelFactory.create("qwen")` | Create model instances by name |
| Facade | `QwenASRModel` | Compose loader+preprocessor+decoder |
| Strategy | `DecodeStrategy` | Pluggable decode stopping logic |
| Template Method | `QwenDecoder._decode_loop()` | Shared loop for batch + stream |
| Registry | `models/registry.py` | Single source of truth for models + deps |
| Repository | `ModelRegistry` | Abstract model storage |
| Router (MVC) | `routes/audio.py`, `routes/models.py` | Separate HTTP concerns |
| Dependency Injection | `app.state` | No globals, testable |

### SOLID

| Principle | Implementation |
|-----------|---------------|
| **S** — Single Responsibility | Each file has one job: loader loads, decoder decodes, router routes |
| **O** — Open/Closed | New model = new directory. Zero changes to existing code |
| **L** — Liskov Substitution | Mock, Qwen, Whisper, Faster-Whisper, Parakeet all substitute via IASRModel |
| **I** — Interface Segregation | IModelLoader, IPreprocessor, IDecoder, IStreamDecoder — small focused interfaces |
| **D** — Dependency Inversion | Engine depends on IASRModel, not QwenASRModel. Routes use app.state, not globals |

## Testing

```bash
# Run with specific model on GPU
MACAW_ASR_TEST_MODEL=qwen pytest tests/
MACAW_ASR_TEST_MODEL=whisper-tiny pytest tests/
MACAW_ASR_TEST_MODEL=faster-whisper-tiny pytest tests/

# API tests (mock model, no GPU)
pytest tests/test_api_server.py tests/test_openai_compat.py

# Performance benchmarks (GPU required)
pytest tests/test_perf_model.py tests/test_perf_engine.py -s

# WER evaluation (GPU + FLEURS dataset)
pytest tests/test_perf_wer.py -s
```

## Configuration

All via environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `MACAW_ASR_MODEL` | `qwen` | Model registry name |
| `MACAW_ASR_MODEL_ID` | `Qwen/Qwen3-ASR-0.6B` | HuggingFace model ID |
| `MACAW_ASR_DEVICE` | `cuda:0` | Inference device |
| `MACAW_ASR_DTYPE` | `bfloat16` | Model dtype |
| `MACAW_ASR_LANGUAGE` | `pt` | Default language |
| `MACAW_ASR_MAX_NEW_TOKENS` | `32` | Max decode tokens |
| `MACAW_ASR_CHUNK_SIZE_SEC` | `1.0` | Streaming chunk trigger |
| `MACAW_ASR_HOME` | `~/.macaw-asr` | Local model storage |

## License

MIT
