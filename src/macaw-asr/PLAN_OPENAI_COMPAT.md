# Plano: Compatibilidade com OpenAI Audio API

> macaw-asr como drop-in replacement da OpenAI Audio API.
> Qualquer SDK OpenAI existente funciona apontando para localhost:8766.

---

## Endpoints a Implementar

### Prioridade 1 — Core (MVP funcional)

| OpenAI Endpoint | Método | Implementar? |
|-----------------|--------|-------------|
| `POST /v1/audio/transcriptions` | multipart/form-data | **Sim** — core do produto |
| `POST /v1/audio/translations` | multipart/form-data | **Sim** — traduz para EN |
| `GET /v1/models` | JSON | **Sim** — lista modelos disponíveis |

### Prioridade 2 — TTS (futuro, quando macaw-tts estiver pronto)

| OpenAI Endpoint | Método | Implementar? |
|-----------------|--------|-------------|
| `POST /v1/audio/speech` | JSON → audio stream | **Futuro** |
| `POST /v1/audio/voices` | multipart | **Futuro** |
| Voice consents CRUD | vários | **Futuro** |

### Prioridade 3 — Features avançadas

| Feature | Status |
|---------|--------|
| `response_format=verbose_json` (timestamps) | Fase 2 |
| `response_format=srt/vtt` | Fase 2 |
| `response_format=diarized_json` | Fase 3 (precisa de modelo de diarização) |
| `stream=true` (SSE text deltas) | Fase 2 |
| `logprobs` | Fase 3 |
| `timestamp_granularities[]=word` | Fase 3 |

---

## Fase 1 — POST /v1/audio/transcriptions (Core)

### Request Format (multipart/form-data)

```bash
curl http://localhost:8766/v1/audio/transcriptions \
  -H "Authorization: Bearer sk-anything" \
  -F file=@audio.wav \
  -F model=qwen \
  -F language=pt \
  -F response_format=json
```

**Campos do form:**
- `file` (required): arquivo de áudio (wav, mp3, m4a, webm, etc.)
- `model` (required): "qwen", "whisper-1", etc. Mapeia para modelos internos
- `language` (optional): ISO-639-1
- `response_format` (optional): "json" (default), "text", "verbose_json", "srt", "vtt"
- `prompt` (optional): contexto/hint para o modelo
- `temperature` (optional): 0.0-1.0
- `stream` (optional): true/false (SSE streaming)
- `timestamp_granularities[]` (optional): "word", "segment"

### Response Format — `json` (default)

```json
{
  "text": "Olá, como você está?"
}
```

### Response Format — `verbose_json`

```json
{
  "task": "transcribe",
  "language": "portuguese",
  "duration": 3.5,
  "text": "Olá, como você está?",
  "segments": [
    {
      "id": 0,
      "start": 0.0,
      "end": 3.5,
      "text": "Olá, como você está?",
      "tokens": [],
      "temperature": 0.0,
      "avg_logprob": 0.0,
      "compression_ratio": 1.0,
      "no_speech_prob": 0.0,
      "seek": 0
    }
  ],
  "usage": {
    "type": "duration",
    "seconds": 4
  }
}
```

### Response Format — `text`

```
Olá, como você está?
```

### Model Mapping

| OpenAI model | macaw-asr model |
|-------------|-----------------|
| `whisper-1` | `qwen` (default) |
| `gpt-4o-transcribe` | `qwen` |
| `gpt-4o-mini-transcribe` | `qwen` |
| `qwen` | `qwen` (native) |

### Audio Format Support

Aceitar via ffmpeg/soundfile: wav, mp3, m4a, webm, ogg, flac, mpga.
Converter internamente para PCM16 no sample_rate do modelo.

### Usage Field

```json
{
  "type": "duration",
  "seconds": 4
}
```

### Auth

- Aceitar `Authorization: Bearer <anything>` — não validar (local server)
- Ou: ignorar header completamente

---

## Fase 2 — Streaming + verbose_json + srt/vtt

### Streaming (SSE)

```bash
curl http://localhost:8766/v1/audio/transcriptions \
  -F file=@audio.wav \
  -F model=qwen \
  -F stream=true
```

Response (SSE):
```
data: {"type":"transcript.text.delta","delta":"Olá"}
data: {"type":"transcript.text.delta","delta":", como"}
data: {"type":"transcript.text.delta","delta":" você está?"}
data: {"type":"transcript.text.done","text":"Olá, como você está?","usage":{"type":"duration","seconds":4}}
```

### SRT Format

```
1
00:00:00,000 --> 00:00:03,500
Olá, como você está?
```

### VTT Format

```
WEBVTT

00:00:00.000 --> 00:00:03.500
Olá, como você está?
```

---

## Fase 3 — Translation + Features avançadas

### POST /v1/audio/translations

Mesmo formato que transcriptions, mas traduz para inglês.

### Diarization, logprobs, word timestamps

Dependem de modelos/features que ainda não temos.

---

## Implementação Técnica

### Mudanças no server/app.py

1. Novo endpoint `POST /v1/audio/transcriptions` (multipart)
2. Manter endpoints Ollama existentes (`/api/transcribe`, `/api/tags`, etc.)
3. Ambas APIs coexistem no mesmo server
4. Audio decode via `soundfile` ou `ffmpeg` (wav, mp3, webm, etc.)

### Novo módulo: `audio/decode.py`

Decodifica qualquer formato de áudio para PCM16 float32:
```python
def decode_audio_file(file_bytes: bytes, filename: str) -> tuple[np.ndarray, int]:
    """Decode audio file to float32 array + sample_rate."""
```

### Testes

Extrair golden tests da spec OpenAI:
- `test_openai_compat_json.py` — response format json
- `test_openai_compat_text.py` — response format text
- `test_openai_compat_verbose.py` — verbose_json
- `test_openai_compat_streaming.py` — SSE stream
- `test_openai_compat_multipart.py` — file upload
- `test_openai_compat_models.py` — model mapping
- `test_openai_compat_errors.py` — error format

### Compatibilidade com OpenAI Python SDK

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

---

## Ordem de Execução

```
Fase 1 (core transcriptions) → Testes → Fase 2 (streaming + formats) → Fase 3 (translation + advanced)
```

---

## DoDs Fase 1

- [ ] `POST /v1/audio/transcriptions` aceita multipart com file upload
- [ ] Aceita wav, mp3, webm (pelo menos)
- [ ] `response_format=json` retorna `{"text": "..."}`
- [ ] `response_format=text` retorna plain text
- [ ] `model=whisper-1` mapeia para qwen
- [ ] `language=pt` funciona
- [ ] Auth header ignorado gracefully
- [ ] OpenAI Python SDK funciona sem modificação
- [ ] `curl -F file=@audio.wav -F model=whisper-1` funciona
- [ ] Testes com audio WAV real
