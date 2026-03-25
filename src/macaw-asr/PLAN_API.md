# Plano: API HTTP equivalente ao Ollama

> Tornar macaw-asr um drop-in ASR server com a mesma UX dev do Ollama.
> Cada fase tem DoDs verificáveis e testes extraídos dos padrões do Ollama.

---

## Gap Analysis: Ollama vs macaw-asr

| Ollama | macaw-asr hoje | Status |
|--------|----------------|--------|
| Gin HTTP server real | Handlers Python sem framework | ❌ |
| NDJSON streaming responses | Nada | ❌ |
| `POST /api/generate` | `POST /api/transcribe` (handler only) | ⚠️ |
| `POST /api/chat` | N/A (ASR não tem chat) | — |
| `POST /api/embed` | N/A | — |
| `GET /api/tags` (list models) | Handler existe | ⚠️ |
| `GET /api/ps` (list running) | Handler existe | ⚠️ |
| `POST /api/show` | Não existe | ❌ |
| `POST /api/pull` | Handler existe | ⚠️ |
| `DELETE /api/delete` | Handler existe | ⚠️ |
| `GET /api/version` | Não existe | ❌ |
| `GET /` (health) | Handler existe | ⚠️ |
| Client lib (dogfooding) | `api/client.py` (urllib, sync) | ⚠️ |
| CLI `serve` command | Placeholder | ❌ |
| CLI `run` command | `transcribe` command | ⚠️ |
| Error format `{"error": "msg"}` | Não padronizado | ❌ |
| `stream` param (true/false) | Não existe | ❌ |
| `keep_alive` param | Scheduler tem TTL | ⚠️ |
| Test with httptest | Nenhum teste HTTP | ❌ |

---

## Fase 1 — HTTP Server Real (FastAPI)

**Objetivo:** Server HTTP funcional que serve todos os endpoints. Usa FastAPI (Python equivalente ao Gin do Ollama).

**Por que FastAPI:** async nativo, OpenAPI docs automático, validação Pydantic, streaming via StreamingResponse — é o Gin do Python.

### 1.1 Wiring FastAPI

**Arquivos:** `server/app.py` (novo), `server/routes.py` (reescrever)

**DoD:**
- [ ] `macaw-asr serve` inicia FastAPI server na porta 8766
- [ ] Todas as rotas abaixo funcionam via curl
- [ ] Graceful shutdown com SIGINT/SIGTERM
- [ ] Logging de cada request (method, path, status, duration)

### 1.2 Endpoints (mapeamento Ollama → ASR)

| Ollama | macaw-asr | Método | Path |
|--------|-----------|--------|------|
| Generate | Transcribe | `POST` | `/api/transcribe` |
| — | Stream Start | `POST` | `/api/transcribe/stream` |
| Show | Show model | `POST` | `/api/show` |
| List models | List models | `GET` | `/api/tags` |
| List running | List running | `GET` | `/api/ps` |
| Pull | Pull | `POST` | `/api/pull` |
| Delete | Delete | `DELETE` | `/api/delete` |
| Version | Version | `GET` | `/api/version` |
| Health | Health | `GET` | `/` |

### 1.3 Request/Response Types (Pydantic, equivalente ao Ollama api/types.go)

**`POST /api/transcribe`:**
```python
# Request
{
    "model": "qwen",                    # Required
    "audio": "<base64 PCM16>",          # Required
    "language": "pt",                   # Optional
    "stream": false,                    # Optional (default: false)
    "keep_alive": "5m",                 # Optional
    "options": {"max_tokens": 32}       # Optional
}

# Response (non-streaming)
{
    "model": "Qwen/Qwen3-ASR-0.6B",
    "created_at": "2026-03-25T10:00:00Z",
    "text": "Olá mundo",
    "done": true,
    "total_duration": 500000000,        # nanoseconds (Ollama convention)
    "load_duration": 0,
    "prompt_eval_duration": 45000000,
    "eval_count": 5,
    "eval_duration": 350000000
}

# Response (streaming, NDJSON)
{"model":"qwen","created_at":"...","text":"Olá","done":false}
{"model":"qwen","created_at":"...","text":" mundo","done":false}
{"model":"qwen","created_at":"...","text":"","done":true,"total_duration":500000000,...}
```

**`POST /api/show`:**
```python
# Request
{"model": "qwen"}

# Response
{
    "model_info": {
        "general.architecture": "qwen3-asr",
        "general.parameter_count": 600000000
    },
    "details": {
        "family": "qwen3-asr",
        "parameter_size": "0.6B",
        "quantization_level": "BF16"
    }
}
```

**`POST /api/pull`** (streaming progress, NDJSON):
```
{"status":"pulling model"}
{"status":"downloading","digest":"sha256:...","total":1200000000,"completed":500000000}
{"status":"success"}
```

**`GET /api/tags`:**
```python
{
    "models": [
        {
            "name": "qwen",
            "model": "Qwen/Qwen3-ASR-0.6B",
            "size": 1200000000,
            "details": {"family": "qwen3-asr", "parameter_size": "0.6B"}
        }
    ]
}
```

**`GET /api/ps`:**
```python
{
    "models": [
        {
            "name": "qwen",
            "model": "Qwen/Qwen3-ASR-0.6B",
            "size": 1200000000,
            "size_vram": 1200000000,
            "expires_at": "2026-03-25T10:05:00Z"
        }
    ]
}
```

**`GET /api/version`:**
```python
{"version": "0.1.0"}
```

**Error format (Ollama padrão):**
```python
# Todos os erros seguem este formato:
{"error": "model 'xyz' not found"}

# Com status HTTP correto:
# 400 — Bad Request (missing body, invalid params)
# 404 — Not Found (model not found)
# 500 — Internal Server Error
```

### 1.4 NDJSON Streaming (Ollama pattern)

**Ollama usa:** `application/x-ndjson` — cada linha é um JSON completo.

**DoD:**
- [ ] Header `Content-Type: application/x-ndjson` quando streaming
- [ ] Header `Content-Type: application/json` quando não streaming
- [ ] Cada chunk é JSON + `\n`
- [ ] Client pode ler com `readline()` ou `bufio.Scanner`
- [ ] Erro durante streaming enviado como `{"error": "msg"}\n`

---

## Fase 2 — Client Library (Dogfooding)

**Objetivo:** Client Python que consome a API, usado pelo CLI (Ollama pattern: CLI usa o client lib).

### 2.1 Reescrever `api/client.py`

**DoD:**
- [ ] `ASRClient.transcribe(audio, model, language)` → `TranscribeResponse`
- [ ] `ASRClient.transcribe_stream(audio, model, fn)` → chama fn para cada chunk
- [ ] `ASRClient.pull(model, fn)` → chama fn para cada progress update
- [ ] `ASRClient.list()` → lista de modelos
- [ ] `ASRClient.ps()` → modelos em execução
- [ ] `ASRClient.show(model)` → detalhes do modelo
- [ ] `ASRClient.delete(model)` → remove modelo
- [ ] `ASRClient.version()` → versão
- [ ] Usa `httpx` (async) em vez de `urllib`
- [ ] Streaming: lê NDJSON linha a linha

### 2.2 CLI usa Client

**DoD:**
- [ ] `macaw-asr serve` — inicia server
- [ ] `macaw-asr transcribe file.wav` — usa client para chamar server (ou direto se no-server)
- [ ] `macaw-asr pull model` — usa client quando server ativo, direto quando não
- [ ] `macaw-asr list` — usa client
- [ ] `macaw-asr ps` — usa client
- [ ] `macaw-asr show model` — usa client
- [ ] `macaw-asr rm model` — usa client
- [ ] `macaw-asr version` — local (sem server)

---

## Fase 3 — Testes HTTP (extraídos do Ollama)

**Objetivo:** Cada endpoint testado com request real, validação de response, erros, streaming.

### 3.1 Test Infrastructure (Ollama routes_generate_test.go pattern)

**Arquivos:** `tests/test_api_server.py`

**Padrões do Ollama a extrair:**

**A. Helper `createRequest` → `client.post()`:**
```python
# Ollama usa httptest.NewRecorder + gin.CreateTestContext
# Nós usamos FastAPI TestClient
from fastapi.testclient import TestClient

def test_transcribe():
    client = TestClient(app)
    response = client.post("/api/transcribe", json={
        "model": "qwen",
        "audio": base64.b64encode(pcm_bytes).decode(),
    })
    assert response.status_code == 200
    data = response.json()
    assert "text" in data
    assert data["done"] is True
```

**B. NDJSON streaming test (Ollama bufio.Scanner pattern):**
```python
def test_transcribe_streaming():
    client = TestClient(app)
    with client.stream("POST", "/api/transcribe", json={
        "model": "qwen",
        "audio": base64.b64encode(pcm_bytes).decode(),
        "stream": True,
    }) as response:
        lines = []
        for line in response.iter_lines():
            lines.append(json.loads(line))
        assert lines[-1]["done"] is True
        assert any(l["text"] for l in lines)
```

**C. Error test (Ollama missing body pattern):**
```python
def test_transcribe_missing_body():
    client = TestClient(app)
    response = client.post("/api/transcribe")
    assert response.status_code == 400
    assert "error" in response.json()

def test_transcribe_model_not_found():
    client = TestClient(app)
    response = client.post("/api/transcribe", json={
        "model": "nonexistent",
        "audio": "...",
    })
    assert response.status_code == 404
    assert "not found" in response.json()["error"]
```

**D. Mock runner test (Ollama mockRunner pattern):**
```python
# Override scheduler to return mock engine
def test_transcribe_with_mock():
    mock_engine = MockEngine(fixed_text="teste")
    app.scheduler.get_runner = lambda config: mock_engine

    response = client.post("/api/transcribe", json={...})
    assert response.json()["text"] == "teste"
```

### 3.2 Testes por Endpoint (extraídos dos padrões Ollama)

| Endpoint | Testes |
|----------|--------|
| `POST /api/transcribe` | happy path, streaming, missing body, missing model, missing audio, model not found, empty audio, large audio |
| `POST /api/show` | happy path, model not found, missing model |
| `GET /api/tags` | empty list, with models |
| `GET /api/ps` | empty, with loaded models |
| `POST /api/pull` | happy path (streaming progress), already exists, invalid model |
| `DELETE /api/delete` | exists, not exists |
| `GET /api/version` | returns version string |
| `GET /` | returns "macaw-asr is running" |

### 3.3 Testes com GPU Real

**DoD:**
- [ ] Testes acima rodam com mock engine (sem GPU, rápido)
- [ ] Testes adicionais rodam com GPU real via `MACAW_ASR_TEST_MODEL=qwen`
- [ ] Teste E2E: `macaw-asr serve &` + `macaw-asr transcribe file.wav` + kill server

---

## Fase 4 — Paridade Completa

### 4.1 `keep_alive` Parameter

**Ollama:** `"keep_alive": "5m"` controla quanto tempo o modelo fica em memória.

**DoD:**
- [ ] Request aceita `keep_alive` como string ("5m", "10s", "0" = unload immediately)
- [ ] Scheduler usa o valor para override do TTL per-request
- [ ] `keep_alive: 0` → unload modelo após response

### 4.2 OpenAPI Documentation

**DoD:**
- [ ] FastAPI gera `/docs` (Swagger UI) automaticamente
- [ ] Todos os endpoints documentados com examples
- [ ] `macaw-asr serve` loga URL do docs no startup

### 4.3 CORS e Middleware

**Ollama:** `allowedHostsMiddleware`, `allowedOriginsMiddleware`

**DoD:**
- [ ] CORS habilitado para `*` (default) ou configurável via `MACAW_ASR_ORIGINS`
- [ ] Request logging middleware (method, path, status, duration)

---

## Ordem de Execução

```
Fase 1 (HTTP Server) → Fase 3 (Testes) → Fase 2 (Client) → Fase 4 (Paridade)
```

**Justificativa:**
1. Server primeiro — sem ele nada funciona
2. Testes antes do client — valida que o server está correto
3. Client depois — consome o server testado
4. Paridade por último — refinamentos

---

## Dependências

```
pip install fastapi uvicorn httpx
```

Nenhuma dependência pesada. FastAPI é ~500KB. Uvicorn é o ASGI server.
