---
paths:
  - "src/api/**/*.py"
  - "src/stt/**/*.py"
  - "src/tts/**/*.py"
  - "src/common/**/*.py"
---

# Regras Python Backend

## Estilo
- Python 3.11+. Use `from __future__ import annotations` em todo arquivo
- Type hints obrigatórios em funções públicas
- `asyncio` para I/O. Nunca bloquear o event loop com operações síncronas
- Logging via `logging.getLogger(__name__)`, nunca `print()`
- Constantes em UPPER_SNAKE_CASE no topo do módulo

## Padrões do Projeto
- Providers implementam ABC (`ASRProvider`, `LLMProvider`, `TTSProvider`) e se registram via `register_*_provider()` no final do módulo
- Configuração via env vars carregadas em `config.py`. Nunca ler `os.environ` diretamente em outros módulos
- Audio interno sempre 8kHz PCM16 mono. API fala 24kHz. Usar `audio/codec.py` para conversão
- Erros de gRPC: `INVALID_ARGUMENT` para input inválido, `INTERNAL` para erros de provider
- Nunca engolir exceções. Se não sabe tratar, deixe subir

## Cuidados Específicos
- System prompt DEVE ter acentuação correta do português (você, não, é, está, cotação)
- Filler phrases NUNCA vão no histórico da conversa (quebra tool calling)
- Emoji deve ser removido antes do TTS (dois paths: session.py e sentence_pipeline.py)
- Qwen3: thinking mode desabilitado (`enable_thinking: False`). `_strip_think()` como safety net
- Após qualquer mudança: reiniciar o servidor e reconectar o browser

## Testes
- pytest-asyncio com `asyncio_mode = "auto"`
- Fakes em `tests/conftest.py`: FakeWebSocket, FakeASR, FakeLLM, FakeTTS
- Sem sleeps. Usar `asyncio.Event` para sincronização
- Padrão AAA (Arrange-Act-Assert)
