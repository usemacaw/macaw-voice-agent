---
name: audit
description: |
  Auditoria completa do sistema: performance, segurança, testes,
  arquitetura, e observabilidade. Gera relatório com ações priorizadas.
allowed-tools: Read, Grep, Glob, Bash, Agent
user-invocable: true
model: opus
---

# Auditoria do Sistema

Execute uma auditoria completa do macaw-voice-agent em todas as dimensões.

## Fluxo

Delegue a análise para os agentes especializados em paralelo:

### 1. Performance (agente: latency-hunter)
- Analisar logs de latência
- Identificar gargalos no pipeline
- Comparar com targets SOTA

### 2. Código (agente: code-reviewer)
- Revisar os 5 arquivos mais críticos:
  - `server/session.py`
  - `pipeline/sentence_pipeline.py`
  - `providers/llm.py`
  - `audio/vad.py`
  - `protocol/event_emitter.py`

### 3. Testes (agente: test-builder)
- Mapear cobertura atual
- Identificar gaps críticos
- Listar testes faltantes por prioridade

### 4. Arquitetura (agente: architect)
- Avaliar decisões arquiteturais
- Identificar acoplamentos problemáticos
- Propor evolução

### 5. Prompt (agente: prompt-engineer)
- Avaliar system prompt atual
- Verificar taxa de falha de tool calling
- Propor melhorias

## Checklist de Segurança
- [ ] Rate limiting funcional (per-session e global)
- [ ] Input validation em conversation items
- [ ] System role injection bloqueado
- [ ] Auth no WebSocket (REALTIME_API_KEY)
- [ ] Origin validation (WS_ALLOWED_ORIGINS)
- [ ] Secrets não expostos em logs
- [ ] Tool arguments sanitizados

## Formato de Output

```
# Auditoria — Macaw Voice Agent — [data]

## Score Geral: X/10

| Dimensão | Score | Status |
|----------|-------|--------|
| Performance | X/10 | [emoji] |
| Código | X/10 | [emoji] |
| Testes | X/10 | [emoji] |
| Arquitetura | X/10 | [emoji] |
| Segurança | X/10 | [emoji] |
| Observabilidade | X/10 | [emoji] |
| Prompt | X/10 | [emoji] |

## Top 5 Ações (por impacto)
1. [Ação] — [dimensão] — [impacto] — [esforço]
2. ...

## Detalhes por Dimensão
[Resultados de cada agente]
```
