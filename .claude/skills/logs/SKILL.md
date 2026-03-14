---
name: logs
description: |
  Mostra os logs recentes do API server.
  Use para debugar problemas ou verificar comportamento.
allowed-tools: Bash
user-invocable: true
---

Mostre os logs do API server. O argumento é opcional e controla quantas linhas mostrar.

```bash
tail -n ${0:-50} /tmp/macaw-api.log
```

Se o usuário passar um termo de busca como argumento, filtre os logs:

```bash
grep -i "$ARGUMENTS" /tmp/macaw-api.log | tail -30
```

Analise os logs e reporte:
- Erros ou warnings relevantes
- Se tools estão sendo chamadas corretamente
- Se há "LLM text (round 0)" com tools=yes (indica falha de tool calling)
- Latências anormais nos estágios do pipeline
