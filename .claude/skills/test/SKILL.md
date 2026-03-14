---
name: test
description: |
  Roda os testes do projeto. Sem argumentos roda todos.
  Com argumento, filtra por nome do teste ou arquivo.
allowed-tools: Bash
user-invocable: true
---

Rode os testes do projeto. Use o argumento para filtrar:

- Sem argumento: roda todos os testes
- Com arquivo: `pytest tests/<arquivo>.py -v`
- Com padrão: `pytest -k "<padrão>" -v`

```bash
cd /home/paulo/Projetos/usemacaw/macaw-voice-agent/src/api

if [ -z "$ARGUMENTS" ]; then
    pytest -v --tb=short
elif [[ "$ARGUMENTS" == test_* ]] || [[ "$ARGUMENTS" == *".py" ]]; then
    pytest "tests/$ARGUMENTS" -v --tb=short 2>/dev/null || pytest "$ARGUMENTS" -v --tb=short
else
    pytest -k "$ARGUMENTS" -v --tb=short
fi
```

Após rodar, analise os resultados:
- Se houver falhas, identifique a causa raiz
- Mostre um resumo: total, passou, falhou
- Se relevante, sugira correções
