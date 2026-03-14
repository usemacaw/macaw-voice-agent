---
name: review
description: |
  Revisa código alterado usando o agente code-reviewer.
  Sem argumento: revisa git diff. Com argumento: revisa arquivo específico.
allowed-tools: Read, Grep, Glob, Bash, Agent
user-invocable: true
model: opus
---

# Code Review

Execute uma revisão de código completa.

## Fluxo

1. **Identificar escopo** da revisão:
   - Sem argumento: `git diff --name-only` para pegar arquivos alterados
   - Com argumento `$ARGUMENTS`: revisar o arquivo ou diretório especificado

2. **Delegar ao agente `code-reviewer`** passando:
   - Lista de arquivos a revisar
   - Contexto: "Revise os seguintes arquivos do macaw-voice-agent focando em performance (latência), correctness (asyncio), error handling, e aderência aos padrões do projeto"

3. **Consolidar resultado**:
   - Agrupar por severidade (Crítico > Warning > Sugestão)
   - Listar ações necessárias antes de merge
   - Se LGTM, diga explicitamente

## Critérios Específicos do Projeto

- **Asyncio**: nunca bloquear event loop, lock scope mínimo
- **Audio buffers**: sem cópias desnecessárias, limites de tamanho
- **Filler**: nunca armazenar no histórico
- **Acentuação**: português correto em strings voltadas ao usuário
- **Testes**: toda lógica nova precisa de teste
