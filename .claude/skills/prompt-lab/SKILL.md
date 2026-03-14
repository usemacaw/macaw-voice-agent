---
name: prompt-lab
description: |
  Laboratório de otimização de system prompts.
  Analisa falhas de tool calling e qualidade de voz nos logs,
  propõe variações de prompt, e define plano de validação.
allowed-tools: Read, Grep, Glob, Bash, Agent
user-invocable: true
model: opus
---

# Prompt Lab

Otimize o system prompt do agente de voz para máxima eficácia.

## Fluxo

### 1. Análise do Estado Atual

Ler o prompt atual:
```bash
grep "LLM_SYSTEM_PROMPT" /home/paulo/Projetos/usemacaw/macaw-voice-agent/src/api/.env
```

Coletar evidências de problemas:
```bash
# Falhas de tool calling (texto quando deveria chamar ferramenta)
grep "LLM text (round 0)" /tmp/macaw-api.log | grep "tools=yes" | tail -10

# Respostas sem acentos
grep -i "cotacao\|voce\b\|nao\b\|esta\b\|tambem\|informacao" /tmp/macaw-api.log | grep -v "acentuação\|SEMPRE\|NUNCA" | tail -10

# Respostas longas demais
grep "output_chars" /tmp/macaw-api.log | tail -20

# Tool calls bem-sucedidas (baseline)
grep "tool_call\|Tool executed" /tmp/macaw-api.log | tail -10
```

### 2. Delegar ao agente `prompt-engineer`

Passar:
- Prompt atual completo
- Evidências coletadas dos logs
- Tipo de problema a resolver (tool calling / voz / formato)

### 3. Aplicar mudanças

Se o usuário aprovar a proposta:
1. Atualizar `LLM_SYSTEM_PROMPT` em `src/api/.env`
2. Reiniciar o servidor (`/restart`)
3. Instruir usuário a reconectar e testar
4. Monitorar logs (`/logs`) por 5 interações
5. Rodar `/benchmark` para medir impacto
