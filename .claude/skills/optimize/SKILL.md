---
name: optimize
description: |
  Analisa e otimiza um módulo ou o pipeline inteiro para latência.
  Sem argumento: analisa logs + métricas. Com argumento: analisa arquivo específico.
allowed-tools: Read, Grep, Glob, Bash, Agent
user-invocable: true
model: opus
---

# Otimizar Latência

Analise e proponha otimizações de latência para o pipeline voice-to-voice.

## Fluxo

1. **Coletar dados**:
   ```bash
   # Métricas recentes
   grep "macaw.metrics\|e2e_ms\|llm_ttft\|pipeline_first_audio\|asr_ms\|tts_synth" /tmp/macaw-api.log | tail -30

   # Sessões com latência alta
   grep "e2e_ms" /tmp/macaw-api.log | tail -20
   ```

2. **Se argumento `$ARGUMENTS` fornecido**: focar análise no arquivo/módulo especificado

3. **Se sem argumento**: delegar ao agente `latency-hunter` para análise completa do pipeline

4. **Para cada gargalo identificado**:
   - Propor fix concreto com diff
   - Estimar impacto em ms
   - Avaliar risco de regressão
   - Implementar se aprovado pelo usuário

## Targets

| Métrica | Target SOTA | Aceitável |
|---------|------------|-----------|
| E2E | <500ms | <800ms |
| ASR | <100ms | <200ms |
| LLM TTFT | <150ms | <300ms |
| TTS TTFB | <100ms | <200ms |
| 1st Audio | <400ms | <600ms |
