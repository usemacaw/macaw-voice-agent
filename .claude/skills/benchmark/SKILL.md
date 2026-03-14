---
name: benchmark
description: |
  Executa benchmarks de latência nos estágios do pipeline.
  Mede ASR, LLM, TTS e E2E com dados reais dos logs.
allowed-tools: Bash, Read, Grep
user-invocable: true
---

# Benchmark de Latência

Coleta e analisa métricas de latência do pipeline voice-to-voice.

## Execução

### 1. Coletar métricas dos logs

```bash
echo "=== Últimas 20 respostas ==="
grep "macaw.metrics" /tmp/macaw-api.log | tail -20 | python3 -c "
import sys, json, re

metrics = []
for line in sys.stdin:
    # Extract JSON from log line
    match = re.search(r'macaw\.metrics.*?({.*})', line)
    if not match:
        continue
    try:
        m = json.loads(match.group(1))
        metrics.append(m)
    except:
        continue

if not metrics:
    print('Nenhuma métrica encontrada nos logs.')
    sys.exit(0)

# Headers
fields = ['turn', 'asr_ms', 'llm_ttft_ms', 'llm_total_ms', 'llm_first_sentence_ms',
          'tts_synth_ms', 'pipeline_first_audio_ms', 'e2e_ms', 'total_ms']
print(f\"{'#':>3} {'ASR':>7} {'TTFT':>7} {'LLM':>7} {'1stSent':>7} {'TTS':>7} {'1stAud':>7} {'E2E':>7} {'Total':>7}\")
print('-' * 70)

for m in metrics:
    vals = [m.get('turn', '?')]
    for f in fields[1:]:
        v = m.get(f)
        vals.append(f'{v:.0f}' if v else '-')
    print(f\"{vals[0]:>3} {vals[1]:>7} {vals[2]:>7} {vals[3]:>7} {vals[4]:>7} {vals[5]:>7} {vals[6]:>7} {vals[7]:>7} {vals[8]:>7}\")

# Summary
print()
print('=== Resumo ===')
for f in fields[1:]:
    values = [m.get(f, 0) for m in metrics if m.get(f)]
    if values:
        avg = sum(values) / len(values)
        mn = min(values)
        mx = max(values)
        p95 = sorted(values)[int(len(values) * 0.95)] if len(values) >= 2 else mx
        print(f'{f:>25}: avg={avg:>7.0f}ms  min={mn:>7.0f}ms  p95={p95:>7.0f}ms  max={mx:>7.0f}ms  (n={len(values)})')

# Tool timings
tool_data = [t for m in metrics for t in m.get('tool_timings', [])]
if tool_data:
    print()
    print('=== Tools ===')
    from collections import defaultdict
    by_name = defaultdict(list)
    for t in tool_data:
        by_name[t['name']].append(t['exec_ms'])
    for name, times in by_name.items():
        avg = sum(times) / len(times)
        print(f'{name:>25}: avg={avg:.0f}ms  min={min(times):.0f}ms  max={max(times):.0f}ms  (n={len(times)}, ok={sum(1 for t in tool_data if t[\"name\"]==name and t[\"ok\"])}/{len(times)})')
" 2>/dev/null || echo "Erro ao processar métricas. Verifique se o servidor está rodando e gerando logs."
```

### 2. Health check dos serviços

```bash
echo "=== Health ==="
curl -s http://localhost:8765/health 2>/dev/null | python3 -m json.tool || echo "API: OFFLINE"
curl -s --max-time 3 http://localhost:8100/v1/models 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(f'vLLM: {d[\"data\"][0][\"id\"]}')" 2>/dev/null || echo "vLLM: OFFLINE"
```

### 3. Análise

Compare com targets SOTA e identifique:
- Estágios acima do target (highlight em vermelho conceitual)
- Tendências (piorando ou melhorando?)
- Outliers (respostas com latência anormal)
- Tool calls lentas
