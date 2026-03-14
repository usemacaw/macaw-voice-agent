---
name: restart
description: |
  Reinicia o API server do macaw-voice-agent.
  Use quando fizer mudanças no backend Python.
allowed-tools: Bash
user-invocable: true
---

Reinicie o API server seguindo estes passos:

1. Mate o processo atual:
```bash
pkill -f "python3 main.py" 2>/dev/null || true
```

2. Aguarde 1 segundo e reinicie:
```bash
cd /home/paulo/Projetos/usemacaw/macaw-voice-agent/src/api && nohup python3 main.py > /tmp/macaw-api.log 2>&1 &
```

3. Aguarde 2 segundos e verifique que está rodando:
```bash
sleep 2 && curl -s http://localhost:8765/health | python3 -m json.tool
```

4. Informe ao usuário para **reconectar o browser** (F5) para pegar o novo código.
