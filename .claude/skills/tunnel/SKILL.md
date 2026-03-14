---
name: tunnel
description: |
  Gerencia SSH tunnels para serviços remotos (vLLM na Vast.ai).
  Use para conectar, reconectar ou verificar status dos tunnels.
allowed-tools: Bash
user-invocable: true
---

Gerencie os SSH tunnels para a Vast.ai.

**Sem argumento ou "status":** verificar tunnels ativos:
```bash
ps aux | grep "ssh -N" | grep -v grep
lsof -i :8100 2>/dev/null | head -5
```

**"connect" ou "up":** conectar tunnel do vLLM:
```bash
# Verifica se já existe
if lsof -i :8100 >/dev/null 2>&1; then
    echo "Tunnel vLLM já está ativo na porta 8100"
else
    echo "Iniciando tunnel vLLM..."
    echo "Execute manualmente (requer senha/key):"
    echo "ssh -N -L 8100:localhost:8000 -p 21068 root@ssh5.vast.ai &"
fi
```

**"kill" ou "down":** desconectar tunnels:
```bash
pkill -f "ssh -N -L 8100" 2>/dev/null && echo "Tunnel vLLM desconectado" || echo "Nenhum tunnel ativo"
```

Reporte o status atual e instrua o usuário se precisar de ação manual.
