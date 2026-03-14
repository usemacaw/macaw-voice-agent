---
name: health
description: |
  Verifica o status de todos os serviços do sistema.
  Use para diagnosticar problemas de conectividade.
allowed-tools: Bash
user-invocable: true
---

Verifique o status de todos os serviços:

1. **API Server:**
```bash
curl -s http://localhost:8765/health | python3 -m json.tool 2>/dev/null || echo "API Server: OFFLINE"
```

2. **vLLM (via SSH tunnel):**
```bash
curl -s --max-time 5 http://localhost:8100/v1/models | python3 -m json.tool 2>/dev/null || echo "vLLM: OFFLINE (verificar SSH tunnel)"
```

3. **STT gRPC:**
```bash
python3 -c "
import grpc, sys
sys.path.insert(0, '/home/paulo/Projetos/usemacaw/macaw-voice-agent/src/shared')
from grpc_gen import stt_service_pb2_grpc
try:
    ch = grpc.insecure_channel('$(grep ASR_REMOTE_TARGET /home/paulo/Projetos/usemacaw/macaw-voice-agent/src/api/.env | cut -d= -f2)')
    grpc.channel_ready_future(ch).result(timeout=5)
    print('STT: ONLINE')
except:
    print('STT: OFFLINE')
" 2>/dev/null || echo "STT: check failed"
```

4. **TTS gRPC:**
```bash
python3 -c "
import grpc, sys
sys.path.insert(0, '/home/paulo/Projetos/usemacaw/macaw-voice-agent/src/shared')
from grpc_gen import tts_service_pb2_grpc
try:
    ch = grpc.insecure_channel('$(grep TTS_REMOTE_TARGET /home/paulo/Projetos/usemacaw/macaw-voice-agent/src/api/.env | cut -d= -f2)')
    grpc.channel_ready_future(ch).result(timeout=5)
    print('TTS: ONLINE')
except:
    print('TTS: OFFLINE')
" 2>/dev/null || echo "TTS: check failed"
```

5. **Processos:**
```bash
ps aux | grep -E "(python3 main.py|vllm|stt|tts)" | grep -v grep
```

Reporte o status de cada serviço de forma clara e sugira ações corretivas se algum estiver offline.
