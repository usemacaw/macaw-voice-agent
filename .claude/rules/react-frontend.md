---
paths:
  - "src/web/**/*.ts"
  - "src/web/**/*.tsx"
  - "src/web/**/*.css"
---

# Regras Frontend React

## Stack
- React 18 + TypeScript 5.6 + Vite 5
- Tailwind CSS v3 (NÃO v4 — Node 18 incompatível)
- lucide-react para ícones
- Sem bibliotecas de áudio — Web Audio API puro

## Padrões
- Hook central: `useRealtimeSession.ts` gerencia todo estado (WebSocket, mensagens, áudio)
- AudioWorklet para capture (24kHz PCM16) e playback. Resampling via interpolação linear
- Labels da UI em português brasileiro (Ouvindo, Respondendo, Pensando)
- Tema dark com cores custom no Tailwind config (bg, surface, accent, etc.)

## Componentes
- `Orb.tsx`: 4 estados de animação (idle, listening, speaking, thinking)
- `TranscriptPanel.tsx`: slide-in da direita com chat bubbles
- `MetricsPanel.tsx`: barras visuais por estágio do pipeline

## Cuidados
- `VITE_WS_URL` define endpoint WebSocket (default: `ws://localhost:8765/v1/realtime`)
- AudioContext precisa de `resume()` no start para Chrome/Safari autoplay policy
- Barge-in: ao receber `speech_started`, limpar fila de playback imediatamente
