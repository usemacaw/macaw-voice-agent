# Web Client

Browser voice chat client for OpenVoiceAPI. React + TypeScript + Vite.

## Commands

```bash
npm install        # Install dependencies
npm run dev        # Dev server (http://localhost:5173)
npm run build      # Production build → dist/
npm run lint       # ESLint check
```

## Architecture

- **Single hook:** `useRealtimeSession.ts` owns all state — WebSocket, messages, audio capture/playback
- **AudioWorklet processors:** `capture-processor.js` (mic→24kHz PCM16) and `playback-processor.js` (PCM16→speaker). Both resample if browser rate differs from 24kHz
- **No audio libraries:** Pure Web Audio API. Resampling via linear interpolation in worklet processors
- **Protocol:** OpenAI Realtime API events over WebSocket. Audio as base64-encoded PCM16 24kHz

## Key Points

- Audio format: PCM16 24kHz (matches OpenAI Realtime API spec)
- Chunk size: 20ms (480 samples @ 24kHz = 960 bytes)
- Barge-in: on `speech_started`, playback queue cleared immediately
- Metrics: logged every 5s (chunks sent, bytes, events received)
- UI: PT-BR labels, dark theme, chat bubbles, speaking pulse animation
- Config: `VITE_WS_URL` env var for WebSocket endpoint
- Session config hardcoded in hook (server VAD, 500ms silence, auto-response)
