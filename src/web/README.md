# Web Client

Browser-based voice chat client for OpenVoiceAPI. Real-time bidirectional audio via WebSocket using the OpenAI Realtime API protocol.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Browser                           │
│                                                      │
│  ┌───────────────────────────────────────────────┐  │
│  │  App.tsx                                      │  │
│  │  • Status indicator (disconnected/connected)  │  │
│  │  • Chat messages (user + assistant)           │  │
│  │  • Speaking indicators with pulse animation   │  │
│  └───────────────────┬───────────────────────────┘  │
│                      │                               │
│            useRealtimeSession hook                   │
│                      │                               │
│         ┌────────────┼────────────┐                  │
│         ▼            ▼            ▼                  │
│  ┌────────────┐ ┌─────────┐ ┌──────────────┐       │
│  │  Capture   │ │   WS    │ │   Playback   │       │
│  │ (mic→PCM)  │ │ (JSON)  │ │ (PCM→speaker)│       │
│  └─────┬──────┘ └────┬────┘ └──────┬───────┘       │
│        │              │             │                │
│  AudioWorklet    WebSocket    AudioWorklet           │
│  (24kHz PCM16)   (events)    (24kHz PCM16)          │
└────────┼──────────────┼─────────────┼────────────────┘
         │              │             │
         └──────────────┼─────────────┘
                        ▼
               OpenVoiceAPI :8765
```

## Quick Start

```bash
# Install
npm install

# Dev server (hot reload)
npm run dev

# Build
npm run build
```

Open http://localhost:5173 and click "Conectar".

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `VITE_WS_URL` | `ws://localhost:8765/v1/realtime` | WebSocket endpoint |

Set in `.env` or `.env.local`:
```
VITE_WS_URL=ws://your-server:8765/v1/realtime
```

## How It Works

### Connection Flow

1. User clicks "Conectar"
2. `AudioCapture` + `AudioPlayback` initialized (AudioWorklet processors)
3. WebSocket opens to `VITE_WS_URL`
4. On connect: sends `session.update` with server VAD config
5. Mic capture starts — 20ms PCM16 chunks (24kHz) sent as base64
6. Server VAD detects speech → triggers ASR → LLM → TTS pipeline
7. Response audio arrives as `response.audio.delta` events (base64 PCM16)
8. Audio decoded and played through speaker

### Barge-in (Interruption)

When server detects new speech during assistant playback:
1. `input_audio_buffer.speech_started` event received
2. Playback queue cleared immediately
3. New user speech processed normally

### Audio Pipeline

| Stage | Format | Rate | Processing |
|-------|--------|------|------------|
| Microphone | Float32 | Browser native | Hardware capture |
| Capture Processor | Int16 PCM | 24kHz | Resample + quantize |
| WebSocket | Base64 string | 24kHz | Encode for JSON |
| Playback (receive) | Base64 string | 24kHz | Decode from JSON |
| Playback Processor | Float32 | Context native | Resample + convert |
| Speaker | Float32 | Browser native | Hardware output |

Both AudioWorklet processors handle resampling if browser sample rate differs from 24kHz (linear interpolation).

## Session Configuration

Hardcoded in `useRealtimeSession.ts`:

```json
{
  "modalities": ["text", "audio"],
  "input_audio_format": "pcm16",
  "output_audio_format": "pcm16",
  "turn_detection": {
    "type": "server_vad",
    "silence_duration_ms": 500,
    "create_response": true,
    "interrupt_response": true
  }
}
```

## Project Structure

```
web-client/
├── src/
│   ├── App.tsx                      # Main UI component
│   ├── App.css                      # Dark theme styling
│   ├── types.ts                     # Message, SessionStatus
│   ├── hooks/
│   │   └── useRealtimeSession.ts    # WebSocket + audio orchestration
│   └── audio/
│       ├── capture.ts               # Mic → AudioWorklet → PCM chunks
│       └── playback.ts              # PCM chunks → AudioWorklet → speaker
├── public/
│   ├── capture-processor.js         # AudioWorklet: resample + F32→I16
│   └── playback-processor.js        # AudioWorklet: I16→F32 + resample
├── package.json
└── vite.config.ts
```

## Tech Stack

- React 18.3 + TypeScript 5.6
- Vite 5.4 (dev server + build)
- Web Audio API (AudioWorklet for low-latency processing)
- WebSocket (native browser API)
- No external audio libraries
