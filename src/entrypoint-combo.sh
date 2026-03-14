#!/bin/bash
# =============================================================================
# Entrypoint for combo STT+TTS image
# Runs both gRPC servers as background processes, waits for either to exit.
# =============================================================================

set -e

echo "Starting STT server (Whisper) on port 50060..."
GRPC_PORT=50060 python3 -m stt.server &
STT_PID=$!

echo "Starting TTS server (Kokoro) on port 50070..."
GRPC_PORT=50070 python3 -m tts.server &
TTS_PID=$!

echo "Both services started (STT PID=$STT_PID, TTS PID=$TTS_PID)"

# Wait for either process to exit
wait -n $STT_PID $TTS_PID
EXIT_CODE=$?

echo "A service exited with code $EXIT_CODE, shutting down..."
kill $STT_PID $TTS_PID 2>/dev/null || true
wait
exit $EXIT_CODE
