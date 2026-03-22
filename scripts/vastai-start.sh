#!/bin/bash
# =============================================================================
# Start all GPU services for macaw-voice-agent on Vast.ai
# =============================================================================
# Usage: bash /workspace/macaw-voice-agent/scripts/vastai-start.sh
#
# Services:
#   1. vLLM (port 8000) — Qwen2.5-7B-Instruct-AWQ
#   2. STT  (port 50060) — Qwen3-ASR-0.6B (gRPC)
#   3. TTS  (port 50070) — Macaw Streaming TTS / Qwen3-TTS-0.6B (gRPC)
#   4. LLM  (port 50080) — gRPC bridge to vLLM REST API
# =============================================================================

set -euo pipefail

REPO_DIR="/workspace/macaw-voice-agent"
SRC_DIR="$REPO_DIR/src"
LOG_DIR="/workspace/logs"

mkdir -p "$LOG_DIR"

echo "=============================================="
echo "  Starting Macaw Voice Agent GPU Services"
echo "=============================================="

# ---------------------------------------------------------------------------
# 1. vLLM — port 8000
# ---------------------------------------------------------------------------
echo "[1/4] Starting vLLM (Qwen2.5-7B-Instruct-AWQ)..."
if pgrep -f "vllm.entrypoints" > /dev/null 2>&1; then
    echo "  ⚠ vLLM already running"
else
    nohup python3 -m vllm.entrypoints.openai.api_server \
        --model Qwen/Qwen2.5-7B-Instruct-AWQ \
        --quantization awq \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.45 \
        --port 8000 \
        --host 0.0.0.0 \
        --trust-remote-code \
        --enable-auto-tool-choice \
        --tool-call-parser hermes \
        > "$LOG_DIR/vllm.log" 2>&1 &
    echo "  ✓ vLLM starting (log: $LOG_DIR/vllm.log)"
fi

echo "  Waiting for vLLM model load..."
for i in $(seq 1 180); do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "  ✓ vLLM ready (${i}s)"
        break
    fi
    [ "$i" -eq 180 ] && { echo "  ✗ vLLM timeout"; tail -10 "$LOG_DIR/vllm.log"; exit 1; }
    sleep 1
done

# ---------------------------------------------------------------------------
# 2. STT — port 50060
# ---------------------------------------------------------------------------
echo "[2/4] Starting STT server (Qwen3-ASR)..."
if pgrep -f "stt.server" > /dev/null 2>&1; then
    echo "  ⚠ STT already running"
else
    cd "$SRC_DIR"
    STT_PROVIDER=qwen \
    STT_LANGUAGE=pt \
    QWEN_STT_MODEL=Qwen/Qwen3-ASR-0.6B \
    QWEN_DEVICE=cuda \
    GRPC_PORT=50060 \
    LOG_LEVEL=INFO \
    PYTHONPATH="$SRC_DIR" \
    nohup python3 -m stt.server > "$LOG_DIR/stt.log" 2>&1 &
    echo "  ✓ STT starting (log: $LOG_DIR/stt.log)"
fi

# ---------------------------------------------------------------------------
# 3. TTS — port 50070
# ---------------------------------------------------------------------------
echo "[3/4] Starting TTS server (Macaw Streaming TTS)..."
if pgrep -f "tts.server" > /dev/null 2>&1; then
    echo "  ⚠ TTS already running"
else
    cd "$SRC_DIR"
    TTS_PROVIDER=macaw-streaming \
    TTS_LANGUAGE=pt \
    QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
    QWEN_TTS_SPEAKER=Ryan \
    QWEN_TTS_LANGUAGE=Portuguese \
    GRPC_PORT=50070 \
    LOG_LEVEL=INFO \
    PYTHONPATH="$SRC_DIR" \
    nohup python3 -m tts.server > "$LOG_DIR/tts.log" 2>&1 &
    echo "  ✓ TTS starting (log: $LOG_DIR/tts.log)"
fi

# ---------------------------------------------------------------------------
# 4. LLM gRPC — port 50080
# ---------------------------------------------------------------------------
echo "[4/4] Starting LLM gRPC server..."
if pgrep -f "llm.server" > /dev/null 2>&1; then
    echo "  ⚠ LLM gRPC already running"
else
    cd "$SRC_DIR"
    LLM_BACKEND_PROVIDER=vllm \
    VLLM_BASE_URL=http://localhost:8000/v1 \
    LLM_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ \
    LLM_TIMEOUT=30.0 \
    GRPC_PORT=50080 \
    LOG_LEVEL=INFO \
    PYTHONPATH="$SRC_DIR" \
    nohup python3 -m llm.server > "$LOG_DIR/llm.log" 2>&1 &
    echo "  ✓ LLM gRPC starting (log: $LOG_DIR/llm.log)"
fi

# Wait for gRPC services
echo ""
echo "Waiting for gRPC services..."
sleep 15

echo ""
echo "=== Service Status ==="
echo "vLLM:     $(curl -s http://localhost:8000/health > /dev/null 2>&1 && echo 'OK' || echo 'FAIL')"
echo "STT:      $(python3 -c 'import grpc; ch=grpc.insecure_channel("localhost:50060"); grpc.channel_ready_future(ch).result(timeout=5); print("OK")' 2>&1)"
echo "TTS:      $(python3 -c 'import grpc; ch=grpc.insecure_channel("localhost:50070"); grpc.channel_ready_future(ch).result(timeout=5); print("OK")' 2>&1)"
echo "LLM gRPC: $(python3 -c 'import grpc; ch=grpc.insecure_channel("localhost:50080"); grpc.channel_ready_future(ch).result(timeout=5); print("OK")' 2>&1)"
echo ""
echo "GPU: $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader)"
echo ""
echo "=============================================="
echo "  SSH tunnels (run on LOCAL machine):"
echo "  ssh -N \\"
echo "    -L 50060:localhost:50060 \\"
echo "    -L 50070:localhost:50070 \\"
echo "    -L 50080:localhost:50080 \\"
echo "    -p <SSH_PORT> root@<SSH_HOST>"
echo ""
echo "  Then update .env:"
echo "    ASR_PROVIDER=remote"
echo "    ASR_REMOTE_TARGET=localhost:50060"
echo "    TTS_PROVIDER=remote"
echo "    TTS_REMOTE_TARGET=localhost:50070"
echo "    LLM_PROVIDER=remote"
echo "    LLM_REMOTE_TARGET=localhost:50080"
echo "=============================================="
