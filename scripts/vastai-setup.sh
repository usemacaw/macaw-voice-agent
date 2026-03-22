#!/bin/bash
# =============================================================================
# Vast.ai GPU Setup — STT + TTS + LLM for macaw-voice-agent
# =============================================================================
# Usage: Run this script on the Vast.ai instance after rsync'ing the repo.
#   bash /workspace/macaw-voice-agent/scripts/vastai-setup.sh
#
# Prerequisites:
#   - NVIDIA GPU with CUDA 12.x
#   - Python 3.10+
#   - PyTorch pre-installed (Vast.ai PyTorch image)
# =============================================================================

set -euo pipefail

REPO_DIR="/workspace/macaw-voice-agent"
LOG_DIR="/workspace/logs"

echo "=============================================="
echo "  Macaw Voice Agent — GPU Service Setup"
echo "=============================================="

mkdir -p "$LOG_DIR"

# ---------------------------------------------------------------------------
# 1. System packages
# ---------------------------------------------------------------------------
echo "[1/6] Installing system packages..."
apt-get update -qq && apt-get install -y -qq --no-install-recommends \
    git ffmpeg espeak-ng sox libsndfile1 curl > /dev/null 2>&1
echo "  ✓ System packages installed"

# ---------------------------------------------------------------------------
# 2. vLLM (Qwen3-8B-AWQ)
# ---------------------------------------------------------------------------
echo "[2/6] Installing vLLM..."
pip install -q vllm 2>&1 | tail -1
echo "  ✓ vLLM installed"

# ---------------------------------------------------------------------------
# 3. STT dependencies (Faster-Whisper)
# ---------------------------------------------------------------------------
echo "[3/6] Installing STT dependencies (Faster-Whisper)..."
pip install -q \
    "faster-whisper>=1.1.0" \
    "ctranslate2>=4.0.0" \
    "grpcio>=1.68.0" \
    "grpcio-health-checking>=1.68.0" \
    "grpcio-reflection>=1.68.0" \
    "python-dotenv>=1.0.0" \
    2>&1 | tail -1
echo "  ✓ STT deps installed"

# ---------------------------------------------------------------------------
# 4. TTS dependencies (Macaw Streaming TTS)
# ---------------------------------------------------------------------------
echo "[4/6] Installing TTS dependencies (Macaw Streaming TTS)..."
pip install -q \
    "scipy>=1.10" \
    "transformers>=4.49" \
    "accelerate" \
    2>&1 | tail -1

# Install the macaw-qwen3-tts-streaming package
cd "$REPO_DIR/qwen3-tts-streaming"
pip install -q -e . 2>&1 | tail -1
cd "$REPO_DIR"
echo "  ✓ TTS deps installed"

# ---------------------------------------------------------------------------
# 5. LLM gRPC server dependencies
# ---------------------------------------------------------------------------
echo "[5/6] Installing LLM server dependencies..."
pip install -q \
    "openai>=1.30.0" \
    "httpx>=0.27.0" \
    2>&1 | tail -1
echo "  ✓ LLM server deps installed"

# ---------------------------------------------------------------------------
# 6. Pre-download models
# ---------------------------------------------------------------------------
echo "[6/6] Pre-downloading models (this may take a few minutes)..."

# Whisper large-v3-turbo (STT)
python3 -c "
from faster_whisper import WhisperModel
print('  Downloading Whisper large-v3-turbo...')
WhisperModel('large-v3-turbo', device='cpu', compute_type='int8')
print('  ✓ Whisper model ready')
" 2>&1

# Qwen3-TTS (will be downloaded on first run by transformers)
echo "  ✓ TTS model will download on first run (~2GB)"

# Qwen3-8B-AWQ (vLLM will download on first run)
echo "  ✓ LLM model will download on vLLM first run (~5GB)"

echo ""
echo "=============================================="
echo "  Setup complete! Start services with:"
echo "  bash $REPO_DIR/scripts/vastai-start.sh"
echo "=============================================="
