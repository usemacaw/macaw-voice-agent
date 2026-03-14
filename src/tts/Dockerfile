# syntax=docker/dockerfile:1.4
# =============================================================================
# Dockerfile TTS Server - Text-to-Speech (CPU/mock)
# =============================================================================
# Build context: src/
#   docker build -f tts/Dockerfile -t tts-server .
# =============================================================================

FROM python:3.11-slim

WORKDIR /app

# Dependencias
COPY tts/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Modulos compartilhados
COPY shared/ ./shared/
COPY common/ ./common/

# TTS providers + server
COPY tts/ ./tts/

# Seguranca: usuario non-root
RUN useradd -r -s /bin/false ttsserver
USER ttsserver

ENV GRPC_HOST=0.0.0.0
ENV GRPC_PORT=50070
ENV LOG_LEVEL=INFO
ENV TTS_PROVIDER=mock
ENV TTS_LANGUAGE=pt
ENV PYTHONPATH=/app

EXPOSE 50070

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import grpc; ch=grpc.insecure_channel('localhost:50070'); grpc.channel_ready_future(ch).result(timeout=3)" || exit 1

CMD ["python", "-m", "tts.server"]
