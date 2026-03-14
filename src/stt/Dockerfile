# syntax=docker/dockerfile:1.4
# =============================================================================
# Dockerfile STT Server - Speech-to-Text (CPU/mock)
# =============================================================================
# Build context: src/
#   docker build -f stt/Dockerfile -t stt-server .
# =============================================================================

FROM python:3.11-slim

WORKDIR /app

# Dependencias
COPY stt/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt && rm /tmp/requirements.txt

# Modulos compartilhados
COPY shared/ ./shared/
COPY common/ ./common/

# STT providers + server
COPY stt/ ./stt/

# Seguranca: usuario non-root
RUN useradd -r -s /bin/false sttserver
USER sttserver

ENV GRPC_HOST=0.0.0.0
ENV GRPC_PORT=50060
ENV LOG_LEVEL=INFO
ENV STT_PROVIDER=mock
ENV STT_LANGUAGE=pt
ENV PYTHONPATH=/app

EXPOSE 50060

HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD python -c "import grpc; ch=grpc.insecure_channel('localhost:50060'); grpc.channel_ready_future(ch).result(timeout=3)" || exit 1

CMD ["python", "-m", "stt.server"]
