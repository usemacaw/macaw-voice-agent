#!/usr/bin/env bash
# =============================================================================
# Deploy STT + TTS na Vast.ai para uso com OpenVoiceAPI local
# =============================================================================
# Provisiona 2 GPUs na Vast.ai (STT + TTS) e gera .env configurado.
#
# Uso:
#   ./open-voice-api/scripts/deploy_vastai.sh              # Provisiona e configura
#   ./open-voice-api/scripts/deploy_vastai.sh --status     # Status das instancias
#   ./open-voice-api/scripts/deploy_vastai.sh --destroy    # Destroi instancias
#
# Pre-requisitos:
#   pip install vastai
#   vastai set api-key <KEY>
#   Docker images pushadas: paulohenriquevn/stt-server-gpu, paulohenriquevn/tts-server-gpu
# =============================================================================

set -euo pipefail

DOCKER_USER="${DOCKER_USER:-paulohenriquevn}"
IMAGE_STT="${DOCKER_USER}/stt-server-gpu:latest"
IMAGE_TTS="${DOCKER_USER}/tts-server-gpu:latest"

# GPU: precisa de >=20GB VRAM para Qwen3 1.7B
GPU_QUERY="gpu_ram>=20 num_gpus=1 inet_up>=200 disk_space>=50 reliability>=0.95"

LABEL_PREFIX="ova-split"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
OVA_DIR="$REPO_ROOT/open-voice-api"

log() { echo "[$(date '+%H:%M:%S')] $*" >&2; }

check_deps() {
    if ! command -v vastai &>/dev/null; then
        echo "ERRO: vastai CLI nao encontrado. Install: pip install vastai"
        exit 1
    fi
}

create_instance() {
    local name="$1" image="$2" env_vars="$3" onstart_cmd="$4"

    log "Buscando oferta para $name..."
    local offer_id
    offer_id=$(vastai search offers "$GPU_QUERY" --order "dph_total" --limit 1 --raw | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(data[0]['id'] if data else '')
")

    if [ -z "$offer_id" ]; then
        echo "ERRO: Nenhuma oferta GPU encontrada"
        exit 1
    fi

    log "Oferta: $offer_id — criando $name..."
    local instance_id
    instance_id=$(vastai create instance "$offer_id" \
        --image "$image" \
        --label "${LABEL_PREFIX}-${name}" \
        --env "$env_vars" \
        --disk 50 \
        --onstart-cmd "$onstart_cmd" \
        --raw | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(data.get('new_contract', ''))
")

    if [ -z "$instance_id" ]; then
        echo "ERRO: Falha ao criar $name"
        exit 1
    fi

    log "$name criado: ID $instance_id"
    echo "$instance_id"
}

wait_for_instance() {
    local instance_id="$1" name="$2" max_wait=600 elapsed=0

    log "Aguardando $name (ID: $instance_id)..."
    while [ $elapsed -lt $max_wait ]; do
        local status
        status=$(vastai show instance "$instance_id" --raw | python3 -c "
import sys, json
data = json.load(sys.stdin)
print(data.get('actual_status', 'unknown'))
")
        if [ "$status" = "running" ]; then
            log "$name rodando!"
            return 0
        fi
        sleep 10
        elapsed=$((elapsed + 10))
        log "  $name: $status (${elapsed}s/${max_wait}s)"
    done

    echo "ERRO: Timeout esperando $name"
    return 1
}

get_endpoint() {
    local instance_id="$1" internal_port="$2"
    vastai show instance "$instance_id" --raw | python3 -c "
import sys, json
data = json.load(sys.stdin)
ip = data.get('public_ipaddr', '')
ports = data.get('ports', {})
key = '${internal_port}/tcp'
if key in ports:
    mapped = ports[key]
    if isinstance(mapped, list) and mapped:
        ext = mapped[0].get('HostPort', '${internal_port}')
    elif isinstance(mapped, dict):
        ext = mapped.get('HostPort', '${internal_port}')
    else:
        ext = '${internal_port}'
else:
    ext = '${internal_port}'
print(f'{ip}:{ext}')
"
}

show_status() {
    log "Status das instancias ${LABEL_PREFIX}-*:"
    vastai show instances --raw | python3 -c "
import sys, json
data = json.load(sys.stdin)
found = False
for inst in data:
    label = inst.get('label', '')
    if label.startswith('${LABEL_PREFIX}'):
        found = True
        status = inst.get('actual_status', '?')
        ip = inst.get('public_ipaddr', 'N/A')
        ports = inst.get('ports', {})
        gpu = inst.get('gpu_name', 'N/A')
        cost = inst.get('dph_total', 0)
        print(f'  {label}: status={status}, ip={ip}, gpu={gpu}, \${cost:.3f}/hr')
        for k, v in ports.items():
            port_str = v[0]['HostPort'] if isinstance(v, list) and v else str(v)
            print(f'    {k} -> {port_str}')
if not found:
    print('  Nenhuma instancia encontrada')
"
}

destroy_all() {
    log "Destruindo instancias ${LABEL_PREFIX}-*..."
    local instances
    instances=$(vastai show instances --raw | python3 -c "
import sys, json
data = json.load(sys.stdin)
for inst in data:
    if inst.get('label', '').startswith('${LABEL_PREFIX}'):
        print(inst['id'])
")
    if [ -z "$instances" ]; then
        log "Nenhuma instancia encontrada"
        return 0
    fi
    for id in $instances; do
        log "Destruindo $id..."
        vastai destroy instance "$id"
    done
    log "Instancias destruidas"
}

generate_env() {
    local stt_endpoint="$1" tts_endpoint="$2"

    cat > "$OVA_DIR/.env" << EOF
# Generated by deploy_vastai.sh at $(date)
# STT/TTS on Vast.ai, LLM via Anthropic API, OpenVoiceAPI local

ASR_PROVIDER=remote
ASR_REMOTE_TARGET=${stt_endpoint}
ASR_REMOTE_TIMEOUT=30.0
ASR_REMOTE_STREAMING=true
ASR_LANGUAGE=pt

TTS_PROVIDER=remote
TTS_REMOTE_TARGET=${tts_endpoint}
TTS_REMOTE_TIMEOUT=60.0
TTS_LANGUAGE=pt
TTS_VOICE=alloy

LLM_PROVIDER=anthropic
LLM_MODEL=claude-haiku-4-5-20251001
ANTHROPIC_API_KEY=$(grep ANTHROPIC_API_KEY "$REPO_ROOT/ai-agent/.env" 2>/dev/null | cut -d= -f2-)
LLM_SYSTEM_PROMPT=You are a helpful voice assistant. Keep responses concise and natural for spoken conversation.
LLM_MAX_TOKENS=1024
LLM_TEMPERATURE=0.8

LOG_LEVEL=INFO
EOF

    log ".env gerado em $OVA_DIR/.env"
}

# ---------------------------------------------------------------------------
main() {
    check_deps

    if [ "${1:-}" = "--status" ]; then
        show_status
        exit 0
    fi

    if [ "${1:-}" = "--destroy" ]; then
        destroy_all
        exit 0
    fi

    log "=== Deploy STT + TTS na Vast.ai ==="
    log "OpenVoiceAPI roda local, STT/TTS em GPUs remotas"
    log ""

    # Provisiona STT
    STT_ID=$(create_instance "stt" "$IMAGE_STT" \
        '-e STT_PROVIDER=qwen-streaming -e QWEN_STT_MODEL=Qwen/Qwen3-ASR-1.7B -e QWEN_STT_GPU_MEM_UTIL=0.80 -e QWEN_STT_ENFORCE_EAGER=true -e GRPC_PORT=50060 -e LOG_LEVEL=INFO -p 50060:50060' \
        "python3 server.py")

    # Provisiona TTS
    TTS_ID=$(create_instance "tts" "$IMAGE_TTS" \
        '-e TTS_PROVIDER=faster -e QWEN_TTS_MODEL=Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice -e QWEN_TTS_SPEAKER=aiden -e FASTER_TTS_CHUNK_SIZE=4 -e GRPC_PORT=50070 -e LOG_LEVEL=INFO -p 50070:50070' \
        "python3 server.py")

    # Espera ficarem prontas
    wait_for_instance "$STT_ID" "stt-server"
    wait_for_instance "$TTS_ID" "tts-server"

    # Pega endpoints
    STT_ENDPOINT=$(get_endpoint "$STT_ID" "50060")
    TTS_ENDPOINT=$(get_endpoint "$TTS_ID" "50070")

    log ""
    log "STT: $STT_ENDPOINT"
    log "TTS: $TTS_ENDPOINT"

    # Gera .env
    generate_env "$STT_ENDPOINT" "$TTS_ENDPOINT"

    log ""
    log "============================================"
    log " Deploy completo!"
    log "============================================"
    log ""
    log " STT Server:  $STT_ENDPOINT  (ID: $STT_ID)"
    log " TTS Server:  $TTS_ENDPOINT  (ID: $TTS_ID)"
    log ""
    log " Para rodar o OpenVoiceAPI local:"
    log ""
    log "   cd $REPO_ROOT"
    log "   PYTHONPATH=open-voice-api:shared python3 open-voice-api/main.py"
    log ""
    log " Para testar (text-only, sem ASR/TTS):"
    log "   PYTHONPATH=open-voice-api:shared python3 open-voice-api/scripts/test_e2e.py --text-only"
    log ""
    log " Para testar (full audio):"
    log "   PYTHONPATH=open-voice-api:shared python3 open-voice-api/scripts/test_e2e.py"
    log ""
    log " Status:   $0 --status"
    log " Destruir: $0 --destroy"
    log "============================================"
}

main "$@"
