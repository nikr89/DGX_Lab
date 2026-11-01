#!/usr/bin/env bash
# Qwen3.8-27B (NVFP4) mit vLLM auf DGX Spark
#   ~/start-qwen3.8-27b.sh        -> startet im Hintergrund, wartet bis API bereit
#   ~/start-qwen3.8-27b.sh -f     -> Vordergrund (Logs live)
#   ~/start-qwen3.8-27b.sh stop   -> stoppen
#   Logs: docker logs -f qwen3.8-27b
set -euo pipefail

MODEL="${MODEL:-unsloth/Qwen3.8-27B-NVFP4}"
SERVED_NAME="${SERVED_NAME:-qwen3.8-27b}"
CONTAINER_NAME="${CONTAINER_NAME:-qwen3.8-27b}"
IMAGE="${IMAGE:-vllm/vllm-openai:latest}"
PORT="${PORT:-8000}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.65}"
KV_CACHE_GB="${KV_CACHE_GB:-40}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-6}"
MAX_BATCHED_TOKENS="${MAX_BATCHED_TOKENS:-16384}"
MTP_TOKENS="${MTP_TOKENS:-1}"
DROP_CACHES="${DROP_CACHES:-1}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
VLLM_CACHE="${VLLM_CACHE:-$HOME/.cache/vllm}"

if [[ "${1:-}" == "stop" ]]; then
  docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 && echo "Gestoppt." || echo "Kein Container '$CONTAINER_NAME'."
  exit 0
fi
FOREGROUND=false
[[ "${1:-}" == "-f" ]] && FOREGROUND=true

MODEL_DIR="$HF_CACHE/hub/models--${MODEL//\//--}"
if [[ -z "$(ls -A "$MODEL_DIR/snapshots" 2>/dev/null)" ]]; then
  echo "FEHLER: Modell nicht im Cache: $MODEL_DIR  (hf download $MODEL)" >&2
  exit 1
fi
echo "Modell gefunden: $MODEL_DIR"
mkdir -p "$VLLM_CACHE"

docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true

if [[ "$DROP_CACHES" == "1" ]]; then
  echo "Leere Page-Cache (sudo) ..."
  sync && sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches' || echo "WARNUNG: drop_caches fehlgeschlagen" >&2
fi
free -h | head -2

if $FOREGROUND; then RUN_MODE=(-it --rm); else RUN_MODE=(-d); fi

docker run "${RUN_MODE[@]}" \
  --name "$CONTAINER_NAME" \
  --gpus all \
  -p "${PORT}:8000" \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e HF_HUB_OFFLINE=1 \
  -e TRANSFORMERS_OFFLINE=1 \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  -v "$VLLM_CACHE:/root/.cache/vllm" \
  "$IMAGE" \
  "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --attention-backend flashinfer \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --kv-cache-memory-bytes "$((KV_CACHE_GB * 1024 * 1024 * 1024))" \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
  --enable-chunked-prefill \
  --async-scheduling \
  --enable-prefix-caching \
  --speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":${MTP_TOKENS}}" \
  --load-format fastsafetensors \
  --generation-config auto \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice

$FOREGROUND && exit 0

echo "Container läuft im Hintergrund. Warte auf API (erster Start ca. 5 Min.) ..."
echo "Logs live ansehen:  docker logs -f $CONTAINER_NAME"
for i in $(seq 1 180); do
  if curl -sf "http://localhost:${PORT}/health" >/dev/null; then
    echo "Bereit: http://localhost:${PORT}/v1  (Modell: $SERVED_NAME)"
    exit 0
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$CONTAINER_NAME"; then
    echo "Container ist abgestürzt:" >&2
    docker logs --tail 50 "$CONTAINER_NAME" >&2 || true
    exit 1
  fi
  sleep 5
done
echo "Timeout – docker logs -f $CONTAINER_NAME" >&2
exit 1
