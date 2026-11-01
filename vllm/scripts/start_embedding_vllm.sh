#!/usr/bin/env bash
# ============================================================
# Qwen3-Embedding-4B via vLLM — Embedding-Server für WissensVault
# Läuft parallel zum bestehenden LLM-Server (Port 8000) auf Port 8003.
# Berührt den bestehenden Qwen-LLM-Container NICHT.
#
# Start:   ./start_embedding_vllm.sh
# Stop:    ./start_embedding_vllm.sh stop
# Status:  ./start_embedding_vllm.sh status
# Test:    curl -s localhost:8003/v1/embeddings -d '{"model":"qwen3-embed","input":"Hallo"}'
# ============================================================
set -euo pipefail

IMAGE="vllm/vllm-openai:latest"
MODEL_REPO="Qwen/Qwen3-Embedding-4B"
CONTAINER="qwen3-embed"
PORT=8003
# Konservativ: 27B-LLM belegt ~85 % VRAM, hier bleiben wir unter ~15 %
GPU_MEM_UTIL="0.12"
MAX_MODEL_LEN="8192"     # Embedding-Chunks sind klein (nomic trunciert auf 512)
SERVED_NAME="qwen3-embed"

cmd="${1:-start}"

if [ "$cmd" = "stop" ]; then
  echo "→ Stopp: $CONTAINER"
  docker rm -f "$CONTAINER" 2>/dev/null || true
  echo "✓ Gestoppt"
  exit 0
fi

if [ "$cmd" = "status" ]; then
  echo "→ Container:"
  docker ps -a --filter "name=$CONTAINER" --format '  {{.Names}}: {{.Status}}'
  echo "→ Health:"
  curl -s -m 4 "http://localhost:$PORT/health" >/dev/null 2>&1 \
    && echo "  ✓ Embedding-Server erreichbar (Port $PORT)" \
    || echo "  ✗ Embedding-Server NICHT erreichbar"
  echo "→ Modell:"
  curl -s -m 4 "http://localhost:$PORT/v1/models" 2>/dev/null | head -c 300 || echo "  (keine Antwort)"
  exit 0
fi

# --- start ---
if docker ps --format '{{.Names}}' | grep -qx "$CONTAINER"; then
  echo "⚠ $CONTAINER läuft schon. Nutze: $0 stop && $0"
  exit 1
fi

echo "→ Starte $CONTAINER ($MODEL_REPO) auf Port $PORT"
echo "  GPU-Mem: $GPU_MEM_UTIL | max-model-len: $MAX_MODEL_LEN"

# Bestehenden (gestoppten) Container aufräumen
docker rm -f "$CONTAINER" 2>/dev/null || true

docker run -d \
  --name "$CONTAINER" \
  --restart unless-stopped \
  --ipc=host \
  --gpus all \
  -p "$PORT:8000" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  "$IMAGE" \
  "$MODEL_REPO" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 --port 8000 \
  --max-model-len "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_MEM_UTIL" \
  --trust-remote-code \
  --dtype bfloat16 \
  --max-num-seqs 64

echo "✓ Gestartet. Warte auf Health-Check (Modell-Download/Ladung kann 1–3 Min dauern)..."
for i in $(seq 1 60); do
  if curl -s -m 3 "http://localhost:$PORT/health" >/dev/null 2>&1; then
    echo "✓ Embedding-Server BEREIT nach ~$((i*5))s"
    exit 0
  fi
  # Früher Abbruch bei OOM/Crash erkennen
  if ! docker ps --filter "name=$CONTAINER" --format '{{.Names}}' | grep -qx "$CONTAINER"; then
    echo "✗ Container ist bereits beendet — Logs:"
    docker logs --tail 40 "$CONTAINER" 2>&1
    exit 1
  fi
  sleep 5
done

echo "⚠ Noch nicht bereit nach 300s — aktuelle Logs:"
docker logs --tail 30 "$CONTAINER" 2>&1
exit 2