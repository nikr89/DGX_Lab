#!/usr/bin/env bash
# ============================================================
# vllm_relaunch.sh — vLLM-Container neu starten (GB10)
# VORBEREITET — manuell anstoßen:
#   ./vllm_relaunch.sh                    # mit aktuellem Modell
#   ./vllm_relaunch.sh unsloth/Qwen3.8-27B-NVFP4
#
# GOLDEN RULE: Qwen 3.6/3.8 (GPU) wird NIEMALS entfernt/changed ohne
# ausdrückliche Zustimmung. Dieses Skript STARTET NUR NEU — es ändert
# weder Modell-Weight noch entfernt etwas. Bei Modellwechsel: vorher
# explicit bestätigen.
# ============================================================
set -euo pipefail

CONTAINER="vllm-qwen38"
MODEL="${1:-unsloth/Qwen3.8-27B-NVFP4}"
PORT=8000
GPU_UTIL=0.85
MAX_LEN=262144
KV_BYTES=42949672960   # 40 GB KV-Cache

echo "=== Vorher: Container-Status ==="
docker ps -a --filter "name=$CONTAINER" --format '{{.Names}}: {{.Status}}'

echo "=== Vorher: RAM ==="
grep -E "MemAvailable" /proc/meminfo

echo ""
echo "=== Stoppe alten Container (Modelle bleiben im HF-Cache) ==="
docker stop "$CONTAINER" 2>/dev/null || true
docker rm "$CONTAINER" 2>/dev/null || true

echo ""
echo "=== Starte neuen Container: $MODEL (Port $PORT) ==="
docker run -d --name "$CONTAINER" --gpus all \
  -p "$PORT:$PORT" \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  -v $HOME/.cache/vllm:/root/.cache/vllm \
  --shm-size=16g \
  vllm/vllm-openai:latest \
  --model "$MODEL" \
  --served-model-name "qwen3.8-27b" \
  --host 0.0.0.0 --port "$PORT" \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --attention-backend flashinfer \
  --gpu-memory-utilization "$GPU_UTIL" \
  --kv-cache-memory-bytes "$KV_BYTES" \
  --max-model-len "$MAX_LEN" \
  --max-num-seqs 10 \
  --max-num-batched-tokens 16384 \
  --enable-chunked-prefill \
  --async-scheduling \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --load-format fastsafetensors \
  --generation-config auto \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice

echo ""
echo "=== Warte auf Readiness (max. 300s) ==="
for i in $(seq 1 60); do
  code=$(curl -s --max-time 5 -o /dev/null -w '%{http_code}' "http://localhost:$PORT/v1/models" 2>/dev/null || echo 000)
  if [ "$code" = "200" ]; then
    echo "READY nach $((i*5))s (HTTP 200)"
    break
  fi
  sleep 5
done
if [ "$code" != "200" ]; then
  echo "FEHLER: nicht ready nach 300s. Logs: docker logs $CONTAINER | tail -50"
  exit 1
fi

echo ""
echo "=== Modell-Liste ==="
curl -s "http://localhost:$PORT/v1/models" | python3.12 -c "import json,sys; [print('  ', m['id']) for m in json.load(sys.stdin)['data']]"
echo "=== RAM nachher ==="
grep -E "MemAvailable" /proc/meminfo
echo "Fertig. Health-Check: ../healthcheck.sh"