#!/usr/bin/env bash
# =============================================================================
# NOTE (2026-11-01): known issues with vLLM v0.31.0 — --max-num-partial-prefills and --disable-log-requests
# are no longer accepted, and fastsafetensors + MTP runs out of memory: use --load-format safetensors.
# The tested configuration is in ../benchmark/run_matrix3.sh (Ornith, MTP 1/2, 85 % memory).
# Ornith 1.5 122B (NVFP4) auf DGX Spark (GB10, 128 GB unified memory)
# vLLM-Template, abgeleitet aus dem bewährten qwen3.8_27b_v7-Setup.
#
# Modell: ornith-ai/Ornith-1.5-122B-NVFP4
#   - qwen3_5_moe, 48 Schichten (3/4 linear attn, 1/4 full attn)
#   - MoE 256 Experten / 8 aktiv
#   - MTP: 1 Schicht  ->  num_speculative_tokens: 1  (max. sinnvoller Wert)
#   - max_position_embeddings: 262144 (256k)
#   - NVFP4 W4A16 + FP8-Attention, ~64 GB auf Disk
#
# Nutzung:
#   1. Bestehende vLLM-Instanzen stoppen (frei RAM lassen!):
#        docker stop qwen3.8_27b_v7 qwen3-embed
#   2. ./serve_ornith15_122b_nvfp4.sh
#   3. API: http://localhost:8001  (API-Key = $HF_TOKEN oder eigener)
# =============================================================================
set -euo pipefail

# ------------------------------ Config ---------------------------------------
MODEL_REPO="ornith-ai/Ornith-1.5-122B-NVFP4"
SNAPSHOT="f049036f794622db10bb69855bf9819207334626"          # siehe hf cache
SERVED_NAME="ornith-1.5-122b"
CONTAINER_NAME="ornith15-122b"
IMAGE="vllm/vllm-openai:latest"
PORT_HOST="8001"                                              # Host-Port
PORT_CTR="8000"
API_KEY="${HF_TOKEN:-hf_token}"  # eigener Wert setzen

# HF-Cache des Hosts direkt einhängen (Modelle liegen dort schon)
HF_CACHE_HOST="$HOME/.cache/huggingface"
HF_CACHE_CTR="/root/.cache/huggingface"
MODEL_PATH="${HF_CACHE_CTR}/hub/models--ornith-ai--Ornith-1.5-122B-NVFP4/snapshots/${SNAPSHOT}"

# ----------------------- Memory-Tuning (Kernstück) ---------------------------
# NVFP4-Weights: ~64 GB.  GB10: 128 GB unified (GPU+CPU geteilt).
# vLLM belegt via unified-memory-Pool:
#   ~64 GB Weights + KV-Cache + Activations + MTP-Draft-Buffers
# KV pro Token (nur die 12 Full-Attn-Schichten, 2 KV-Heads x 256 dim x 2 x 12
#   = 12.288 Werte/Token):
#     FP8   -> 12,3 KB/Token  -> ~1,15 GB pro 256k-Request
#     FP16  -> 24,6 KB/Token  -> ~2,30 GB pro 256k-Request
# => Bei fp8 KV und ~45-50 GB restlichem KV-Budget:
#     ~30-40 parallele 256k-Sessions (hartes Limit via --max-num-seqs)
#     ~200-400 parallele Requests bei typischem 8-32k-Kontext
KV_CACHE_DTYPE="fp8_e4m3"        # fp8 = halber KV-Fussabdruck (Kritisch!)
MAX_MODEL_LEN="262144"           # 256k (maximal; fallback: 131072)
MAX_NUM_SEQS="32"                # Harte Obergrenze paralleler Requests
MAX_BATCHED_TOKENS="16384"       # Prefill-Batch-Size (Chunked-Prefill)
MAX_PARTIAL_PREFILLS="2"         # max. gleichzeitige lange Prefills
GPU_MEM_UTIL="0.90"              # Unified-Memory-Auslastung (GB10-Default)
BLOCK_SIZE="16"                  # Standard
# CPU-Offload NUR falls wirklich OOM; verlangsamt Token:
#   --cpu-offload-gb 4

# MTP (Multi-Token-Prediction): exakt so viele Draft-Tokens wie MTP-Schichten
# das Modell hat (mtp_num_hidden_layers: 1). 2-3 (wie beim Qwen) sind hier
# NICHT sinnvoll – vLLM würde nur eine Schicht nutzen und Rest ignorieren.
MTP_CONFIG='{"method":"mtp","num_speculative_tokens":1}'

# ------------------------- Parsers (wie Qwen-Setup) --------------------------
# Das Model-Repo bringt eine chat_template.jinja mit, die Tools nativ
# unterstützt (verifiziert: "# Tools\n\nYou have access to..." im Template).
# Qwen3-Kompatibler XML-Parser + Auto-Tool-Choice, identisch zum Qwen-Container.
REASONING_PARSER="qwen3"
TOOL_CALL_PARSER="qwen3_xml"
CHAT_KWARGS='{"enable_thinking": false}'

# ------------------------------- Start ---------------------------------------
MODEL_FULL="${MODEL_PATH}"
# Fallback: wenn Snapshot-Pfad im Cache nicht existiert, Repo-Namen nutzen
# (vLLM zieht via HF_HOME selbst)
if [[ ! -d "$MODEL_PATH" ]]; then
  echo "[WARN] Snapshot-Pfad fehlt, nutze Repo-Name: ${MODEL_REPO}"
  MODEL_FULL="${MODEL_REPO}"
fi

echo "==> Starte ${CONTAINER_NAME} (${MODEL_FULL})"
echo "    API: http://localhost:${PORT_HOST}  (Key: ${API_KEY:0:10}...)"

docker run -d \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  --ipc=host \
  --restart unless-stopped \
  -p "${PORT_HOST}:${PORT_CTR}" \
  -v "${HF_CACHE_HOST}:${HF_CACHE_CTR}" \
  -e HF_TOKEN="${API_KEY}" \
  -e HF_HUB_DISABLE_PROGRESS_BARS=1 \
  "${IMAGE}" \
  "${MODEL_FULL}" \
  --served-model-name "${SERVED_NAME}" \
  --host 0.0.0.0 \
  --port "${PORT_CTR}" \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --dtype bfloat16 \
  --load-format fastsafetensors \
  --max-model-len "${MAX_MODEL_LEN}" \
  --gpu-memory-utilization "${GPU_MEM_UTIL}" \
  --kv-cache-dtype "${KV_CACHE_DTYPE}" \
  --block-size "${BLOCK_SIZE}" \
  --max-num-seqs "${MAX_NUM_SEQS}" \
  --max-num-batched-tokens "${MAX_BATCHED_TOKENS}" \
  --max-num-partial-prefills "${MAX_PARTIAL_PREFILLS}" \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --async-scheduling \
  --attention-backend flashinfer \
  --speculative-config "${MTP_CONFIG}" \
  --reasoning-parser "${REASONING_PARSER}" \
  --tool-call-parser "${TOOL_CALL_PARSER}" \
  --enable-auto-tool-choice \
  --generation-config auto \
  --default-chat-template-kwargs "${CHAT_KWARGS}" \
  --api-key "${API_KEY}" \
  --disable-log-requests

echo "==> Log-Tail:"
sleep 5
docker logs --tail 30 "${CONTAINER_NAME}" 2>&1 | tail -30
