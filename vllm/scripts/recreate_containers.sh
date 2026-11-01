#!/usr/bin/env bash
# Recreate vLLM containers on the current vllm/vllm-openai:latest image.
# Old containers are renamed to <name>-old (rollback: docker rm <name> && docker rename <name>-old <name>).
# New containers are only created, not started.
set -euo pipefail

IMAGE=vllm/vllm-openai:latest
COMMON=(
  --gpus all --ipc host --ulimit memlock=-1:-1
  -v $HOME/.cache/vllm:/root/.cache/vllm
  -v $HOME/.cache/huggingface:/root/.cache/huggingface
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
  --entrypoint vllm
)

recreate() {
  local name=$1 port=$2; shift 2
  if docker inspect "$name-old" >/dev/null 2>&1; then
    echo "$name-old exists already, skipping $name" >&2; return
  fi
  docker rename "$name" "$name-old"
  docker create --name "$name" -p "$port:8000" "${COMMON[@]}" "$IMAGE" serve "$@" >/dev/null
  echo "recreated $name (port $port)"
}

recreate vllm-qwen36 8001 \
  nvidia/Qwen3.6-35B-A3B-NVFP4 --served-model-name qwen3.6-35b-a3b --host 0.0.0.0 --port 8000 \
  --quantization modelopt --trust-remote-code --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.27 --kv-cache-memory-bytes 8589934592 \
  --max-model-len 262144 --max-num-seqs 8 --max-num-batched-tokens 8192 --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --load-format fastsafetensors --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice

recreate qwen3-embed 8003 \
  Qwen/Qwen3-Embedding-4B --served-model-name qwen3-embed --host 0.0.0.0 --port 8000 \
  --runner pooling --convert embed --gpu-memory-utilization 0.10 --kv-cache-memory-bytes 3221225472 \
  --max-model-len 8192 --load-format fastsafetensors

recreate vllm-qwen38-alt 8000 \
  unsloth/Qwen3.8-27B-NVFP4 --served-model-name qwen3.8-27b --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 --trust-remote-code --kv-cache-dtype fp8 --attention-backend flashinfer \
  --gpu-memory-utilization 0.85 --kv-cache-memory-bytes 42949672960 \
  --max-model-len 262144 --max-num-seqs 10 --max-num-batched-tokens 16384 \
  --enable-chunked-prefill --async-scheduling --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --load-format fastsafetensors --generation-config auto \
  --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice
