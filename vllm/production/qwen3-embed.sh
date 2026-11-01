#!/usr/bin/env bash
# Qwen3-Embedding-4B embedding server on port 8003 (container qwen3-embed), runs next to the main LLM.
# Reconstructed from `docker inspect qwen3-embed` on 2026-11-01 (vLLM v0.24.0 image at creation time).
set -euo pipefail
docker rm -f qwen3-embed >/dev/null 2>&1 || true
docker run -d --name qwen3-embed --gpus all --ipc=host --ulimit memlock=-1:-1 \
  -p 8003:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" -v "$HOME/.cache/vllm:/root/.cache/vllm" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  --entrypoint vllm vllm/vllm-openai:latest serve \
  Qwen/Qwen3-Embedding-4B --served-model-name qwen3-embed --host 0.0.0.0 --port 8000 \
  --runner pooling --convert embed --gpu-memory-utilization 0.10 --kv-cache-memory-bytes 3221225472 \
  --max-model-len 8192 --load-format fastsafetensors
