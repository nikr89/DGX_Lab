#!/usr/bin/env bash
# Production: Qwen3.8-27B NVFP4, MTP 3, 48 GiB KV cache, 256k context (container qwen3.8_27b_v7).
# Reconstructed from `docker inspect qwen3.8_27b_v7` on 2026-11-01 (vLLM v0.31.0).
set -euo pipefail
docker rm -f qwen3.8_27b_v7 >/dev/null 2>&1 || true
docker run -d --name qwen3.8_27b_v7 --gpus all --ipc=host --ulimit memlock=-1:-1 \
  -p 8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" -v "$HOME/.cache/vllm:/root/.cache/vllm" \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  --entrypoint vllm vllm/vllm-openai:latest serve \
  unsloth/Qwen3.8-27B-NVFP4 --served-model-name qwen3.8-27b --host 0.0.0.0 --port 8000 \
  --tensor-parallel-size 1 --trust-remote-code \
  --kv-cache-dtype fp8 --attention-backend flashinfer --kv-cache-memory-bytes 51539607552 \
  --max-model-len 262144 --max-num-seqs 10 --max-num-batched-tokens 16384 \
  --enable-chunked-prefill --async-scheduling --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}' \
  --load-format fastsafetensors --generation-config auto \
  --reasoning-parser qwen3 --tool-call-parser qwen3_xml --enable-auto-tool-choice \
  --default-chat-template-kwargs '{"enable_thinking": false}'
