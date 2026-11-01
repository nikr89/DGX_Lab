#!/usr/bin/env bash
# Nemotron 3.5 Super VL 120B-A12B NVFP4, MTP 1, conservative "safe" profile (container nemotron-vl).
# Reconstructed from `docker inspect nemotron-vl` on 2026-11-01 (vLLM v0.31.0).
# Conservative on purpose: --enforce-eager (no CUDA graphs), FlashInfer FP4 MoE off, 64k context, 70 % memory.
# See ../benchmark/run_matrix3.sh for the faster profile with CUDA graphs + FlashInfer (MTP 2, 512k).
set -euo pipefail
docker rm -f nemotron-vl >/dev/null 2>&1 || true
docker run -d --name nemotron-vl --gpus all --ipc=host \
  -p 127.0.0.1:8000:8000 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" -v "$HOME/.cache/vllm:/root/.cache/vllm" \
  -e HF_HUB_OFFLINE=1 -e VLLM_USE_FLASHINFER_MOE_FP4=0 -e VLLM_FLASHINFER_MOE_BACKEND=throughput \
  --entrypoint vllm vllm/vllm-openai:latest serve \
  unsloth/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 --served-model-name nemotron-3.5-super-vl \
  --host 0.0.0.0 --port 8000 --trust-remote-code --dtype bfloat16 --load-format safetensors \
  --max-model-len 65536 --gpu-memory-utilization 0.70 --kv-cache-dtype fp8 \
  --max-num-seqs 8 --max-num-batched-tokens 8192 --enforce-eager \
  --enable-chunked-prefill --async-scheduling \
  --speculative-config '{"method":"mtp","num_speculative_tokens":1}' \
  --reasoning-parser nemotron_v3 --tool-call-parser qwen3_coder --enable-auto-tool-choice \
  --generation-config auto --default-chat-template-kwargs '{"enable_thinking": false}'
