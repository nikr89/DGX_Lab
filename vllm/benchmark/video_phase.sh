#!/usr/bin/env bash
# Video-/Bild-Tests mit aktivem Vision-Encoder. Kontext-Leiter: 262144 -> 131072 -> 65536.
cd $HOME/infra_service
OUT=$HOME/infra_service/benchmarks/matrix_20261028_1106
HF=$HOME/.cache/huggingface; VC=$HOME/.cache/vllm
until grep -q FERTIG $OUT/log.txt; do sleep 30; done
try() { # label model served extra...
  label=$1; model=$2; served=$3; shift 3
  for len in 262144 131072 65536; do
    echo "=== video $label len=$len $(date +%T)" | tee -a $OUT/log.txt
    docker rm -f bench-srv >/dev/null 2>&1
    docker run -d --name bench-srv --gpus all --ipc=host --restart no -p 127.0.0.1:8000:8000 -v $HF:/root/.cache/huggingface -v $VC:/root/.cache/vllm -e HF_HUB_OFFLINE=1 "${ENVS[@]}" vllm/vllm-openai:latest $model --served-model-name $served --host 0.0.0.0 --port 8000 --trust-remote-code --dtype bfloat16 --load-format safetensors --kv-cache-dtype fp8_e4m3 --enable-chunked-prefill --async-scheduling --generation-config auto --default-chat-template-kwargs '{"enable_thinking": false}' --enable-auto-tool-choice --max-model-len $len --gpu-memory-utilization $UTIL --speculative-config '{"method":"mtp","num_speculative_tokens":1}' "$@" >/dev/null
    ok=0; for i in $(seq 1 270); do curl -sf localhost:8000/health >/dev/null && { ok=1; break; }; docker ps --format '{{.Names}}' | grep -qx bench-srv || break; sleep 10; done
    if [[ $ok -eq 1 ]]; then python3.12 monitorens/video_bench.py $label-$len > $OUT/$label.video.txt 2>&1; docker rm -f bench-srv >/dev/null; return; fi
    echo "START FEHLGESCHLAGEN video $label len=$len" | tee -a $OUT/log.txt; docker rm -f bench-srv >/dev/null 2>&1
  done
}
ENVS=(); UTIL=0.85
try ornith-video ornith-ai/Ornith-1.5-122B-NVFP4 ornith-1.5-122b --max-num-seqs 8 --max-num-batched-tokens 16384 --reasoning-parser qwen3 --tool-call-parser qwen3_xml --attention-backend flashinfer
ENVS=(-e VLLM_USE_FLASHINFER_MOE_FP4=0 -e VLLM_FLASHINFER_MOE_BACKEND=throughput); UTIL=0.70
try nemotron-video unsloth/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 nemotron-3.5-super-vl --max-num-seqs 4 --max-num-batched-tokens 8192 --enforce-eager --reasoning-parser nemotron_v3 --tool-call-parser qwen3_coder
# Qwen (vorhandener Container)
docker start qwen3.8_27b_v7 >/dev/null; for i in $(seq 1 60); do curl -sf localhost:8000/health >/dev/null && break; sleep 10; done
python3.12 monitorens/video_bench.py qwen-video > $OUT/qwen.video.txt 2>&1
docker stop qwen3.8_27b_v7 >/dev/null
echo "VIDEO FERTIG $(date +%T)" | tee -a $OUT/log.txt
