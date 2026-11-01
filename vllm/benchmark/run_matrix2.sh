#!/usr/bin/env bash
# Matrix: Modell x MTP. Jede Konfiguration: Start, 3x Qualität, Skills, Speed/Parallel/Needle, Logs, Stop.
cd $HOME/infra_service
OUT=$HOME/infra_service/benchmarks/matrix_20261028_1106
HF=$HOME/.cache/huggingface; VC=$HOME/.cache/vllm
COMMON=(--host 0.0.0.0 --port 8000 --trust-remote-code --dtype bfloat16 --load-format safetensors --kv-cache-dtype fp8_e4m3 --enable-chunked-prefill --async-scheduling --generation-config auto --default-chat-template-kwargs '{"enable_thinking": false}' --enable-auto-tool-choice)
run_cfg() { # label model-path served mtp maxlen util seqs parser-args... 
  label=$1; shift; model=$1; served=$2; mtp=$3; maxlen=$4; util=$5; shift 5; extra=("$@")
  spec=(); [[ $mtp -gt 0 ]] && spec=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$mtp}")
  echo "=== $label $(date +%T)" | tee -a $OUT/log.txt
  docker rm -f bench-srv >/dev/null 2>&1
  docker run -d --name bench-srv --gpus all --ipc=host --restart no -p 127.0.0.1:8000:8000 -v $HF:/root/.cache/huggingface -v $VC:/root/.cache/vllm -e HF_HUB_OFFLINE=1 "${ENVS[@]}" vllm/vllm-openai:latest $model --served-model-name $served "${COMMON[@]}" --max-model-len $maxlen --gpu-memory-utilization $util "${spec[@]}" "${extra[@]}" >/dev/null
  ok=0; for i in $(seq 1 270); do curl -sf localhost:8000/health >/dev/null && { ok=1; break; }; docker ps --format '{{.Names}}' | grep -qx bench-srv || break; sleep 10; done
  if [[ $ok -eq 0 ]]; then echo "START FEHLGESCHLAGEN $label" | tee -a $OUT/log.txt; docker logs bench-srv > $OUT/$label.startfail.log 2>&1; docker rm -f bench-srv >/dev/null 2>&1; return; fi
  for r in 1 2 3; do python3.12 monitorens/model_bench.py > $OUT/$label.quality$r.txt 2>&1; done
  python3.12 monitorens/skill_bench.py $label > $OUT/$label.skills.txt 2>&1
  python3.12 monitorens/long_bench.py $label $NEEDLES > $OUT/$label.long.txt 2>&1
  docker logs bench-srv 2>&1 | grep -E "SpecDecoding|KV cache size|Maximum concurrency" > $OUT/$label.spec.log
  docker rm -f bench-srv >/dev/null 2>&1
}
ENVS=(-e VLLM_USE_FLASHINFER_MOE_FP4=0 -e VLLM_FLASHINFER_MOE_BACKEND=throughput); NEEDLES="32000 128000 245000 500000 1000000"
for m in 0 1 3; do run_cfg nemotron-mtp$m unsloth/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 nemotron-3.5-super-vl $m 1048576 0.70 --max-num-seqs 8 --max-num-batched-tokens 8192 --enforce-eager --limit-mm-per-prompt '{"image":0,"video":0}' --reasoning-parser nemotron_v3 --tool-call-parser qwen3_coder; done
# Qwen: vorhandener Container (MTP 3, 256k)
echo "=== qwen-mtp3 $(date +%T)" | tee -a $OUT/log.txt
docker start qwen3.8_27b_v7 >/dev/null; for i in $(seq 1 60); do curl -sf localhost:8000/health >/dev/null && break; sleep 10; done
for r in 1 2 3; do python3.12 monitorens/model_bench.py > $OUT/qwen-mtp3.quality$r.txt 2>&1; done
python3.12 monitorens/skill_bench.py qwen-mtp3 > $OUT/qwen-mtp3.skills.txt 2>&1
python3.12 monitorens/long_bench.py qwen-mtp3 32000 128000 245000 > $OUT/qwen-mtp3.long.txt 2>&1
docker logs qwen3.8_27b_v7 2>&1 | grep -E "SpecDecoding|KV cache size|Maximum concurrency" | tail -300 > $OUT/qwen-mtp3.spec.log
docker stop qwen3.8_27b_v7 >/dev/null
echo "FERTIG $(date +%T)" | tee -a $OUT/log.txt
