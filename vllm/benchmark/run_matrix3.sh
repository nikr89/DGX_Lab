#!/usr/bin/env bash
# Matrix 3: Nemotron MTP 2/3 (CUDA-Graphs + FlashInfer, Vision an), Ornith MTP 1/2, Qwen MTP 1/2.
# Je Konfiguration: Start, 3x Qualität, Claude-Skills, Hermes-Skills, echte Videos, Speed/Parallel/Needle (ohne 1 Mio.), Logs, Stop.
# Stoppt qwen3.8_27b_v7 am Anfang und startet ihn am Ende wieder. Aufruf: nohup monitorens/run_matrix3.sh &
cd $HOME/infra_service
OUT=${OUT:-$HOME/infra_service/benchmarks/matrix3_$(date +%Y%m%d_%H%M)}; mkdir -p $OUT  # OUT=... setzen = Lauf fortsetzen
HF=$HOME/.cache/huggingface; VC=$HOME/.cache/vllm
COMMON=(--host 0.0.0.0 --port 8000 --trust-remote-code --dtype bfloat16 --kv-cache-dtype fp8_e4m3 --enable-chunked-prefill --async-scheduling --generation-config auto --default-chat-template-kwargs '{"enable_thinking": false}' --enable-auto-tool-choice)
run_cfg() { # label model-path served mtp maxlen util extra... ; Rückgabe 1 bei Startfehler
  label=$1; shift; model=$1; served=$2; mtp=$3; maxlen=$4; util=$5; shift 5; extra=("$@")
  [[ -f $OUT/$label.spec.log ]] && { echo "=== $label schon fertig, übersprungen" | tee -a $OUT/log.txt; return 0; }
  spec=(); [[ $mtp -gt 0 ]] && spec=(--speculative-config "{\"method\":\"mtp\",\"num_speculative_tokens\":$mtp}")
  echo "=== $label (len=$maxlen util=$util) $(date +%T)" | tee -a $OUT/log.txt
  docker rm -f bench-srv >/dev/null 2>&1
  docker run -d --name bench-srv --gpus all --ipc=host --restart no -p 127.0.0.1:8000:8000 -v $HF:/root/.cache/huggingface -v $VC:/root/.cache/vllm -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 "${ENVS[@]}" vllm/vllm-openai:latest $model --served-model-name $served "${COMMON[@]}" --max-model-len $maxlen --gpu-memory-utilization $util "${spec[@]}" "${extra[@]}" >/dev/null
  ok=0; t0=$(date +%s); for i in $(seq 1 270); do curl -sf localhost:8000/health >/dev/null && { ok=1; break; }; docker ps --format '{{.Names}}' | grep -qx bench-srv || break; sleep 10; done
  if [[ $ok -eq 0 ]]; then echo "START FEHLGESCHLAGEN $label (len=$maxlen util=$util)" | tee -a $OUT/log.txt; docker logs bench-srv > $OUT/$label.startfail.log 2>&1; docker rm -f bench-srv >/dev/null 2>&1; return 1; fi
  echo "bereit nach $(( $(date +%s) - t0 )) s" | tee -a $OUT/log.txt
  for r in 1 2 3; do python3.12 -u monitorens/model_bench.py > $OUT/$label.quality$r.txt 2>&1; done
  python3.12 -u monitorens/skill_bench.py $label > $OUT/$label.skills.txt 2>&1
  python3.12 -u monitorens/hermes_skill_bench.py $label > $OUT/$label.hermes_skills.txt 2>&1
  python3.12 -u monitorens/video_real_bench.py $label > $OUT/$label.video_real.txt 2>&1
  python3.12 -u monitorens/long_bench.py $label $NEEDLES > $OUT/$label.long.txt 2>&1
  docker logs bench-srv 2>&1 | grep -E "SpecDecoding|KV cache size|Maximum concurrency|CUDA graph" > $OUT/$label.spec.log
  docker rm -f bench-srv >/dev/null 2>&1
}
watchdog() { # MemAvailable < 6 GiB -> bench-srv sofort beenden (globaler OOM am 01.11. hat Systemdienste getötet)
  while sleep 2; do a=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo)
    if (( a < 6 )) && docker ps --format '{{.Names}}' | grep -qx bench-srv; then echo "OOM-SCHUTZ $(date +%T): MemAvailable ${a} GiB, bench-srv beendet" | tee -a $OUT/log.txt; docker kill bench-srv >/dev/null 2>&1; fi; done; }
watchdog & WD=$!
restore() { kill $WD 2>/dev/null; docker rm -f bench-srv >/dev/null 2>&1; docker start qwen3.8_27b_v7 >/dev/null 2>&1; }
trap restore EXIT  # Qwen auch bei Abbruch wieder starten
docker stop qwen3.8_27b_v7 >/dev/null 2>&1; echo "qwen3.8_27b_v7 gestoppt $(date +%T)" | tee -a $OUT/log.txt

# Nemotron: Standardeinstellungen (kein --enforce-eager, FlashInfer-FP4-MoE an), Vision an. Bei Startfehler kleinerer Speicher/Kontext.
ENVS=(); NEEDLES="32000 128000 245000 500000"
NEMO=(--max-num-seqs 8 --max-num-batched-tokens 8192 --limit-mm-per-prompt '{"image":4,"video":1}' --reasoning-parser nemotron_v3 --tool-call-parser qwen3_coder --load-format safetensors)
# 01.11.: 1M startet bei MTP 2 und 3 nicht; MTP 3 mit 0.75 löste globalen OOM aus -> MTP 3 direkt 512k mit 0.70
run_cfg m3-nemotron-mtp2 unsloth/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 nemotron-3.5-super-vl 2 1048576 0.80 "${NEMO[@]}" ||
run_cfg m3-nemotron-mtp2 unsloth/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 nemotron-3.5-super-vl 2 524288 0.75 "${NEMO[@]}"
run_cfg m3-nemotron-mtp3 unsloth/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-NVFP4 nemotron-3.5-super-vl 3 524288 0.70 "${NEMO[@]}"

# Ornith: wie Matrix 1, Vision bleibt an (Standard)
ENVS=(); NEEDLES="32000 128000 245000"
for m in 1 2; do run_cfg m3-ornith-mtp$m ornith-ai/Ornith-1.5-122B-NVFP4 ornith-1.5-122b $m 262144 0.85 --max-num-seqs 32 --max-num-batched-tokens 16384 --reasoning-parser qwen3 --tool-call-parser qwen3_xml --attention-backend flashinfer --load-format safetensors; done

# Qwen: Parameter wie qwen3.8_27b_v7, nur MTP geändert (KV-Cache fest 48 GiB statt gpu-memory-utilization).
# Eigener Modellname, damit Hermes-Cronjobs (rufen "qwen3.8-27b") nicht in die Messung laufen.
ENVS=(); NEEDLES="32000 128000 245000"
for m in 1 2 3; do run_cfg m3-qwen-mtp$m unsloth/Qwen3.8-27B-NVFP4 qwen3.8-27b-bench $m 262144 0.90 --kv-cache-memory-bytes 51539607552 --attention-backend flashinfer --max-num-seqs 10 --max-num-batched-tokens 16384 --enable-prefix-caching --load-format fastsafetensors --reasoning-parser qwen3 --tool-call-parser qwen3_xml; done

python3.12 -u monitorens/matrix_report.py $OUT > /dev/null 2>&1
restore; for i in $(seq 1 60); do curl -sf localhost:8000/health >/dev/null && break; sleep 10; done
echo "FERTIG $(date +%T), qwen3.8_27b_v7 wieder gestartet" | tee -a $OUT/log.txt
