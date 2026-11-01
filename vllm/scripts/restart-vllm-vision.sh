#!/bin/bash
# ============================================
# Recovery-Script: Qwen3.6 + Qwen2-VL-2B Instruct
# ============================================
# Regeln:
#   - Qwen 3.6 (focused_roentgen) wird NICHT berührt, stoppt oder geändert
#   - Nur Qwen 2-VL-2B wird auf Port 8001 gestartet
#   - Qwen 3.6 muss bereits als Container "focused_roentgen" existieren
# ============================================

set -euo pipefail

echo "========================================"
echo "  🔄 Qwen2-VL-2B Recovery (Qwen3.6 untouched)"
echo "========================================"
echo ""

# --- CONFIG ---
VL_MODEL="Qwen/Qwen2-VL-2B-Instruct"
VL_EXT_PORT=8001
VL_NAME="qwen2-vl-2b"

HF_CACHE="$HOME/.cache/huggingface"

# --- 0. Check prerequisites ---
echo "✅ Prüfe Voraussetzungen..."

if ! command -v docker &> /dev/null; then
    echo "❌ Docker nicht gefunden."
    echo "   Installation: curl -fsSL https://get.docker.com | sh"
    echo "   sudo usermod -aG docker \$USER && newgrp docker"
    exit 1
fi

if ! docker info &> /dev/null; then
    echo "❌ Docker daemon nicht erreichbar."
    echo "   Starten: sudo systemctl start docker"
    exit 1
fi

echo "   Docker OK"

# --- 1. Pull image ---
echo ""
echo "📦 Prüfe Docker image..."
if ! docker image inspect vllm/vllm-openai:latest &> /dev/null; then
    echo "   Pull vllm/vllm-openai:latest..."
    docker pull vllm/vllm-openai:latest
else
    echo "   ✅ Image vorhanden"
fi

# --- 2. Alte VL-Container stoppen (Qwen 3.6 untouched!) ---
echo ""
echo "🛑 Prüfe Container..."
for name in "$VL_NAME"; do
    if docker ps -a --format '{{.Names}}' | grep -q "^${name}$"; then
        echo "   Stoppe $name..."
        docker stop "$name" 2>/dev/null || true
        docker rm "$name" 2>/dev/null || true
    fi
done

echo "   ⚠️ focused_roentgen (Qwen 3.6) bleibt unverändert"

# --- 3. Download model ---
echo ""
echo "🤖 Prüfe Model-Cache..."

# Qwen2-VL-2B-Instruct (GPU — ca. 3GB)
VL_MODEL_DIR="$HF_CACHE/hub/models--Qwen--Qwen2-VL-2B-Instruct"
if [ -d "$VL_MODEL_DIR" ] && [ -d "$VL_MODEL_DIR/files" ]; then
    echo "   ✅ Qwen2-VL-2B-Instruct (GPU) vorhanden ($(du -sh "$VL_MODEL_DIR" | cut -f1))"
else
    echo "   ⬇️  Lade Qwen2-VL-2B-Instruct (ca. 3 GB)..."
    huggingface-cli download "$VL_MODEL" --cache-dir "$HF_CACHE" 2>&1 | tail -3
fi

echo ""
echo "   ✅ Modelle bereit"

# --- 4. Start Qwen2-VL-2B container (PORT 8001) ---
echo ""
echo "🚀 Starte Qwen2-VL-2B-Instruct (VL)"
echo "   Port: 8001 → intern 8000"
echo "   GPU Memory: 20%"

docker run -d \
    --name "$VL_NAME" \
    --gpus all \
    -p "$VL_EXT_PORT":8000 \
    -v "$HF_CACHE:/root/.cache/huggingface" \
    vllm/vllm-openai:latest \
    "$VL_MODEL" \
    --served-model-name qwen2-vl \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --gpu-memory-utilization 0.20 \
    --max-model-len 4096 \
    --load-format fastsafetensors

echo "   ✅ Qwen2-VL-2B-Instruct gestartet (extern Port $VL_EXT_PORT)"

# --- 5. Verification ---
echo ""
echo "⏳ Warte auf Ready-Status (VL: 10-30 Sekunden)..."
sleep 45

echo ""
echo "========================================"
echo "  📊 Status check"
echo "========================================"

# Check if containers are running
VL_RUNNING=$(docker ps --format '{{.Names}} {{.Status}}' | grep "$VL_NAME")

if [ -n "$VL_RUNNING" ]; then
    echo "✅ Qwen2-VL-2B-Instruct: RUNNING (Port $VL_EXT_PORT)"
else
    echo "❌ Qwen2-VL-2B-Instruct: FAILED — Logs: docker logs $VL_NAME"
fi

echo ""
echo "========================================"
echo "  📝 API Endpunkte"
echo "========================================"
echo ""
echo "  Qwen 3.6 (LLM):    http://localhost:8000/v1/chat/completions"
echo "  Qwen2-VL (VL):     http://localhost:$VL_EXT_PORT/v1/chat/completions"
echo ""
echo "  # Test VL:"
echo "  curl -s http://localhost:$VL_EXT_PORT/v1/models | jq '.data[].id'"
echo ""
echo "========================================"
echo "  📝 Schnellstart/Stop (nach erstem Start)"
echo "========================================"
echo ""
echo "  Container starten (ohne Download):"
echo "    docker start $VL_NAME"
echo ""
echo "  Container stoppen (RAM sparen):"
echo "    docker stop $VL_NAME"
echo ""
echo "  Container entfernen + neu erstellen:"
echo "    bash <(cat /pfad/zum/restart-vllm-vision.sh)"
echo ""
echo "========================================"