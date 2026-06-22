#!/bin/bash
set -e

MODEL="${WHISPER_MODEL:-base}"

echo "==> Building and running with GPU (CTranslate2 SM 5.0)..."
echo "    First build takes ~15-20 min (CUDA compilation)."
echo ""

docker compose up --build
