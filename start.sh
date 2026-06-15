#!/usr/bin/env bash
cd "$(dirname "$0")"
export LD_LIBRARY_PATH="/usr/local/lib/ollama/cuda_v12:${LD_LIBRARY_PATH}"
export WORK_DIR="$(pwd)/tmp"
mkdir -p "$WORK_DIR"
exec .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 5000
