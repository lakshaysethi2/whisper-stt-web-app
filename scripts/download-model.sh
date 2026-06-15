#!/usr/bin/env bash
set -euo pipefail

MODEL="${1:-base}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
WHISPER_DIR="${SCRIPT_DIR}/../whisper.cpp"

if [ ! -d "$WHISPER_DIR" ]; then
  echo "Error: whisper.cpp not found at $WHISPER_DIR"
  echo "Run: git clone --recursive https://github.com/ggerganov/whisper.cpp.git whisper.cpp"
  exit 1
fi

cd "$WHISPER_DIR"
echo "Downloading model: $MODEL"
bash models/download-ggml-model.sh "$MODEL"
echo "Model saved to: models/ggml-${MODEL}.bin"
