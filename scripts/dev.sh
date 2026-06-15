#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
WHISPER_DIR="$PROJECT_DIR/whisper.cpp"

# Build whisper.cpp if not already built
if [ ! -f "$WHISPER_DIR/build/bin/whisper-cli" ]; then
  echo "Building whisper.cpp..."
  cd "$WHISPER_DIR"
  cmake -B build -DGGML_CUDA=ON 2>/dev/null || cmake -B build
  cmake --build build --config Release -j$(nproc)
  cd "$PROJECT_DIR"
fi

# Download model if not present
MODEL="${WHISPER_MODEL:-base}"
if [ ! -f "$WHISPER_DIR/models/ggml-${MODEL}.bin" ]; then
  bash "$SCRIPT_DIR/download-model.sh" "$MODEL"
fi

export WHISPER_CPP_DIR="$WHISPER_DIR"
export UPLOAD_DIR="$PROJECT_DIR/uploads"
export TMP_DIR="$PROJECT_DIR/tmp"

mkdir -p "$UPLOAD_DIR" "$TMP_DIR"

echo "Starting dev server..."
cd "$PROJECT_DIR"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
