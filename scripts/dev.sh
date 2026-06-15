#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

export UPLOAD_DIR="$PROJECT_DIR/uploads"
export TMP_DIR="$PROJECT_DIR/tmp"

mkdir -p "$UPLOAD_DIR" "$TMP_DIR"

echo "Starting dev server..."
cd "$PROJECT_DIR"
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
