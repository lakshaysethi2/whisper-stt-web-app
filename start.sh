#!/usr/bin/env bash
cd "$(dirname "$0")"
export WORK_DIR="$(pwd)/tmp"
mkdir -p "$WORK_DIR"
exec .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
