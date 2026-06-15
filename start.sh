#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
export WORK_DIR="$(pwd)/tmp"
mkdir -p "$WORK_DIR"

if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
    .venv/bin/pip install -r requirements.txt
fi

exec .venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8000
