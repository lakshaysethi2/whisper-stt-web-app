#!/usr/bin/env bash
# Accuracy benchmark for whisper-stt-web-app.
#
# Assumes the app is running and healthy at $BASE_URL (default http://127.0.0.1:8000).
# Uploads each test_audio/*.flac, records timings, computes word accuracy
# vs the corresponding *_transcript.txt ground truth, and prints a Markdown table.
#
# Usage:
#   ./scripts/run_accuracy_bench.sh                    # uses default localhost:8000
#   BASE_URL=http://myhost:8561 ./scripts/run_accuracy_bench.sh
#
# Prerequisites: docker (for python3 + levenshtein), curl, jq
#
set -euo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT="$PWD"

BASE_URL="${BASE_URL:-http://127.0.0.1:8000}"
TEST_AUDIO="$PROJECT_ROOT/test_audio"
HEALTH_ENDPOINT="${BASE_URL}/health"
TRANSCRIBE_ENDPOINT="${BASE_URL}/api/transcribe"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# -------------------------------------------------------------------
# 1. Verify app is healthy
# -------------------------------------------------------------------
echo "=== Whisper STT Accuracy Benchmark ==="
echo "Base URL: $BASE_URL"
echo ""

echo -n "[1/4] Checking health ... "
HEALTH=$(curl -sf "$HEALTH_ENDPOINT" 2>/dev/null || true)
if [ -z "$HEALTH" ]; then
    echo -e "${RED}FAILED${NC}"
    echo "App is not reachable at $HEALTH_ENDPOINT"
    echo "Start the app first, e.g."
    echo "  docker compose up -d"
    echo "  # or"
    echo "  BASE_URL=http://host:8561 $0"
    exit 1
fi

MODEL=$(echo "$HEALTH" | jq -r '.model // "unknown"')
DEVICE=$(echo "$HEALTH" | jq -r '.device // "unknown"')
COMPUTE=$(echo "$HEALTH" | jq -r '.compute_type // "unknown"')
echo -e "${GREEN}OK${NC}  model=$MODEL  device=$DEVICE/$COMPUTE"
echo ""

# -------------------------------------------------------------------
# 2. Gather test files
# -------------------------------------------------------------------
echo "[2/4] Finding test fixtures ..."
FIXTURES=()
for f in "$TEST_AUDIO"/*.flac; do
    [ -f "$f" ] || continue
    base=$(basename "$f" .flac)
    txt="$TEST_AUDIO/${base}_transcript.txt"
    if [ -f "$txt" ]; then
        FIXTURES+=("$base")
        echo "      $base.flac  →  ${base}_transcript.txt"
    else
        echo -e "${YELLOW}      $base.flac (no transcript, skipping)${NC}"
    fi
done

if [ ${#FIXTURES[@]} -eq 0 ]; then
    echo -e "${RED}No test fixtures found in $TEST_AUDIO${NC}"
    echo "Run scripts/build_test_audio.sh first."
    exit 1
fi
echo ""

# -------------------------------------------------------------------
# 3. Run benchmark per fixture
# -------------------------------------------------------------------
echo "[3/4] Running benchmark ..."
echo ""

RESULTS_DIR=$(mktemp -d)
trap 'rm -rf "$RESULTS_DIR"' EXIT

RESULTS_JSON="$RESULTS_DIR/results.json"
echo '[]' > "$RESULTS_JSON"

for base in "${FIXTURES[@]}"; do
    flac="$TEST_AUDIO/${base}.flac"
    txt="$TEST_AUDIO/${base}_transcript.txt"

    echo "  --- $base ---"

    # Get audio duration
    DURATION=$(docker run --rm --network host \
        -v "$flac:/audio.flac:ro" \
        python:3.11-slim \
        sh -c 'apt-get update -qq && apt-get install -y -qq sox flac >/dev/null 2>&1 && soxi -D /audio.flac 2>/dev/null || echo "0"' 2>/dev/null)
    echo "  Audio duration: ${DURATION}s"

    # Upload and transcribe, recording wall time
    START_TIME=$(date +%s.%N)
    RESPONSE_FILE="$RESULTS_DIR/${base}_response.json"
    HTTP_CODE=$(curl -s -o "$RESPONSE_FILE" -w "%{http_code}" \
        -X POST "$TRANSCRIBE_ENDPOINT" \
        -F "file=@$flac" \
        -F "language=en")
    END_TIME=$(date +%s.%N)
    WALL_TIME=$(echo "$END_TIME - $START_TIME" | bc -l 2>/dev/null || echo "0")
    WALL_TIME_ROUNDED=$(printf "%.2f" "$WALL_TIME")

    if [ "$HTTP_CODE" != "200" ]; then
        echo -e "  ${RED}HTTP $HTTP_CODE — skipping${NC}"
        continue
    fi

    # Extract fields from response
    PROCESS_TIME=$(jq -r '.process_time // 0' "$RESPONSE_FILE")
    RESP_DEVICE=$(jq -r '.device // "unknown"' "$RESPONSE_FILE")
    RESP_COMPUTE=$(jq -r '.compute_type // "unknown"' "$RESPONSE_FILE")
    TRANS_TEXT=$(jq -r '.text // ""' "$RESPONSE_FILE")

    # Save transcription for debugging
    echo "$TRANS_TEXT" > "$RESULTS_DIR/${base}_transcribed.txt"

    # Compute accuracy using Levenshtein distance via python-Levenshtein
    ACCURACY=$(docker run --rm --network host \
        -v "$txt:/ground_truth.txt:ro" \
        -v "$RESULTS_DIR/${base}_transcribed.txt:/hypothesis.txt:ro" \
        -v "$PROJECT_ROOT/scripts:/scripts:ro" \
        python:3.11-slim \
        sh -c 'pip install -q python-Levenshtein >/dev/null 2>&1 && python3 /scripts/_compute_wer.py /ground_truth.txt /hypothesis.txt' 2>/dev/null)
    ACC_PCT=$(echo "$ACCURACY" | awk "{print \$1}")
    echo -e "  Accuracy: ${ACC_PCT}%  Process: ${PROCESS_TIME}s  Wall: ${WALL_TIME_ROUNDED}s  Device: ${RESP_DEVICE} (${RESP_COMPUTE})"

    # Append to results
    TMP=$(mktemp)
    jq --arg name "$base" \
       --arg dur "$DURATION" \
       --arg acc "$ACC_PCT" \
       --arg proc "$PROCESS_TIME" \
       --arg wall "$WALL_TIME_ROUNDED" \
       --arg dev "$RESP_DEVICE" \
       --arg comp "$RESP_COMPUTE" \
       '. += [{"file": $name, "duration_s": ($dur|tonumber), "accuracy_pct": ($acc|tonumber), "process_time_s": ($proc|tonumber), "wall_time_s": ($wall|tonumber), "device": $dev, "compute_type": $comp}]' \
       "$RESULTS_JSON" > "$TMP" && mv "$TMP" "$RESULTS_JSON"
done

echo ""

# -------------------------------------------------------------------
# 4. Print results table
# -------------------------------------------------------------------
echo "[4/4] Results"
echo ""

# Print Markdown table
echo "| File | Duration (s) | Accuracy % | Process (s) | Wall (s) | Device |"
echo "|------|-------------|-----------|-------------|----------|--------|"

jq -r '.[] | [.file, .duration_s, .accuracy_pct, .process_time_s, .wall_time_s, .device] | @tsv' "$RESULTS_JSON" | \
while IFS=$'\t' read -r file dur acc proc wall dev; do
    printf "| %s | %.1f | %.2f | %.2f | %.2f | %s |\n" "$file" "$dur" "$acc" "$proc" "$wall" "$dev"
done

echo ""

# Summary
PASS_COUNT=0
FAIL_COUNT=0
TOTAL_COUNT=0
while IFS=$'\t' read -r file dur acc proc wall dev; do
    TOTAL_COUNT=$((TOTAL_COUNT + 1))
    if [ "$(echo "$acc >= 95.0" | awk '{if ($1+0 >= 95.0) print 1; else print 0}' 2>/dev/null)" = "1" ]; then
        PASS_COUNT=$((PASS_COUNT + 1))
    else
        FAIL_COUNT=$((FAIL_COUNT + 1))
    fi
done < <(jq -r '.[] | [.file, .duration_s, .accuracy_pct, .process_time_s, .wall_time_s, .device] | @tsv' "$RESULTS_JSON")

echo "Passed: $PASS_COUNT / $TOTAL_COUNT  (threshold: >= 95% accuracy)"
echo "Model: $MODEL | Device: $DEVICE/$COMPUTE"
echo ""

if [ "$FAIL_COUNT" -eq 0 ] && [ "$TOTAL_COUNT" -gt 0 ]; then
    echo -e "${GREEN}All accuracy gates PASSED.${NC}"
else
    echo -e "${RED}Some accuracy gates FAILED.${NC}"
fi
