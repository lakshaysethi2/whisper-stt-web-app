#!/usr/bin/env bash
# Build test audio fixtures from LibriSpeech (OpenSLR #12).
#
# Downloads the dev-clean subset, extracts speaker 3081 / chapter 166546,
# and produces 1min.flac, 5min.flac, full.flac with matching *_transcript.txt
# ground-truth files in test_audio/.
#
# Usage:  ./scripts/build_test_audio.sh
#
# Prerequisites: docker (for curl, ffmpeg, flac, sox, grep, sed, awk)
# The script uses a lightweight Debian-based container so no host tools
# beyond docker are needed.
#
# The downloaded tarball is cached inside a Docker volume; only the first
# run downloads from upstream.
#
set -euo pipefail
cd "$(dirname "$0")/.."
PROJECT_ROOT="$PWD"
TEST_AUDIO="$PROJECT_ROOT/test_audio"
mkdir -p "$TEST_AUDIO"

SPEAKER="3081"
CHAPTER="166546"
DEVCLEAN_TGZ="dev-clean.tar.gz"
DEVCLEAN_URL="https://www.openslr.org/resources/12/${DEVCLEAN_TGZ}"
CACHE_DIR="/tmp/librispeech-cache"
mkdir -p "$CACHE_DIR"

echo "=== LibriSpeech test-fixture builder ==="
echo "Speaker: $SPEAKER  Chapter: $CHAPTER"
echo ""

# -------------------------------------------------------------------
# 1. Download dev-clean tarball if not cached
# -------------------------------------------------------------------
TARBALL="$CACHE_DIR/$DEVCLEAN_TGZ"
if [ -f "$TARBALL" ]; then
    echo "[1/4] Using cached tarball: $TARBALL"
else
    echo "[1/4] Downloading dev-clean (~330 MB) from OpenSLR..."
    echo "      URL: $DEVCLEAN_URL"
    docker run --rm --network host \
        -v "$CACHE_DIR:/cache" \
        -w /cache \
        python:3.11-slim \
        sh -c "apt-get update -qq && apt-get install -y -qq curl >/dev/null 2>&1 && curl -fsSL -o /cache/$DEVCLEAN_TGZ '$DEVCLEAN_URL'"
    echo "      Download complete."
fi

# -------------------------------------------------------------------
# 2. Extract only our speaker/chapter from the tarball
# -------------------------------------------------------------------
echo "[2/4] Extracting speaker $SPEAKER / chapter $CHAPTER..."
EXTRACT_DIR="$CACHE_DIR/extracted"
mkdir -p "$EXTRACT_DIR"

# Extract just the directory for our speaker/chapter
tar -xzf "$TARBALL" -C "$EXTRACT_DIR" \
    "LibriSpeech/dev-clean/${SPEAKER}/${CHAPTER}" \
    2>/dev/null || tar -xzf "$TARBALL" -C "$EXTRACT_DIR" \
    "dev-clean/${SPEAKER}/${CHAPTER}" 2>/dev/null || true

# Check if extraction succeeded
CHAPTER_DIR=""
for candidate in "$EXTRACT_DIR/LibriSpeech/dev-clean/$SPEAKER/$CHAPTER" \
                 "$EXTRACT_DIR/dev-clean/$SPEAKER/$CHAPTER"; do
    if [ -d "$candidate" ]; then
        CHAPTER_DIR="$candidate"
        break
    fi
done

if [ -z "$CHAPTER_DIR" ]; then
    echo "ERROR: Could not find chapter directory after extraction."
    echo "Contents of extract dir:"
    find "$EXTRACT_DIR" -maxdepth 4 -type d | head -20
    exit 1
fi

echo "      Found chapter at: $CHAPTER_DIR"

# -------------------------------------------------------------------
# 3. Build ground-truth transcripts from the .trans.txt file
# -------------------------------------------------------------------
echo "[3/4] Building ground-truth transcripts..."

TRANS_FILE=$(find "$CHAPTER_DIR" -name "*.trans.txt" | head -1)
if [ -z "$TRANS_FILE" ]; then
    echo "ERROR: No .trans.txt file found in $CHAPTER_DIR"
    exit 1
fi

# Build full transcript: join all lines in utterance order
FULL_TRANSCRIPT="$TEST_AUDIO/full_transcript.txt"
# The .trans.txt lines are: SPEAKER-CHAPTER-UTTID TEXT
# Extract just the text part, preserving order
awk '{$1=""; sub(/^ /, ""); print}' "$TRANS_FILE" > "$FULL_TRANSCRIPT"

echo "      Full transcript: $(wc -w < "$FULL_TRANSCRIPT") words"

# -------------------------------------------------------------------
# 4. Concatenate FLACs and build truncated fixtures
# -------------------------------------------------------------------
echo "[4/4] Building audio fixtures..."

# Collect sorted flac files
FLAC_FILES=$(find "$CHAPTER_DIR" -name "*.flac" | sort)
FLAC_COUNT=$(echo "$FLAC_FILES" | wc -l)
echo "      Found $FLAC_COUNT flac utterances"

# Combine all flacs into full.flac with sox
# Use Docker with sox/ffmpeg installed
echo "      Building full.flac ..."
docker run --rm --network host \
    -v "$CHAPTER_DIR:/app/input:ro" \
    -v "$TEST_AUDIO:/app/output" \
    -w /app \
    python:3.11-slim \
    sh -c '
        apt-get update -qq && apt-get install -y -qq sox flac >/dev/null 2>&1
        # Find sorted flac files in input dir
        files=$(find /app/input -name "*.flac" | sort)
        if [ -z "$files" ]; then
            echo "ERROR: No flac files found in /app/input"
            exit 1
        fi
        # Build sox command to concatenate all flacs
        # sox requires all files have the same sample rate/channels; LibriSpeech is uniform
        sox $files /app/output/full.flac
        echo "      full.flac created: $(soxi -D /app/output/full.flac 2>/dev/null || echo "?") seconds"
    '

echo "      Building 1min.flac and 5min.flac ..."
# Trim first 60 and 300 seconds from full.flac
docker run --rm --network host \
    -v "$TEST_AUDIO:/app" \
    -w /app \
    python:3.11-slim \
    sh -c '
        apt-get update -qq && apt-get install -y -qq sox flac >/dev/null 2>&1
        # Get total duration
        duration=$(soxi -D /app/full.flac 2>/dev/null || echo "0")
        echo "      full.flac duration: ${duration}s"

        # Trim 1min (60s) - handle if chapter is shorter
        sox /app/full.flac /app/1min.flac trim 0 60 2>/dev/null || \
            cp /app/full.flac /app/1min.flac

        # Trim 5min (300s)
        sox /app/full.flac /app/5min.flac trim 0 300 2>/dev/null || \
            cp /app/full.flac /app/5min.flac

        for f in 1min 5min; do
            dur=$(soxi -D /app/${f}.flac 2>/dev/null || echo "?")
            echo "      ${f}.flac: ${dur}s"
        done
    '

# Build truncated transcripts matching the trimmed audio
echo "      Building truncated transcripts ..."

# Get line-level transcripts to build corresponding text
# Each line in the trans file corresponds to one utterance
# We need to figure out how many utterances fit in 60s / 300s
# For simplicity, build transcripts from the full transcript but compute
# approximate word counts based on the audio duration ratio.
# Actually, the proper way is to use soxi on each utterance and sum until we
# hit the time limit. Let's do that properly.

docker run --rm --network host \
    -v "$CHAPTER_DIR:/app/input:ro" \
    -v "$TEST_AUDIO:/app/output" \
    -v "$PROJECT_ROOT/scripts:/app/scripts:ro" \
    -w /app \
    python:3.11-slim \
    sh -c '
        apt-get update -qq && apt-get install -y -qq sox flac >/dev/null 2>&1
        python3 /app/scripts/_build_transcripts.py /app/input /app/output
    '
    sync

# The full_transcript.txt was already created from the .trans.txt file in step 3

# Show results
echo ""
echo "=== Results ==="
for f in 1min 5min full; do
    flac="$TEST_AUDIO/${f}.flac"
    txt="$TEST_AUDIO/${f}_transcript.txt"
    if [ -f "$flac" ]; then
        dur=$(docker run --rm --network host -v "$TEST_AUDIO:/app" -w /app python:3.11-slim sh -c "apt-get update -qq && apt-get install -y -qq sox flac >/dev/null 2>&1 && soxi -D /app/${f}.flac 2>/dev/null || echo \"?\"" 2>/dev/null || echo "?")
        size=$(stat -c%s "$flac" 2>/dev/null || stat -f%z "$flac" 2>/dev/null || echo "?")
        words=$(wc -w < "$txt" 2>/dev/null || echo "0")
        echo "  $f.flac: ${dur}s, ${size} bytes, $words words"
    fi
done

echo ""
echo "Done. Fixtures in test_audio/"
