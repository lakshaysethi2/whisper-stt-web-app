# User Requirements — whisper-stt-web-app

This file tracks user requirements and design decisions for the project.

## Core purpose

A self-hosted speech-to-text web application using faster-whisper (CTranslate2).
Upload audio files or record from microphone; transcription runs on user's own server.

## Key requirements

### Deployment

1. **Docker-only deployment** — all operations (build, run, test) use Docker.
2. **CPU-first** — default model is `base`, GPU support is optional override.
3. **ARM-compatible** — Oracle A1 (ARM64) is a primary deployment target.
4. **Single command** to deploy: `docker compose up -d`.
5. **Port 8561** is the default host port (→ container 8000).

### Storage & disk

1. **tmpfs work dir** — `/tmp/whisper-stt` is tmpfs, size=2G, no host disk used.
2. **MAX_FILE_SIZE** default: 512 MB (safe on 17G free disk).
3. **MIN_FREE_DISK_BYTES** default: 2 GB — uploads rejected below threshold.
4. **Auto cleanup** — stale jobs/chunks older than 30 min removed every 10 min.
5. **Startup cleanup** — all stale data wiped on container start.

### Model

1. Default model: `base` (CPU-friendly, ~150 MB disk cache).
2. Model overridable via `WHISPER_MODEL` env var.
3. GPU auto-detection with fallback to CPU int8.
4. Supported: tiny, base, small, medium, large-v3, large-v3-turbo.

### Testing

1. Test audio fixtures in `test_audio/` — LibriSpeech speaker 3081, chapter 166546.
2. Fixtures reproducible via `scripts/build_test_audio.sh`.
3. Accuracy benchmark via `scripts/run_accuracy_bench.sh`.
4. Accuracy gate: >= 95% word accuracy on clean read speech (1min, 5min).

### API

1. `GET /health` — health check with model/device info.
2. `POST /api/transcribe` — direct file upload + transcription.
3. `POST /api/upload/start|chunk|finish` — chunked upload for large files.
4. `GET /api/transcribe/status/{job_id}` — poll async transcription result.
5. `GET /api/models` — list available models and current device.
6. `GET /api/upload/config` — client configuration for chunked upload.

### GPU (optional)

1. GPU support via `-f docker-compose.gpu.yml` override.
2. Default GPU model: `large-v3-turbo` (better quality on GPU).
3. CUDA compute capability 5.0+ supported.
4. Auto-selects float16/float32 based on GPU architecture.
