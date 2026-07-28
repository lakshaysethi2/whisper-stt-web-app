# Whisper STT Web App

A self-hosted speech-to-text web application powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2). Upload audio files or record directly from your microphone — transcription runs on your server.

- **CPU-friendly** — Default model is `base` (74M params). Runs efficiently on CPU with int8 quantization.
- **GPU optional** — Activate GPU support with a compose override for NVIDIA GPUs.
- **ARM compatible** — Dockerfile.cpu targets ARM64 (Oracle A1, Raspberry Pi) and x86_64.
- **Mobile-first PWA** — Install on your phone like a native app.

## Features

- **Live recording** — Record audio directly in the browser
- **File upload** — Upload audio (MP3, WAV, M4A, FLAC, OGG, etc.) or video (MP4, AVI, MOV, MKV, etc.) files
- **GPU-accelerated** (optional) — Runs on NVIDIA GPU with CUDA for maximum speed
- **CPU inference** — Uses int8 quantization for efficient CPU transcription
- **Auto-detection** — Automatically selects optimal compute type (float16/float32/int8) based on hardware
- **Mobile-first PWA** — Install on your phone like a native app
- **Private** — Self-hosted, your audio never leaves your server
- **Docker-ready** — One command to deploy

## Quick Start

### With Docker (recommended)

```bash
git clone https://github.com/lakshaysethi2/whisper-stt-web-app.git
cd whisper-stt-web-app
docker compose up --build
```

Open http://localhost:8561 in your browser.

### With GPU

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build
```

### Manual Setup

**Requirements:**
- Python 3.10+
- 4 GB+ system RAM

```bash
git clone https://github.com/lakshaysethi2/whisper-stt-web-app.git
cd whisper-stt-web-app
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./start.sh
```

## Deployment Guide

### CPU / ARM host (e.g. Oracle A1)

See [docs/deploy-orc.md](docs/deploy-orc.md) for a detailed guide on deploying to
an Oracle A1 ARM instance with tight disk.

Key settings for CPU-only ARM hosts:

- Default port: **8561** (→ container 8000)
- Default model: **base** (CPU-friendly)
- Image: `Dockerfile.cpu` (slim, ARM-compatible, ~500 MB)
- Disk: tmpfs work dir (size=2G), auto-cleanup, uploads rejected below 2 GB free

### GPU host

On a machine with an NVIDIA GPU and `nvidia-container-toolkit`:

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build
```

The GPU override:
- Builds from the CUDA-enabled `Dockerfile` (nvidia/cuda base)
- Sets `WHISPER_MODEL=large-v3-turbo` for best quality
- Adds NVIDIA GPU device reservation

## Configuration

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL` | `base` | Model to use (tiny, base, small, medium, large-v3, large-v3-turbo) |
| `WHISPER_LANGUAGE` | `en` | Language for transcription |
| `HOST_PORT` | `8561` | Host port mapping |
| `MAX_FILE_SIZE` | `536870912` (512 MB) | Max upload size in bytes |
| `MIN_FREE_DISK_BYTES` | `2147483648` (2 GB) | Minimum free disk space before rejecting uploads |
| `MEM_LIMIT` | `8g` | Container memory limit |
| `HF_TOKEN` | — | Hugging Face token for gated models |

See [.env.example](.env.example) for all options.

### Model Recommendations by Hardware

| Hardware | Recommended Model | Compute Type |
|----------|-------------------|--------------|
| CPU only | tiny, base | int8 |
| 2 GB VRAM | tiny, base | float16/int8 |
| 4 GB VRAM | base, small | float32 or float16 |
| 8 GB VRAM | large-v3-turbo | float16, batch=16 |
| 12 GB+ VRAM | large-v3 | float16, batch=16 |

## GPU Compatibility

| Architecture | Compute Capability | Example GPUs | Compute Type |
|--------------|-------------------|--------------|--------------|
| Maxwell | 5.0 | GeForce 940MX, GTX 950M | float32 |
| Pascal | 6.0-6.1 | GTX 1050, GTX 1080 Ti, Tesla P100 | float16 |
| Volta | 7.0 | Tesla V100, GTX 1080 Ti | float16 |
| Turing | 7.5 | RTX 2060, RTX 2080 Ti, T4 | float16 |
| Ampere | 8.0-8.6 | A100, RTX 3060, RTX 3090 | float16 |
| Ada Lovelace | 8.9 | RTX 4060, RTX 4090 | float16 |
| Blackwell | 9.0 | B100, B200 | float16 |

> **Note on Maxwell GPUs (CC 5.0):** Older GPUs like the 940MX lack FP16 tensor cores. The app automatically detects this and uses `float32` compute type.

### Known Limitations

- **Maxwell (CC 5.0) CUDA inference:** The pre-built CTranslate2 pip binary (`faster-whisper`) does not include working SM 5.0 CUDA kernels. The model loads into GPU memory but inference falls back to CPU. To get true GPU acceleration on Maxwell GPUs, compile CTranslate2 from source with `-DCUDA_ARCH_LIST="5.0"` and `-DWITH_MKL=OFF -DOPENMP_RUNTIME=COMP`.
- **whisper.cpp alternative:** [whisper.cpp](https://github.com/ggerganov/whisper.cpp) supports CC 5.0 natively via GGML_CUDA. Build with `-DGGML_CUDA_ARCH_LIST="5.0"` for Maxwell GPUs. Requires manual compilation.

## Disk Safety

The app includes multiple safeguards for deployment on hosts with limited disk:

1. **tmpfs work dir** — `/tmp/whisper-stt` is a RAM-backed tmpfs volume capped at 2 GB.
   No host disk space is used for processing; all data is lost on restart.
2. **MAX_FILE_SIZE** — Default 512 MB upload limit prevents runaway files.
3. **MIN_FREE_DISK_BYTES** — Uploads are rejected when host free space drops below 2 GB.
4. **Periodic cleanup** — Stale job directories and chunk sessions older than 30 minutes
   are removed every 10 minutes.
5. **Startup cleanup** — All stale data is wiped when the container starts.

## Test Audio & Accuracy

The repo includes test audio fixtures from LibriSpeech (public domain) and an accuracy
benchmark script. See [test_audio/SOURCES.md](test_audio/SOURCES.md) for source details.

```bash
# Rebuild test fixtures (requires internet)
./scripts/build_test_audio.sh

# Run accuracy benchmark (app must be running)
BASE_URL=http://localhost:8561 ./scripts/run_accuracy_bench.sh
```

Current fixtures:
- **1min.flac** (~60s, ~1 MB) / **5min.flac** (~300s, ~5 MB) / **full.flac** (~480s, ~8 MB)
- Ground-truth transcripts: `*_transcript.txt`
- Total: ~14 MB committed in git (no LFS needed)

## Benchmark Results

### GPU Benchmarks (older)

| GPU | Model | Compute | Time | Realtime | VRAM |
|-----|-------|---------|------|----------|------|
| RTX 3060 (8GB) | large-v3-turbo | fp16, batch=16 | 38.0s | 95x | ~1.3 GB |
| RTX 3060 (8GB) | large-v3-turbo | int8, batch=16 | 65.7s | 55x | ~2 GB |
| RTX 3060 (8GB) | small.en | fp16, batch=16 | 47.37s | 76x | ~2 GB |
| 940MX (4GB) | base | float32, batch=16 | 0.08s | 375x | ~0.3 GB |

### CPU Accuracy Benchmarks (LibriSpeech, base model)

See [docs/deploy-orc.md](docs/deploy-orc.md) or run the benchmark yourself.
Prior results with `base` model on CPU int8 show >95% word accuracy on
clean read speech (1min: 97.58%, 5min: 96.26%).

### Performance Notes

- **Batch processing:** `batch_size=16` provides significant speedup on GPUs with sufficient VRAM
- **VAD (Voice Activity Detection):** Skips silence, providing ~2x speedup on real-world audio
- **Model size vs quality:** `large-v3-turbo` offers best quality; `base` is fastest for quick transcription
- **CPU int8:** Extremely fast for `base` model (200x+ realtime on modern CPUs)

## Architecture

```
┌─────────────────────┐     ┌──────────────────────────────┐
│   Mobile Browser     │     │      FastAPI Server           │
│                      │     │                               │
│  ┌──────────────┐   │     │  ┌─────────────────────────┐  │
│  │ Audio Record  │   │────▶│  │  /api/transcribe        │  │
│  │ or File Upload│   │     │  │  (multipart upload)     │  │
│  └──────────────┘   │     │  └──────────┬──────────────┘  │
│                      │     │             │                  │
│  ┌──────────────┐   │     │  ┌──────────▼──────────────┐  │
│  │ Transcript   │◀──│◀────│  │  faster-whisper (CPU    │  │
│  │ Display      │   │     │  │  or GPU)                │  │
│  └──────────────┘   │     │  └─────────────────────────┘  │
└─────────────────────┘     └──────────────────────────────┘
```

## Supported Formats

Supports audio and video files including WAV, MP3, FLAC, OGG, M4A, AAC, WMA, OPUS, MP4, AVI, MOV, MKV, FLV, WMV, and any container format supported by FFmpeg.

## API

### `POST /api/transcribe`

Upload an audio or video file for transcription.

**Request:** `multipart/form-data`
- `file` — Audio or video file (required)
- `language` — Language code, e.g. `en`, `es`, `fr` (optional, defaults to `en`)

**Response:** `application/json`
```json
{
  "id": "abc123",
  "text": "Hello, this is a transcription...",
  "language": "en",
  "duration": 12.5,
  "process_time": 1.2,
  "realtime_factor": 10.4,
  "device": "cpu",
  "compute_type": "int8",
  "segments": [
    {"text": "Hello,", "t0": 0, "t1": 500},
    {"text": "this is a transcription...", "t0": 500, "t1": 2500}
  ]
}
```

### `GET /health`

Health check with device information.

```json
{
  "status": "ok",
  "model": "base",
  "device": "cpu",
  "compute_type": "int8",
  "compute_capability": 0
}
```

### `GET /api/models`

List available models and current device info.

### `GET /api/upload/config`

Get chunked upload configuration (chunk size, thresholds, allowed extensions).

### `POST /api/upload/start`

Start a chunked upload session for files larger than 50 MB.

### `POST /api/upload/chunk/{upload_id}`

Upload a single chunk.

### `POST /api/upload/finish/{upload_id}`

Finalize upload and begin transcription. Returns `{ job_id, status, progress }`;
poll `GET /api/transcribe/status/{job_id}` for the result.

### `GET /api/transcribe/status/{job_id}`

Poll the status of a chunked-upload transcription job. Jobs expire from the
in-memory store 30 minutes after creation.

## License

MIT — use it however you like.
