# Whisper STT Web App

A self-hosted speech-to-text web application powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2) **and** [CrisperWhisper 2.0](https://github.com/nyrahealth/CrisperWhisper) (verbatim ASR with word-level timestamps). Upload audio files or record directly from your microphone — transcription runs on your server.

- **CPU-friendly** — Default model is `base` (74M params). Runs efficiently on CPU with int8 quantization.
- **Verbatim ASR** — Optional CrisperWhisper 2.0 models transcribe exactly what was said, including `[um]`, repeats and stutters, with word-level timestamps; `intended` mode returns the clean version.
- **GPU optional** — Activate GPU support with a compose override for NVIDIA GPUs.
- **ARM compatible** — Dockerfile.cpu targets ARM64 (Oracle A1, Raspberry Pi) and x86_64.
- **Mobile-first PWA** — Install on your phone like a native app.

## Features

- **Live recording** — Record audio directly in the browser
- **File upload** — Upload audio (MP3, WAV, M4A, FLAC, OGG, etc.) or video (MP4, AVI, MOV, MKV, etc.) files
- **Verbatim transcription (CrisperWhisper 2.0)** — Word-for-word ASR with fillers/repetitions preserved, plus word-level timestamps; `intended` mode gives clean readable text
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
| `WHISPER_MODEL` | `base` | Model to use (tiny, base, small, medium, large-v3, large-v3-turbo, crisperwhisper-large/turbo/medium/small) |
| `WHISPER_LANGUAGE` | `en` | Language for transcription |
| `CRISPER_BACKEND` | `auto` | CrisperWhisper runtime: `auto` (prefer ct2 when the `[ct2]` extra is installed, else transformers), `ct2`, or `transformers` |
| `CRISPER_MODE` | `verbatim` | Default CrisperWhisper transcription mode when a request doesn't specify one: `verbatim` or `intended` |
| `HOST_PORT` | `8561` | Host port mapping |
| `MAX_FILE_SIZE` | `536870912` (512 MB) | Max upload size in bytes |
| `MIN_FREE_DISK_BYTES` | `536870912` (512 MB) | Minimum free space in WORK_DIR before rejecting uploads (keep below the tmpfs size) |
| `JOB_RETENTION_SECONDS` | `604800` (1 week) | How long transcription **results** (status.json) are kept |
| `AUDIO_RETENTION_SECONDS` | `1800` (30 min) | How long the uploaded **recording** is kept before deletion |
| `MEM_LIMIT` | `8g` | Container memory limit |
| `HF_TOKEN` | — | Hugging Face token for gated models |

See [.env.example](.env.example) for all options.

### Model Recommendations by Hardware

| Hardware | Recommended Model | Compute Type |
|----------|-------------------|--------------|
| CPU only | tiny, base, crisperwhisper-small | int8 / fp32 |
| 2 GB VRAM | tiny, base | float16/int8 |
| 4 GB VRAM | base, small | float32 or float16 |
| 8 GB VRAM | large-v3-turbo, crisperwhisper-turbo | float16, batch=16 |
| 12 GB+ VRAM | large-v3, crisperwhisper-large | float16, batch=16 |

### Mandatory Model Choice

Users must **explicitly choose** a model before transcribing. The UI offers two families:

| Model | Tradeoff |
|-------|----------|
| **base** | Faster, lighter on CPU/RAM, OK for drafts/short audio; **lower** transcript quality. |
| **large-v3-turbo** | Much better quality (recommended for best accuracy); slower and heavier on CPU/RAM; long files take longer. |
| **crisperwhisper-small** | Verbatim word-for-word ASR (fillers/repeats preserved), word-level timestamps; ~0.5 GB download; lighter of the CrisperWhisper set. |
| **crisperwhisper-medium** | Verbatim word-for-word; near-large quality; ~1.5 GB download. |
| **crisperwhisper-turbo** | Verbatim word-for-word; fastest large option; ~1.6 GB download. |
| **crisperwhisper-large** | Verbatim word-for-word; best open quality; ~3.1 GB download; heaviest. |

- No pre-selected default — the user must actively pick.
- The server rejects transcribe/finish requests with a `400` error if `model` is missing or invalid.
- The server shows the runtime device (cpu vs cuda) so the user understands server load.
- Models are loaded on demand with lazy caching: switching model in the UI loads it into memory.

## CrisperWhisper 2.0 (verbatim ASR)

[CrisperWhisper 2.0](https://github.com/nyrahealth/CrisperWhisper) (models: `nyralabs/CrisperWhisper2.0_{large,turbo,medium,small}`) is a Whisper-class ASR that makes the verbatim-vs-intended choice explicit and controllable:

- **Verbatim (default)** — exactly what was said: `[um] so we we need to, to reschedule the th- thursday meeting`
- **Intended** — the clean version the speaker meant: `So we need to reschedule the Thursday meeting`
- **Word-level timestamps** — every result includes per-word `t0`/`t1` (ms) alongside the usual segments.
- Long audio (>30 s) is handled automatically by the library (continuation longform); the backend serializes inference on a single dedicated thread (upstream requirement).

### Choosing verbatim vs intended

- Pick **verbatim** when the raw spoken audio matters: clinical notes, disfluency analysis, dataset construction, anything where fillers/stutters/cut-offs are signal.
- Pick **intended** when you want clean, readable text: meeting minutes, subtitles, general transcription.
- The UI shows a "Transcription style" selector whenever a `crisperwhisper-*` model is chosen; the API accepts a `mode=verbatim|intended` parameter per job (defaults to `verbatim`, or `CRISPER_MODE` if set). Faster-whisper models ignore the mode.

### Backend / GPU-CPU tradeoffs

- This app installs `crisperwhisper[transformers]` (pure PyTorch) **on purpose**:
  - The `[ct2]` extra conflicts with faster-whisper — faster-whisper depends on upstream `ctranslate2`, which overwrites the `ctranslate2-crisperwhisper` fork's files in site-packages.
  - The `[ct2]` wheels are Linux **x86_64 only**; audio.lak.nz is ARM64, so the transformers backend is the only installable option there.
- `CRISPER_BACKEND=auto` (default) picks the ct2 fork only when it is actually installed (`ctranslate2-crisperwhisper` distribution present); otherwise it uses transformers. To force a backend, set `CRISPER_BACKEND=ct2|transformers`.
- The transformers backend runs on CPU with fp32 (best speed on ARM) and on CUDA with fp16. It loads with eager attention (required for word-timing cross-attention), so it is slower than the ct2 runtime (~4-5x difference on GPU) but fully functional: verbatim/intended, word timestamps, longform and hallucination repair all work.
- On the live CPU-only host, expect roughly realtime-factor 0.4 on `crisperwhisper-small` (measured on a 4-core ARM64 Oracle A1); larger models are proportionally slower. CrisperWhisper jobs show "working, progress unknown" in the UI because the library does not expose incremental progress.

### Model downloads & disk

| Choice | HuggingFace repo | Download | RAM (CPU fp32) |
|--------|------------------|----------|----------------|
| crisperwhisper-small | `nyralabs/CrisperWhisper2.0_small` | ~0.5 GB | ~1 GB |
| crisperwhisper-medium | `nyralabs/CrisperWhisper2.0_medium` | ~1.5 GB | ~3 GB |
| crisperwhisper-turbo | `nyralabs/CrisperWhisper2.0_turbo` | ~1.6 GB | ~3.2 GB |
| crisperwhisper-large | `nyralabs/CrisperWhisper2.0_large` | ~3.1 GB | ~6.2 GB |

Models are downloaded to the HF cache volume (`hf-cache` → `/cache` in the container; `HF_HOME` elsewhere) on first use. The CrisperWhisper model weights are under the [Nyra Health Non-Commercial Research License](https://huggingface.co/nyralabs/CrisperWhisper2.0_large/blob/main/LICENSE.md) (free for research/non-commercial use; commercial licensing available).

### Browser-based STT (third-party)

The app includes a UI section linking to free browser-based Whisper demos that run entirely on your device:

- [Xenova/whisper-web](https://huggingface.co/spaces/Xenova/whisper-web)
- [Xenova/whisper-webgpu](https://huggingface.co/spaces/Xenova/whisper-webgpu)

These run on the client machine (not this server). Large files may struggle on phones.

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
3. **MIN_FREE_DISK_BYTES** — Uploads are rejected when WORK_DIR free space drops below 512 MB (keep this below the tmpfs work-dir size).
4. **Periodic cleanup** — Stale job directories and chunk sessions older than 30 minutes
   are removed every 10 minutes.
5. **Startup cleanup** — Only **expired** jobs/recordings are removed; completed transcripts
   survive restarts and are reloaded from disk. (Exception: if `WORK_DIR` is a tmpfs volume,
   a container restart wipes everything, transcripts included.)

### Job retention (transcripts vs recordings)

Retention is **split**: the transcription result is worth keeping, the recording is not.

- **Transcripts** (`status.json` with the full result text) are kept for
  `JOB_RETENTION_SECONDS` (default **1 week**, requirement ≥ 1 h). Completed/failed jobs
  are persisted to disk and reloaded on restart, so bookmarkable `/j/{job_id}` links keep
  working across restarts.
- **Recordings** (the uploaded input audio) are deleted **sooner**: immediately when
  transcription finishes, and no later than `AUDIO_RETENTION_SECONDS` (default **30 min**)
  for abandoned/crashed jobs.
- **Interrupted jobs**: `status.json` is written at job creation, so a job killed by a
  restart/crash is restored as `"interrupted"` (with `recording_retained` and re-upload
  guidance) instead of a misleading "expired". Only a job whose directory no longer exists
  is reported as expired.

⚠️ **tmpfs caveat**: on deployments where `WORK_DIR=/tmp/whisper-stt` is a RAM-backed
`tmpfs` volume (see [docs/deploy-orc.md](docs/deploy-orc.md)), a container restart wipes
**all** jobs — transcripts included — because the volume itself is volatile. Point
`WORK_DIR` at a real disk path to make transcripts survive restarts.

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
- **940MX specific:** Uses `float32` compute type due to lack of FP16 tensor cores. Uses ~338 MiB VRAM for base model.

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
- `model` — Whisper model name, `base`, `large-v3-turbo`, or `crisperwhisper-{large,turbo,medium,small}` (required, no default)
- `mode` — CrisperWhisper transcription mode, `verbatim` or `intended` (optional; defaults to `verbatim`; ignored by faster-whisper models)

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

CrisperWhisper results add `mode` (`verbatim`/`intended`), `backend` (`ct2`/`transformers`), a top-level `words` list (`{word, t0, t1}` in ms), and a per-segment `words` array; `text`/`segments` keep the same shape so the UI is unchanged.

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

Finalize upload and begin transcription. Query params: `language`, `model` (required), `mode` (`verbatim`|`intended`, optional, CrisperWhisper only). Returns `{ job_id, status, progress }`;
poll `GET /api/transcribe/status/{job_id}` for the result.

### `GET /api/transcribe/status/{job_id}`

Poll the status of a chunked-upload transcription job. `POST /api/upload/finish/{upload_id}`
returns a `job_id` immediately (well under Cloudflare's 100 s response timeout); poll this
endpoint every 2 seconds until `status` is `completed` or `failed`.

**Response:** `application/json`

```json
{
  "job_id": "abc12345",
  "status": "processing",
  "progress": 0.3,
  "elapsed_seconds": 12.4
}
```

When `status` is `"completed"`, a `result` field is included with the full transcription
(text, segments, language, duration, etc.). When `status` is `"failed"`, an `error` field
is included. When `status` is `"interrupted"`, the job was in flight when the server
restarted/crashed: `error` contains re-upload guidance and `recording_retained` says
whether the uploaded recording is still on disk. An `interrupted` job is **never** reported
as expired while its directory exists.

Jobs are kept for `JOB_RETENTION_SECONDS` (default 1 week); the uploaded recording is
deleted earlier (`AUDIO_RETENTION_SECONDS`, default 30 min, and immediately on completion).
Running jobs are never cleaned up. Job status is persisted to disk under
`WORK_DIR/<job_id>/status.json` so that completed/failed results survive container restart
(best-effort; on a tmpfs `WORK_DIR` a restart still wipes everything). The status response
includes `retention_seconds`, `expires_at` and `expires_in_seconds` so clients can show an
honest expiry.

### `GET /j/{job_id}`

SPA route for bookmarkable job pages. Serves the same `index.html` as `/`; the frontend
detect the job id from the URL path and resumes polling / displays results automatically.

### `GET /health`

Health check with device information.

## License

MIT — use it however you like.
