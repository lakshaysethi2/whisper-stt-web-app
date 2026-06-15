# Whisper STT Web App

A self-hosted speech-to-text web application powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2). Upload audio files or record directly from your microphone — transcription runs on your server's GPU.

## Features

- 🎤 **Live recording** — Record audio directly in the browser
- 📁 **File upload** — Upload MP3, WAV, M4A, FLAC, OGG, and more
- 🚀 **GPU-accelerated** — Runs on NVIDIA GPU with CUDA for maximum speed
- 📱 **Mobile-first PWA** — Install on your phone like a native app
- ⚡ **Fast** — 38s transcription for 1 hour of audio (95x realtime)
- 🔒 **Private** — Self-hosted, your audio never leaves your server
- 🐳 **Docker-ready** — One command to deploy with GPU support

## Benchmark Results

Tested on a **1-hour Hawkins lecture** (60 min, 16kHz mono WAV) with an **NVIDIA RTX 3060 (8GB VRAM)**:

### GPU Benchmarks

| Backend | Model | Config | Time | Realtime Speed | VRAM |
|---------|-------|--------|------|----------------|------|
| whisper.cpp CUDA | base.en | VAD + speed flags | 30.65s | 117x | ~1 GB |
| whisper.cpp CUDA | small.en | VAD + speed flags | 47.37s | 76x | ~2 GB |
| faster-whisper | large-v3-turbo | fp16, single | 97.8s | 37x | ~3 GB |
| faster-whisper | large-v3-turbo | int8, single | 65.7s | 55x | ~2 GB |
| faster-whisper | large-v3-turbo | fp16, batch=16 | 39.9s | 90x | ~3 GB |
| **faster-whisper** | **large-v3-turbo** | **int8, batch=16** | **38.0s** | **95x** | **~1.3 GB** |

### Winner: faster-whisper + large-v3-turbo + int8 + batch=16

- **38 seconds** for 1 hour of audio
- **Best quality** model (809M params, distilled from large-v3)
- **Lowest VRAM** at ~1.3 GB (int8 quantization)
- **No compilation needed** — pure Python, pip install

### What We Tested

We benchmarked both **whisper.cpp** (C++ with CUDA) and **faster-whisper** (Python with CTranslate2) across multiple configurations:

- **whisper.cpp** was originally built without CUDA (`GGML_CUDA=OFF`) — all transcription was CPU-only. After rebuilding with `GGML_CUDA=ON`, GPU acceleration kicked in and speeds improved dramatically.
- **faster-whisper** with batched inference (`BatchedInferencePipeline`) processes multiple audio segments simultaneously, achieving near-parallel speedup on GPU.
- **VAD (Voice Activity Detection)** skips silence, providing ~2x speedup on real-world audio with natural pauses.
- **int8 quantization** reduces VRAM usage by ~60% while maintaining quality, and is slightly faster than float16.

## Quick Start

### With Docker (recommended)

```bash
git clone https://github.com/lakshaysethi2/whisper-stt-web-app.git
cd whisper-stt-web-app
docker compose up --build
```

Open http://localhost:8000 in your browser.

### Manual Setup

**Requirements:**
- Python 3.10+
- NVIDIA GPU with **2 GB+ VRAM** (large-v3-turbo default) or 8 GB+ for larger models
- CUDA toolkit + cuDNN
- 4 GB+ system RAM

```bash
git clone https://github.com/lakshaysethi2/whisper-stt-web-app.git
cd whisper-stt-web-app
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./start.sh
```

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL` | `large-v3-turbo` | Model to use |
| `WHISPER_LANGUAGE` | `en` | Language for transcription |
| `MAX_FILE_SIZE` | `262144000` | Max upload size (250 MB) |

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
│  │ Transcript   │◀──│◀────│  │  faster-whisper (GPU)   │  │
│  │ Display      │   │     │  │  int8 + batch=16 + VAD  │  │
│  └──────────────┘   │     │  └─────────────────────────┘  │
└─────────────────────┘     └──────────────────────────────┘
```

## Supported Audio Formats

WAV, MP3, FLAC, OGG, M4A, AAC, WMA, OPUS, and any format supported by FFmpeg.

## API

### `POST /api/transcribe`

Upload an audio file for transcription.

**Request:** `multipart/form-data`
- `file` — Audio file (required)
- `language` — Language code, e.g. `en`, `es`, `fr` (optional, defaults to `en`)

**Response:** `application/json`
```json
{
  "id": "abc123",
  "text": "Hello, this is a transcription...",
  "language": "en",
  "duration": 12.5,
  "segments": [
    {"text": "Hello,", "t0": 0, "t1": 500},
    {"text": "this is a transcription...", "t0": 500, "t1": 2500}
  ]
}
```

### `GET /api/models`

List available models.

## License

MIT — use it however you like.
