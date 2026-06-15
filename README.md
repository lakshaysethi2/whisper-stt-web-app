# Whisper STT Web App

A self-hosted speech-to-text web application powered by [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2). Upload audio files or record directly from your microphone — transcription runs on your server's GPU.

> **GPU Support:** Supports NVIDIA GPUs across all CUDA compute capabilities (5.0+). Auto-detects GPU architecture and selects optimal compute type.

## Features

- **Live recording** — Record audio directly in the browser
- **File upload** — Upload MP3, WAV, M4A, FLAC, OGG, and more
- **GPU-accelerated** — Runs on NVIDIA GPU with CUDA for maximum speed
- **Auto-detection** — Automatically selects optimal compute type (float16/float32) based on GPU
- **Mobile-first PWA** — Install on your phone like a native app
- **Private** — Self-hosted, your audio never leaves your server
- **Docker-ready** — One command to deploy with GPU support

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

> **Note on Maxwell GPUs (CC 5.0):** Older GPUs like the 940MX lack FP16 tensor cores. The app automatically detects this and uses `float32` compute type. Performance will be lower than modern GPUs but functional with GPU acceleration.

## Benchmark Results

### Test Environment
- **Audio:** Synthetic test audio (15-30 seconds, 16kHz mono WAV)
- **Default model:** base (74M params)

### GPU Benchmarks

| GPU | Model | Compute | Time | Realtime | VRAM |
|-----|-------|---------|------|----------|------|
| RTX 3060 (8GB) | large-v3-turbo | fp16, batch=16 | 38.0s | 95x | ~1.3 GB |
| RTX 3060 (8GB) | large-v3-turbo | int8, batch=16 | 65.7s | 55x | ~2 GB |
| RTX 3060 (8GB) | small.en | fp16, batch=16 | 47.37s | 76x | ~2 GB |
| 940MX (4GB) | base | float32, batch=16 | 0.08s | 375x | ~0.3 GB |

### Performance Notes

- **Batch processing:** `batch_size=16` provides significant speedup on GPUs with sufficient VRAM
- **VAD (Voice Activity Detection):** Skips silence, providing ~2x speedup on real-world audio
- **Model size vs quality:** `large-v3-turbo` offers best quality; `base` is fastest for quick transcription
- **940MX specific:** Uses `float32` compute type due to lack of FP16 tensor cores. Uses ~338 MiB VRAM for base model.

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
- **NVIDIA GPU** with 2 GB+ VRAM (recommended) or CPU fallback (slow)
- CUDA toolkit + cuDNN (for GPU mode)
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
| `WHISPER_MODEL` | `base` | Model to use (tiny, base, small, medium, large-v3, large-v3-turbo) |
| `WHISPER_LANGUAGE` | `en` | Language for transcription |
| `MAX_FILE_SIZE` | `262144000` | Max upload size (250 MB) |

### Model Recommendations by VRAM

| VRAM | Recommended Model | Compute Type |
|------|-------------------|--------------|
| 2 GB | tiny, base | float16/int8 |
| 4 GB | base, small | float32 (Maxwell) or float16 |
| 8 GB | large-v3-turbo | float16, batch=16 |
| 12 GB+ | large-v3 | float16, batch=16 |

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
│  │ Display      │   │     │  │  Auto-detect compute    │  │
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
  "process_time": 1.2,
  "realtime_factor": 10.4,
  "device": "cuda",
  "compute_type": "float16",
  "segments": [
    {"text": "Hello,", "t0": 0, "t1": 500},
    {"text": "this is a transcription...", "t0": 500, "t1": 2500}
  ]
}
```

### `GET /api/models`

List available models and current device info.

### `GET /health`

Health check with device information.

## License

MIT — use it however you like.
