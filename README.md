# Whisper STT Web App

A self-hosted speech-to-text web application powered by [whisper.cpp](https://github.com/ggerganov/whisper.cpp). Upload audio files or record directly from your microphone — transcription runs on your server's GPU.

## Features

- 🎤 **Live recording** — Record audio directly in the browser
- 📁 **File upload** — Upload MP3, WAV, M4A, FLAC, OGG, and more
- 🚀 **GPU-accelerated** — Runs whisper.cpp with CUDA on your server GPU
- 📱 **Mobile-first PWA** — Install on your phone like a native app
- 🌐 **Multi-language** — Auto-detect or manually select from 99 languages
- ⚡ **Fast** — Async processing with real-time status updates
- 🔒 **Private** — Self-hosted, your audio never leaves your server
- 🐳 **Docker-ready** — One command to deploy

## Quick Start

### With Docker (recommended)

```bash
# Clone the repo
git clone https://github.com/yourusername/whisper-stt-web-app.git
cd whisper-stt-web-app

# Build and run
docker compose up --build
```

Open http://localhost:8000 in your browser.

### Manual Setup

**Requirements:** Python 3.10+, CMake, CUDA toolkit (for GPU support)

```bash
# Clone with submodules
git clone --recursive https://github.com/yourusername/whisper-stt-web-app.git
cd whisper-stt-web-app

# Build whisper.cpp
cd whisper.cpp
cmake -B build -DGGML_CUDA=ON
cmake --build build --config Release -j
cd ..

# Install Python dependencies
pip install -r requirements.txt

# Download a model (base is a good balance of speed/quality)
bash scripts/download-model.sh base

# Run the server
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Model Sizes

| Model  | Parameters | English-only | Multilingual | Required VRAM | Relative Speed |
|--------|-----------|-------------|-------------|---------------|----------------|
| tiny   | 39 M      | yes         | yes         | ~1 GB         | ~32x           |
| base   | 74 M      | yes         | yes         | ~1 GB         | ~16x           |
| small  | 244 M     | yes         | yes         | ~2 GB         | ~6x            |
| medium | 769 M     | yes         | yes         | ~5 GB         | ~2x            |
| large  | 1550 M    | no          | yes         | ~10 GB        | 1x             |

Use `WHISPER_MODEL=large` environment variable to change the model.

## Configuration

Environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| `WHISPER_MODEL` | `base` | Model size to use |
| `WHISPER_LANGUAGE` | (auto) | Force a specific language |
| `WHISPER_THREADS` | (auto) | Number of CPU threads |
| `MAX_FILE_SIZE` | `262144000` | Max upload size in bytes (250 MB) |
| `HOST` | `0.0.0.0` | Server bind address |
| `PORT` | `8000` | Server port |

## Architecture

```
┌─────────────────────┐     ┌──────────────────────────┐
│   Mobile Browser     │     │      FastAPI Server       │
│                      │     │                           │
│  ┌──────────────┐   │     │  ┌─────────────────────┐  │
│  │ Audio Record  │   │────▶│  │  /api/transcribe    │  │
│  │ or File Upload│   │ WS  │  │  (multipart upload) │  │
│  └──────────────┘   │     │  └──────────┬──────────┘  │
│                      │     │             │              │
│  ┌──────────────┐   │     │  ┌──────────▼──────────┐  │
│  │ Transcript   │◀──│◀────│  │  whisper.cpp (GPU)  │  │
│  │ Display      │   │     │  │  via subprocess      │  │
│  └──────────────┘   │     │  └─────────────────────┘  │
└─────────────────────┘     └──────────────────────────┘
```

## Supported Audio Formats

WAV, MP3, FLAC, OGG, M4A, AAC, WMA, OPUS, and any format supported by FFmpeg.

## API

### `POST /api/transcribe`

Upload an audio file for transcription.

**Request:** `multipart/form-data`
- `file` — Audio file (required)
- `language` — Language code, e.g. `en`, `es`, `fr` (optional, auto-detect if omitted)

**Response:** `application/json`
```json
{
  "id": "abc123",
  "text": "Hello, this is a transcription...",
  "language": "en",
  "duration": 12.5,
  "segments": [...]
}
```

### `GET /api/status/{id}`

Check transcription status (for WebSocket fallback).

### `GET /api/models`

List available models.

## License

MIT — use it however you like.
