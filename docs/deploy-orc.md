# Deploy on a CPU / ARM VPS

This guide covers deploying whisper-stt-web-app on a CPU-only (no NVIDIA GPU)
Linux host, e.g. an ARM VPS. The steps apply to any host where
`Dockerfile.cpu` is the right image (ARM64 or x86_64 without a GPU).

## Host environment

| Property | Value |
|----------|-------|
| CPU | ARM or x86_64 (no GPU required) |
| OS | Linux (any distro with Docker) |

Choose a host with enough RAM for the model you plan to run — see the model
table below and the `MAX_FILE_SIZE` / `MIN_FREE_DISK_BYTES` safeguards for
disk sizing.

## Port

The web app listens on the host port configured by `HOST_PORT`
(default **8561**, mapped to container port 8000). This is documented across
compose, docs, and `.env.example`.

If you use a domain, point it (e.g. via your DNS provider or a reverse proxy
such as Cloudflare Tunnel) to the host port above.

## Quick start

```bash
# Clone
git clone https://github.com/lakshaysethi2/whisper-stt-web-app.git
cd whisper-stt-web-app

# Copy environment (edit if needed — defaults are production-safe)
cp .env.example .env

# Build and start (CPU-only variant, ARM-compatible)
docker compose up -d --build

# Check health
curl -sf http://localhost:8561/health

# View logs
docker compose logs -f
```

## Model choice

| Model | RAM use | Disk cache | Notes |
|-------|---------|------------|-------|
| `tiny` | ~200 MB | ~75 MB | Fastest, lower accuracy |
| `base` (default) | ~400 MB | ~150 MB | Good accuracy, recommended |
| `small` | ~1.2 GB | ~500 MB | Better accuracy, more RAM |
| `medium` | ~3.2 GB | ~1.5 GB | High accuracy |
| `large-v3-turbo` | ~3.4 GB | ~1.7 GB | Best accuracy, heavy |

**Default is `base`** — best tradeoff for CPU inference on this host.

To switch:
```bash
WHISPER_MODEL=small docker compose up -d
```

## Disk management

Disk is tight on typical VPS hosts. The app includes several safeguards:

1. **tmpfs work dir**: `/tmp/whisper-stt` is a `tmpfs` volume capped at 2 GB
   — data is lost on restart, never fills the host disk.
2. **MAX_FILE_SIZE**: Default **512 MB** upload limit (configurable via
   `MAX_FILE_SIZE` env var).
3. **MIN_FREE_DISK_BYTES**: Uploads rejected when host disk free drops below
   **2 GB** (configurable via env var).
4. **Periodic cleanup**: Stale jobs and chunk sessions older than 30 minutes
   are automatically removed every 10 minutes.
5. **Startup cleanup**: All stale job directories are cleaned on container start.

### Manual disk ops

```bash
# Check disk usage
df -h /

# Prune unused Docker data (safe; keeps recent build cache)
docker system prune -f --filter "until=24h"

# Restart container (wipes tmpfs work dir)
docker compose restart

# Full rebuild from scratch
docker compose down -v
docker compose up -d --build
```

## Running the accuracy benchmark

```bash
# Ensure app is up and healthy
curl -sf http://localhost:8561/health

# Run benchmark (may take a few minutes)
BASE_URL=http://localhost:8561 ./scripts/run_accuracy_bench.sh
```

> The benchmark uploads each `test_audio/*.flac`, transcribes it, and compares
> against the corresponding `*_transcript.txt` using word-level accuracy via
> Levenshtein distance. A markdown table is printed.

## Updating

```bash
git pull
docker compose up -d --build
```

## GPU deployment

On a GPU-equipped host, use the GPU override:
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

This:
- Builds from the CUDA-enabled `Dockerfile` (instead of `Dockerfile.cpu`)
- Sets `WHISPER_MODEL=large-v3-turbo` by default
- Adds NVIDIA GPU device reservation
- Enables `NVIDIA_VISIBLE_DEVICES=all`
