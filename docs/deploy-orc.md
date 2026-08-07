# Deploy on orc (Oracle A1 ARM VPS)

This guide covers deploying whisper-stt-web-app on `ubuntu@orc-4cpu.lak.nz` — an
**Oracle A1** ARM instance with **no NVIDIA GPU**, ~24 GB RAM, and tight disk
(~17 GB free, 91% used).

## Host environment

| Property | Value |
|----------|-------|
| CPU | ARM (Oracle A1, 4 cores) |
| RAM | ~24 GB |
| Disk | ~17 GB free (91% used) |
| GPU | None |
| OS | Ubuntu (ARM64) |

## Port

The web app listens on host port **8561** (→ container port 8000).
This is documented across compose, docs, and `.env.example`.

Cloudflare (captain-managed) points the domain to port 8561.

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
| `crisperwhisper-small` | ~1 GB (fp32) | ~0.5 GB | Verbatim ASR + word timestamps; lightest CrisperWhisper |
| `crisperwhisper-medium` | ~3 GB (fp32) | ~1.5 GB | Verbatim ASR; near-large quality |
| `crisperwhisper-turbo` | ~3.2 GB (fp32) | ~1.6 GB | Verbatim ASR; fast large option |
| `crisperwhisper-large` | ~6.2 GB (fp32) | ~3.1 GB | Verbatim ASR; best quality, heavy |

**Default is `base`** — best tradeoff for CPU inference on this host.

To switch:
```bash
WHISPER_MODEL=small docker compose up -d
```

### CrisperWhisper 2.0 on ARM (this host)

This host is ARM64, so CrisperWhisper runs on the **pure-PyTorch (`transformers`) backend**:

- The `[ct2]` extra is not installable on ARM64 (its wheels are Linux x86_64 only) and would conflict with faster-whisper anyway (upstream `ctranslate2` overwrites the fork's files).
- The app installs `crisperwhisper[transformers]`; `CRISPER_BACKEND=auto` (default) resolves to `transformers` automatically.
- The transformers backend loads with fp32 on CPU and uses eager attention for word timings — expect roughly realtime-factor 0.4 on `crisperwhisper-small` (measured on this A1), slower on larger models. CrisperWhisper jobs show "working, progress unknown" because the library exposes no incremental progress.
- The `Dockerfile.cpu` installs CPU-only `torch` first (from the PyTorch CPU index) so the `nvidia-*` CUDA wheel bundle is not pulled into the image; expect the image to grow by ~1.5-2 GB for torch + transformers + crisperwhisper.
- Model downloads land in the `hf-cache` volume (`/cache`). Watch disk: each CrisperWhisper model is 0.5-3.1 GB on disk.
- **UI/API**: choose a `crisperwhisper-*` model in the UI; a "Transcription style" selector appears (verbatim default, or `intended`). The API also accepts `mode=verbatim|intended` per job. Faster-whisper models ignore the mode.

## Disk management

Disk is tight. The app includes several safeguards:

1. **tmpfs work dir**: `/tmp/whisper-stt` is a `tmpfs` volume capped at 2 GB
   — data is lost on restart, never fills the host disk.
2. **MAX_FILE_SIZE**: Default **512 MB** upload limit (configurable via
   `MAX_FILE_SIZE` env var).
3. **MIN_FREE_DISK_BYTES**: Uploads rejected when WORK_DIR free drops below
   **512 MB** (keep below the tmpfs work-dir size; configurable via env var).
4. **Periodic cleanup**: Stale jobs and chunk sessions older than 30 minutes
   are automatically removed every 10 minutes. Transcripts (status.json) are
   kept for `JOB_RETENTION_SECONDS` (default 1 week); recordings are deleted
   earlier (`AUDIO_RETENTION_SECONDS`, default 30 min, and immediately on
   completion).
5. **Startup cleanup**: Only expired jobs/recordings are removed on start —
   completed transcripts survive restarts and are reloaded from disk.

> ⚠️ **tmpfs caveat**: `/tmp/whisper-stt` is a `tmpfs` volume (RAM-backed, 2 GB
> cap) — a container restart wipes **everything**, including completed
> transcripts, no matter how new they are. Transcripts only survive restarts if
> `WORK_DIR` points at real disk. If restart-survival matters for whisper-test,
> consider moving `WORK_DIR` to a persistent volume (deployment config change —
> not part of this PR).

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

## GPU deployment (not applicable to orc)

On a GPU-equipped host, use the GPU override:
```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

This:
- Builds from the CUDA-enabled `Dockerfile` (instead of `Dockerfile.cpu`)
- Sets `WHISPER_MODEL=large-v3-turbo` by default
- Adds NVIDIA GPU device reservation
- Enables `NVIDIA_VISIBLE_DEVICES=all`
