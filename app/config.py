import json
import logging
import os
import shutil
import time
from pathlib import Path

logger = logging.getLogger(__name__)

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")

# Captain rule: GPU-node deployments must never transcribe on the main host's
# CPU (an ARM VPS CPU job once OOM-crashed the host). When this is false the
# dispatch layer refuses local transcription: jobs fail cleanly with "No
# transcription node available" when no GPU node can take them, instead of
# silently falling back to CPU. Set true for plain CPU-only single-host
# deployments (the docker-compose.yml default keeps the public quick start
# working).
WHISPER_ALLOW_LOCAL_CPU = os.getenv("WHISPER_ALLOW_LOCAL_CPU", "false").strip().lower() in ("1", "true", "yes", "on")

# --- CrisperWhisper 2.0 (verbatim ASR) ---
# Backend selection for CrisperWhisper models: "auto" (prefer ct2 when the
# crisperwhisper[ct2] extra is installed, else transformers), "ct2", or
# "transformers".
# NOTE: on this stack the [ct2] extra cannot be used alongside faster-whisper
# (faster-whisper depends on upstream ctranslate2, which overwrites the
# ctranslate2-crisperwhisper fork's files) and its wheels are Linux x86_64
# only (not installable on the reference ARM64 host), so "auto" resolves to transformers here.
CRISPER_BACKEND = os.getenv("CRISPER_BACKEND", "auto").strip().lower()
# Default transcription mode for CrisperWhisper models when a request does not
# specify one. "verbatim" preserves fillers/stutters; "intended" is clean text.
CRISPER_MODE = os.getenv("CRISPER_MODE", "verbatim").strip().lower()
CRISPER_MODES = ("verbatim", "intended")

# App-facing model choice -> exact HuggingFace repo the upstream supports.
# See https://github.com/nyrahealth/CrisperWhisper (models table).
CRISPER_MODEL_IDS = {
    "crisperwhisper-large": "nyralabs/CrisperWhisper2.0_large",
    "crisperwhisper-turbo": "nyralabs/CrisperWhisper2.0_turbo",
    "crisperwhisper-medium": "nyralabs/CrisperWhisper2.0_medium",
    "crisperwhisper-small": "nyralabs/CrisperWhisper2.0_small",
}
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "536870912"))   # 512 MB (default; safer for tight-disk hosts)
# Minimum free bytes in WORK_DIR before uploads are rejected. Must be well
# below the default tmpfs work-dir size (2 GB) or the check trips as soon as
# a single job dir exists.
MIN_FREE_DISK_BYTES = int(os.getenv("MIN_FREE_DISK_BYTES", "536870912"))  # 512 MB default
# Transcripts (status.json / result text) are kept for 1 week (captain requirement,
# supersedes the earlier 2h default). Minimum requirement: >= 1 hour.
JOB_RETENTION_SECONDS = int(os.getenv("JOB_RETENTION_SECONDS", "604800"))  # 1 week default
# Recordings (input audio) are deleted sooner than transcripts (default 30 min).
# The input is also removed immediately when a job completes/fails (cleanup_job_audio),
# so this TTL mainly covers abandoned/crashed jobs whose input was never cleaned up.
AUDIO_RETENTION_SECONDS = int(os.getenv("AUDIO_RETENTION_SECONDS", "1800"))  # 30 minutes default

WORK_DIR = Path(os.getenv("WORK_DIR", "/tmp/whisper-stt"))
WORK_DIR.mkdir(parents=True, exist_ok=True)


ALLOWED_EXTENSIONS = {
    # Audio formats
    ".mp3", ".wav", ".flac", ".ogg", ".m4a",
    ".aac", ".wma", ".opus",
    # Video formats
    ".mp4", ".webm", ".avi", ".mov", ".mkv",
    ".flv", ".wmv", ".mpeg", ".mpg", ".3gp",
    ".m4v", ".asf",
}

UI_MODEL_CHOICES = [
    # faster-whisper (CTranslate2) models — unchanged behaviour
    "base",
    "large-v3-turbo",
    # CrisperWhisper 2.0 (verbatim ASR with word-level timestamps) models
    "crisperwhisper-large",
    "crisperwhisper-turbo",
    "crisperwhisper-medium",
    "crisperwhisper-small",
]

SUPPORTED_MODELS = [
    {"name": "tiny", "params": "39M", "vram_fp32": "200", "vram_fp16": "128"},
    {"name": "base", "params": "74M", "vram_fp32": "400", "vram_fp16": "256"},
    {"name": "small", "params": "244M", "vram_fp32": "1200", "vram_fp16": "600"},
    {"name": "medium", "params": "769M", "vram_fp32": "3200", "vram_fp16": "1600"},
    {"name": "large-v3", "params": "1550M", "vram_fp32": "6200", "vram_fp16": "3100"},
    {"name": "large-v3-turbo", "params": "809M", "vram_fp32": "3400", "vram_fp16": "1700"},
    # CrisperWhisper 2.0: verbatim ASR (fillers/repetitions preserved) with
    # word-level timestamps. Disk sizes are the HF safetensors downloads;
    # RAM use is roughly 2x on CPU (fp32).
    {"name": "crisperwhisper-large", "params": "1550M", "vram_fp32": "6200", "vram_fp16": "3100", "disk_gb": "3.1", "family": "crisperwhisper"},
    {"name": "crisperwhisper-turbo", "params": "809M", "vram_fp32": "3400", "vram_fp16": "1700", "disk_gb": "1.6", "family": "crisperwhisper"},
    {"name": "crisperwhisper-medium", "params": "769M", "vram_fp32": "3200", "vram_fp16": "1600", "disk_gb": "1.5", "family": "crisperwhisper"},
    {"name": "crisperwhisper-small", "params": "244M", "vram_fp32": "1200", "vram_fp16": "600", "disk_gb": "0.5", "family": "crisperwhisper"},
]


def is_crisper_model(model_name: str) -> bool:
    """True if the (already validated) model choice is a CrisperWhisper model."""
    return model_name in CRISPER_MODEL_IDS


def validate_model(model_name: str | None) -> str:
    """Validate a model choice. Returns the validated model name or raises HTTPException(400)."""
    if not model_name or not isinstance(model_name, str) or not model_name.strip():
        from fastapi import HTTPException
        raise HTTPException(
            400,
            "Model choice is required. Choose one of: " + ", ".join(UI_MODEL_CHOICES),
        )
    model_name = model_name.strip()
    valid_names = UI_MODEL_CHOICES
    if model_name not in valid_names:
        from fastapi import HTTPException
        raise HTTPException(
            400,
            f"Invalid model: '{model_name}'. Valid choices: {', '.join(valid_names)}",
        )
    return model_name


def validate_mode(mode: str | None) -> str:
    """Validate a CrisperWhisper transcription mode ('verbatim'|'intended').

    Returns the validated mode, defaulting to CRISPER_MODE ('verbatim') when
    the request does not specify one. Faster-whisper models ignore the mode.
    """
    if mode is None or not str(mode).strip():
        return CRISPER_MODE if CRISPER_MODE in CRISPER_MODES else "verbatim"
    mode = str(mode).strip().lower()
    if mode not in CRISPER_MODES:
        from fastapi import HTTPException
        raise HTTPException(
            400,
            f"Invalid mode: '{mode}'. Valid choices: {', '.join(CRISPER_MODES)}",
        )
    return mode


def get_job_dir(job_id: str) -> Path:
    d = WORK_DIR / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def cleanup_job(job_id: str) -> None:
    """Remove the entire job directory. Used only on pre-registration error paths
    (invalid uploads, disk-pressure aborts) where nothing should be kept."""
    d = WORK_DIR / job_id
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)


def cleanup_job_audio(job_id: str) -> None:
    """Delete the uploaded recording for a job but KEEP status.json (the transcript).

    Called when a transcription finishes (completed or failed): the recording is the
    bulk of job disk usage and is no longer needed, while the result must remain
    available for resume until JOB_RETENTION_SECONDS expires.
    """
    d = WORK_DIR / job_id
    if not d.exists():
        return
    try:
        for p in list(d.iterdir()):
            if p.is_file() and p.name.startswith("input"):
                p.unlink(missing_ok=True)
    except OSError:
        pass


def job_created_at(job_dir: Path) -> float:
    """Best-effort creation timestamp for a job directory.

    Prefers `created_at` from status.json (written on completion/failure, so it is
    stable across restarts); falls back to the directory mtime.
    """
    try:
        sf = job_dir / "status.json"
        if sf.exists():
            data = json.loads(sf.read_text())
            created = data.get("created_at")
            if isinstance(created, (int, float)) and created > 0:
                return float(created)
    except Exception:
        pass
    try:
        return job_dir.stat().st_mtime
    except OSError:
        return 0.0


def cleanup_stale_input_audio(job_dir: Path, now: float | None = None) -> bool:
    """Delete input recording files older than AUDIO_RETENTION_SECONDS in a job dir.

    The transcript (status.json) is kept; only the recording is removed. Returns True
    if any file was deleted.
    """
    if now is None:
        now = time.time()
    deleted = False
    try:
        for f in list(job_dir.iterdir()):
            if f.is_file() and f.name.startswith("input"):
                try:
                    if now - f.stat().st_mtime > AUDIO_RETENTION_SECONDS:
                        f.unlink(missing_ok=True)
                        deleted = True
                except OSError:
                    pass
    except OSError:
        pass
    return deleted


def check_disk_space(path: str | Path = None) -> bool:
    """Return True if free disk at path is above MIN_FREE_DISK_BYTES."""
    target = Path(path or WORK_DIR)
    try:
        usage = shutil.disk_usage(str(target))
        return usage.free >= MIN_FREE_DISK_BYTES
    except OSError:
        # If we cannot check, assume safe (transient mount issue, etc.)
        return True


def cleanup_all_jobs() -> None:
    """Retention-aware startup cleanup — does NOT wipe everything.

    - Job directories older than JOB_RETENTION_SECONDS are removed entirely.
    - Input recordings older than AUDIO_RETENTION_SECONDS are removed while the
      transcript (status.json) is kept until the job itself expires.
    - The chunked-upload sessions directory ("chunks") is preserved.

    Completed/failed transcripts therefore survive restarts and are reloaded by
    app.main._load_persisted_jobs().
    """
    if not WORK_DIR.exists():
        return
    now = time.time()
    for p in list(WORK_DIR.iterdir()):
        if p.name == "chunks":
            continue
        if p.is_file():
            p.unlink(missing_ok=True)
            continue
        if not p.is_dir():
            continue
        try:
            age = now - job_created_at(p)
            if age > JOB_RETENTION_SECONDS:
                logger.info(
                    "Startup cleanup: removing expired job dir %s (age %.0fs)",
                    p.name, age,
                )
                shutil.rmtree(p, ignore_errors=True)
            elif cleanup_stale_input_audio(p, now):
                logger.info(
                    "Startup cleanup: removed stale input audio for job %s",
                    p.name,
                )
        except Exception:
            pass

