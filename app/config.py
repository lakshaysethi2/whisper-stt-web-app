import os
import shutil
from pathlib import Path

WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")
WHISPER_LANGUAGE = os.getenv("WHISPER_LANGUAGE", "en")

# --- CrisperWhisper 2.0 (verbatim ASR) ---
# Backend selection for CrisperWhisper models: "auto" (prefer ct2 when the
# crisperwhisper[ct2] extra is installed, else transformers), "ct2", or
# "transformers".
# NOTE: on this stack the [ct2] extra cannot be used alongside faster-whisper
# (faster-whisper depends on upstream ctranslate2, which overwrites the
# ctranslate2-crisperwhisper fork's files) and its wheels are Linux x86_64
# only (audio.lak.nz is ARM64), so "auto" resolves to transformers here.
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
MAX_FILE_SIZE = int(os.getenv("MAX_FILE_SIZE", "536870912"))   # 512 MB (default; safer for tight-disk VPS)
MIN_FREE_DISK_BYTES = int(os.getenv("MIN_FREE_DISK_BYTES", "2147483648"))  # 2 GB minimum free space
JOB_RETENTION_SECONDS = int(os.getenv("JOB_RETENTION_SECONDS", "7200"))  # 2 hours default

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
    d = WORK_DIR / job_id
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)


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
    if WORK_DIR.exists():
        for p in WORK_DIR.iterdir():
            # Preserve the chunked-upload sessions directory; it is cleaned
            # by its own logic in app.main.
            if p.is_dir() and p.name != "chunks":
                shutil.rmtree(p, ignore_errors=True)
            elif p.is_file():
                p.unlink(missing_ok=True)

