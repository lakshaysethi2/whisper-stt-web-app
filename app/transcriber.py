import asyncio
import logging
import os
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from faster_whisper import WhisperModel, BatchedInferencePipeline

from app.config import WHISPER_MODEL, CRISPER_MODEL_IDS, PARAKEET_MODEL_IDS, CRISPER_BACKEND, is_crisper_model, is_parakeet_model

logger = logging.getLogger(__name__)

# CrisperWhisper's load_audio uses soundfile first and only falls back to
# librosa if installed. soundfile cannot read MP4/M4A/MOV containers, and we
# deliberately avoid pulling librosa/numba. Pre-decode with ffmpeg (already in
# the image) to 16 kHz mono WAV — the format Whisper models expect.
_CRISPER_WAV_SUFFIXES = {".wav"}
_CRISPER_SAMPLE_RATE = 16000

# Multi-model cache: model_name -> (WhisperModel, BatchedInferencePipeline)
_models: dict[str, tuple[WhisperModel, BatchedInferencePipeline]] = {}
_device_info = {}

# Shared progress dict for real-time transcription progress tracking.
# Keyed by job_id, value is progress float (0.0 to 1.0) or None when unknown.
_progress: dict[str, float | None] = {}

# --- CrisperWhisper 2.0 backend (verbatim ASR with word-level timestamps) ---
# crisperwhisper is imported lazily so the app still boots (and faster-whisper
# keeps working) if the extra is not installed.
CrisperWhisperModel = None
_crisper_import_error: Exception | None = None
try:
    from crisperwhisper import CrisperWhisperModel  # noqa: E402
except Exception as e:  # pragma: no cover - only on minimal installs
    _crisper_import_error = e
    logger.warning("crisperwhisper not importable; CrisperWhisper models unavailable: %s", e)

# Upstream serving note: load AND run CrisperWhisper models on one dedicated
# thread (the ct2 recovery primitives are affine to the creating thread; this
# also serializes inference).
_crisper_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="crisperwhisper")
# app model name -> CrisperWhisperModel instance
_crisper_models: dict[str, object] = {}
_resolved_crisper_backend: str | None = None

# --- Parakeet TDT 0.6B v3 backend (FastConformer-TDT, transformers) ---
_parakeet_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="parakeet")
_parakeet_models: dict[str, object] = {}  # name -> (model, processor)
_parakeet_import_error: Exception | None = None
try:
    # transformers >=4.57 puts Parakeet TDT under AutoModelForTDT; on older installs
    # (e.g. 4.56) the class is still present as ParakeetForTDT.
    try:
        from transformers import AutoModelForTDT, AutoProcessor  # noqa: E402 # type: ignore
    except ImportError:
        from transformers.models.parakeet.modeling_parakeet import ParakeetForTDT as AutoModelForTDT  # type: ignore # noqa: E402
        from transformers import AutoProcessor  # type: ignore # noqa: E402
    import torch  # noqa: E402 # type: ignore
    import soundfile as sf  # noqa: E402 # type: ignore
    import numpy as np  # noqa: E402 # type: ignore
    _parakeet_available = True
except Exception as e:  # pragma: no cover - minimal installs
    AutoModelForTDT = AutoProcessor = None  # type: ignore
    torch = None  # type: ignore
    sf = None  # type: ignore
    np = None  # type: ignore
    _parakeet_available = False
    _parakeet_import_error = e
    logger.warning("parakeet (transformers) not importable; Parakeet models unavailable: %s", e)


def _get_gpu_info() -> dict:
    """Query nvidia-smi for GPU name, memory and compute capability."""
    import subprocess
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,compute_cap", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            line = result.stdout.strip().split("\n")[0]
            parts = line.rsplit(",", 2)
            cc = parts[2].strip()
            try:
                major, minor = cc.split(".")
                cc_int = int(major) * 10 + int(minor)
            except (ValueError, IndexError):
                cc_int = None
            return {
                "name": parts[0].strip(),
                "vram_mb": int(parts[1].strip()),
                "cc": cc_int,
            }
    except (subprocess.TimeoutExpired, FileNotFoundError, IndexError, ValueError):
        pass
    return {}


def _gpu_name_to_cc(name: str) -> int:
    """Map GPU name to approximate compute capability (fallback when the
    driver's compute_cap query is unavailable)."""
    name_lower = name.lower()
    if any(x in name_lower for x in ["940mx", "950m", "960m", "940m", "gtx 9", "gtx9"]):
        return 50
    if any(x in name_lower for x in ["mx330", "mx150", "mx250", "mx350", "p100", "p40", "p4", "gtx 10", "gtx10", "tesla p"]):
        return 61
    if any(x in name_lower for x in ["v100", "tesla v"]):
        return 70
    if any(x in name_lower for x in ["rtx 20", "rtx20", "t4", "quadro rtx"]):
        return 75
    if any(x in name_lower for x in ["a100", "a30", "a10", "rtx 30", "rtx30", "l4"]):
        return 80
    if any(x in name_lower for x in ["rtx 40", "rtx40", "l40", "l40s"]):
        return 89
    if any(x in name_lower for x in ["b100", "b200", "b40"]):
        return 90
    return 70


def _detect_device() -> tuple[str, str, int]:
    """Detect the best device and compute type for the available GPU.

    Returns (device, compute_type, compute_capability_x10).
    Compute capability is reported as integer (e.g., 50 for CC 5.0).
    """
    try:
        import ctypes
        ctypes.CDLL("libcuda.so.1")
    except OSError:
        try:
            import ctypes
            ctypes.CDLL("libcuda.so")
        except OSError:
            logger.info("No CUDA library found, using CPU int8")
            return "cpu", "int8", 0

    gpu_info = _get_gpu_info()
    if not gpu_info:
        logger.warning("CUDA found but nvidia-smi failed, falling back to CPU int8")
        return "cpu", "int8", 0

    device_name = gpu_info["name"]
    vram_mb = gpu_info["vram_mb"]
    cc_int = gpu_info.get("cc") or _gpu_name_to_cc(device_name)
    cc_major = cc_int // 10
    cc_minor = cc_int % 10

    logger.info("GPU: %s (CC %d.%d, %d MB VRAM)", device_name, cc_major, cc_minor, vram_mb)

    if cc_major >= 7:
        compute_type = "float16"
    elif cc_major >= 5:
        compute_type = "float32"
    else:
        logger.warning("GPU CC %d.%d too old for CUDA kernels, falling back to CPU", cc_major, cc_minor)
        return "cpu", "int8", cc_int

    return "cuda", compute_type, cc_int


def get_loaded_models() -> list[str]:
    """Return list of currently loaded model names."""
    return list(_models.keys()) + list(_crisper_models.keys()) + list(_parakeet_models.keys())


def get_crisper_backend_info() -> dict:
    """Report the configured and actually-resolved CrisperWhisper backend."""
    return {"configured": CRISPER_BACKEND, "resolved": _resolved_crisper_backend}


def _ct2_fork_installed() -> bool:
    """True when the ctranslate2-crisperwhisper fork (not upstream ctranslate2)
    is the installed CTranslate2. faster-whisper ships upstream ctranslate2,
    which conflicts with the fork, so 'auto' must not blindly prefer ct2."""
    try:
        import importlib.metadata
        importlib.metadata.distribution("ctranslate2-crisperwhisper")
        return True
    except importlib.metadata.PackageNotFoundError:
        return False


def _resolve_crisper_backend_choice() -> str:
    """Map CRISPER_BACKEND to the concrete backend the library must use.

    'auto' prefers the ct2 fork when it is actually installed, otherwise
    transformers. We never pass 'auto' straight through: the library's own
    auto-resolution picks ct2 whenever any ctranslate2 (including upstream,
    pulled in by faster-whisper) is importable, which would load a broken
    combo here.
    """
    choice = CRISPER_BACKEND
    if choice not in ("auto", "ct2", "transformers"):
        logger.warning("Unknown CRISPER_BACKEND %r; using 'auto'", choice)
        choice = "auto"
    if choice == "auto":
        return "ct2" if _ct2_fork_installed() else "transformers"
    return choice


def _load_crisper_model_sync(model_name: str) -> None:
    """Load a CrisperWhisper model. Runs on the dedicated crisper thread so
    the model is created and later used on the same thread (upstream serving
    requirement for the ct2 recovery primitives)."""
    global _resolved_crisper_backend
    if model_name in _crisper_models:
        return
    if CrisperWhisperModel is None:
        raise RuntimeError(
            "crisperwhisper is not installed; install it with "
            "`pip install 'crisperwhisper[transformers]'` (CPU) or "
            "`pip install 'crisperwhisper[ct2]'` (Linux x86_64 GPU)"
            + (f" (import error: {_crisper_import_error})" if _crisper_import_error else "")
        )
    hf_id = CRISPER_MODEL_IDS[model_name]
    backend = _resolve_crisper_backend_choice()
    device = _device_info.get("device", "cpu")
    # Torch CUDA 13 wheels have no sm_50/sm_61 kernels (need >=sm_75); fall back
    # to CPU for the pure-PyTorch transformers backend on these Maxwell/Pascal
    # workers (CTranslate2 faster-whisper still uses the source-built sm_50 lib).
    cc = _device_info.get("compute_capability", 0)
    if device == "cuda" and cc and cc < 75 and backend != "ct2":
        logger.warning("CrisperWhisper CC %d < 75: torch has no kernel, using CPU", cc)
        device = "cpu"
    if device == "cuda":
        device_arg, compute_type = "cuda", "float16"
    else:
        device_arg = "cpu"
        # ct2 quantizes on CPU with int8_float16; the transformers engine maps
        # float32/int8 to fp32 (best on CPU, no fp16 NEON accel assumed).
        compute_type = "int8_float16" if backend == "ct2" else "float32"
    logger.info(
        "Loading CrisperWhisper %s (backend=%s, device=%s, compute_type=%s)...",
        hf_id, backend, device_arg, compute_type,
    )
    try:
        model = CrisperWhisperModel(
            hf_id,
            backend=backend,
            device=device_arg,
            compute_type=compute_type,
        )
    except Exception as e:
        msg = str(e).lower()
        if device_arg == "cuda" and ("no kernel image" in msg or "not compatible" in msg or "invalid device" in msg):
            logger.warning("CrisperWhisper CUDA load failed (%s), retrying on CPU", e)
            device_arg, compute_type = "cpu", "float32"
            model = CrisperWhisperModel(hf_id, backend=backend, device=device_arg, compute_type=compute_type)
        else:
            raise
    _crisper_models[model_name] = model
    _resolved_crisper_backend = getattr(model, "backend", backend)
    logger.info("CrisperWhisper %s loaded (backend=%s).", hf_id, _resolved_crisper_backend)


def load_crisper_model(model_name: str) -> None:
    """Load a CrisperWhisper model on the dedicated crisper thread (blocking)."""
    _crisper_executor.submit(_load_crisper_model_sync, model_name).result()


def _prepare_wav_audio(audio_path: str) -> tuple[str, str | None]:
    """Ensure audio is a WAV CrisperWhisper/soundfile can decode.

    Returns (path_to_use, temp_path_to_delete_or_None).
    WAV inputs are passed through unchanged. Other containers (mp4, m4a, mov,
    webm, …) are decoded via ffmpeg to 16 kHz mono PCM WAV in a temp file.
    """
    suffix = Path(audio_path).suffix.lower()
    if suffix in _CRISPER_WAV_SUFFIXES:
        return audio_path, None

    fd, wav_path = tempfile.mkstemp(prefix="crisper_", suffix=".wav")
    os.close(fd)
    cmd = [
        "ffmpeg", "-y",
        "-i", audio_path,
        "-ac", "1",
        "-ar", str(_CRISPER_SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        wav_path,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except FileNotFoundError as e:
        try:
            os.unlink(wav_path)
        except OSError:
            pass
        raise RuntimeError(
            "ffmpeg is required to decode non-WAV audio for CrisperWhisper"
        ) from e
    except subprocess.TimeoutExpired:
        try:
            os.unlink(wav_path)
        except OSError:
            pass
        raise RuntimeError(
            f"ffmpeg timed out decoding {audio_path} for CrisperWhisper"
        ) from None

    if result.returncode != 0 or not os.path.isfile(wav_path) or os.path.getsize(wav_path) == 0:
        err = (result.stderr or result.stdout or "").strip()
        try:
            os.unlink(wav_path)
        except OSError:
            pass
        raise RuntimeError(
            f"ffmpeg failed to decode {audio_path} for CrisperWhisper"
            + (f": {err[-500:]}" if err else "")
        )

    logger.info(
        "CrisperWhisper: pre-decoded %s -> 16kHz mono WAV via ffmpeg",
        Path(audio_path).name,
    )
    return wav_path, wav_path


def _prepare_crisper_audio(audio_path: str) -> tuple[str, str | None]:
    """Compat alias — CrisperWhisper callers use the generic wav helper."""
    return _prepare_wav_audio(audio_path)


def _crisper_words_to_segments(words: list[dict]) -> list[dict]:
    """Group word-level timestamps into segment-sized chunks the existing UI
    already renders (text/t0/t1), keeping the per-word data inside."""
    segments: list[dict] = []
    cur: list[dict] = []

    def flush() -> None:
        if not cur:
            return
        segments.append({
            "text": " ".join(w["word"] for w in cur),
            "t0": cur[0]["t0"],
            "t1": cur[-1]["t1"],
            "words": list(cur),  # copy: cur is cleared after append
        })
        cur.clear()

    for w in words:
        cur.append(w)
        if len(cur) >= 10 or w["word"].rstrip().endswith((".", "!", "?", "…")):
            flush()
    flush()
    return segments


def _transcribe_crisper_sync(
    audio_path: str,
    language: str,
    job_id: str,
    model_name: str,
    mode: str,
) -> dict:
    """CrisperWhisper transcription; runs on the dedicated crisper thread.
    Returns the same transcript/segments/word-timestamps shape the app
    returns for faster-whisper jobs (segments get an extra per-word list)."""
    model = _crisper_models[model_name]
    backend = _resolved_crisper_backend or _resolve_crisper_backend_choice()
    device = _device_info.get("device", "cpu")
    compute_type = (
        "float16" if device == "cuda"
        else "int8_float16" if backend == "ct2" else "float32"
    )

    decode_path, temp_wav = _prepare_crisper_audio(audio_path)
    try:
        start = time.monotonic()
        try:
            cw = model.transcribe(
                decode_path,
                language=language or "en",
                mode=mode,
                word_timestamps=True,
            )
        except Exception as e:
            msg = str(e).lower()
            if "no kernel image" in msg or "not compatible" in msg:
                logger.warning("CrisperWhisper CUDA kernel missing, retrying as CPU job: %s", e)
                raise RuntimeError(f"CUDA kernel missing for this GPU ({e}); retry on CPU") from e
            raise
        elapsed = time.monotonic() - start

        logger.info(
            "CrisperWhisper transcription done in %.2fs device=%s model=%s mode=%s",
            elapsed, device, model_name, mode,
        )

        words: list[dict] = []
        for w in (getattr(cw, "words", None) or []):
            if w.start is None or w.end is None:
                continue
            words.append({
                "word": w.word,
                "t0": int(round(float(w.start) * 1000)),
                "t1": int(round(float(w.end) * 1000)),
            })

        duration = float(getattr(cw, "duration", 0.0) or 0.0)
        process_time = float(getattr(cw, "processing_time", 0.0) or 0.0)
        if process_time <= 0:
            process_time = elapsed
        result = {
            "id": job_id,
            "text": (getattr(cw, "text", "") or "").strip(),
            "language": getattr(cw, "language", None) or "en",
            "mode": mode,
            "model": model_name,
            "backend": backend,
            "device": device,
            "compute_type": compute_type,
            "segments": _crisper_words_to_segments(words),
            "words": words,
            "duration": duration,
            "process_time": round(process_time, 2),
        }
        if duration > 0 and process_time > 0:
            result["realtime_factor"] = round(duration / process_time, 1)
        return result
    finally:
        if temp_wav:
            try:
                os.unlink(temp_wav)
            except OSError:
                pass


def _load_parakeet_model_sync(model_name: str) -> None:
    """Load a Parakeet model (transformers). Runs on the dedicated parakeet thread."""
    if model_name in _parakeet_models:
        return
    if not _parakeet_available or AutoModelForTDT is None:
        raise RuntimeError(
            "parakeet (transformers) is not installed; install torch + transformers>=4.57 "
            "and soundfile (`pip install transformers soundfile torch accelerate`)"
            + (f" (import error: {_parakeet_import_error})" if _parakeet_import_error else "")
        )
    hf_id = PARAKEET_MODEL_IDS[model_name]
    device = _device_info.get("device", "cpu")
    # Parakeet via torch: fp16 on any CUDA device (Pascal's fp16 works in torch,
    # unlike CTranslate2's CC>=7 rule for faster-whisper), fp32 on CPU.
    # Use device_map="auto" behaviour manually.
    torch_device = "cuda" if device == "cuda" else "cpu"
    dtype = torch.float16 if torch_device == "cuda" else torch.float32
    # Keep single-GPU path simple; device_map is not needed for 600M on 2GB.
    logger.info("Loading Parakeet %s on %s (%s)...", hf_id, torch_device, dtype)
    try:
        processor = AutoProcessor.from_pretrained(hf_id, trust_remote_code=False)
        model = AutoModelForTDT.from_pretrained(
            hf_id, dtype=dtype, trust_remote_code=False
        )
        model.to(torch_device)
        model.eval()
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            raise RuntimeError(
                f"Parakeet {model_name} OOM on {torch_device}; try the lighter 'base' model. "
                f"({e})"
            ) from e
        raise
    _parakeet_models[model_name] = (model, processor)
    logger.info("Parakeet %s loaded on %s.", hf_id, torch_device)


def load_parakeet_model(model_name: str) -> None:
    """Load a Parakeet model on the dedicated parakeet thread (blocking)."""
    _parakeet_executor.submit(_load_parakeet_model_sync, model_name).result()


def _parakeet_words_to_segments(words: list[dict]) -> list[dict]:
    """Group word timestamps into segments (same shape as crisper helper)."""
    return _crisper_words_to_segments(words)


def _transcribe_parakeet_sync(
    audio_path: str,
    language: str,
    job_id: str,
    model_name: str,
) -> dict:
    """Parakeet transcription on the parakeet executor thread."""
    model, processor = _parakeet_models[model_name]
    device = _device_info.get("device", "cpu")
    torch_device = "cuda" if device == "cuda" else "cpu"
    compute_type = "float16" if torch_device == "cuda" else "float32"
    decode_path, temp_wav = _prepare_wav_audio(audio_path)
    try:
        # 16 kHz mono wav -> numpy float32 array
        if sf is None:
            raise RuntimeError("soundfile is required to load audio for Parakeet")
        data, sr = sf.read(decode_path, dtype="float32", always_2d=False)
        if sr != 16000:
            # ffmpeg already resampled, but double-check
            raise RuntimeError(f"expected 16kHz wav, got {sr} Hz")
        # processor handles long-form chunking; for >24min full-attn OOM,
        # the local-attn config (rel_pos_local_attn, [256,256]) covers 3h —
        # ponytail: leave the knob to the operator; default full-attn is fine for 1min fixtures.
        start = time.monotonic()
        # Parakeet auto-detects language; the `language` arg is ignored (kept for API parity).
        inputs = processor(data, sampling_rate=16000, return_tensors="pt")
        # move tensors to device/dtype
        inputs = {k: (v.to(torch_device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        # Some processors return attention_mask; ensure it's on device
        with torch.inference_mode():
            output = model.generate(**inputs, return_dict_in_generate=True)
            # decode with word timestamps if durations are available
            durations = getattr(output, "durations", None)
            sequences = getattr(output, "sequences", output)
            if durations is not None:
                try:
                    decoded, timestamps = processor.decode(sequences, durations=durations, skip_special_tokens=True)
                except Exception:
                    decoded = processor.batch_decode(sequences, skip_special_tokens=True)
                    timestamps = None
            else:
                decoded = processor.batch_decode(sequences, skip_special_tokens=True)
                timestamps = None
            if isinstance(decoded, list):
                text = (decoded[0] or "").strip() if decoded else ""
            else:
                text = (decoded or "").strip()
        elapsed = time.monotonic() - start
        logger.info("Parakeet transcription done in %.2fs device=%s model=%s", elapsed, torch_device, model_name)
        # Build words/segments from timestamp data when available; otherwise single segment.
        words: list[dict] = []
        if timestamps is not None:
            # timestamps shape varies by transformers version; normalize best-effort.
            try:
                # Often a list per sample of list of (word, t0, t1) or dicts.
                raw = timestamps[0] if isinstance(timestamps, (list, tuple)) and timestamps else []
                for tok in raw:
                    if isinstance(tok, dict):
                        # {"text": ..., "timestamp": (t0, t1)} or {"word": ..., "t0":..., "t1":...}
                        w = tok.get("word") or tok.get("text") or ""
                        t0 = tok.get("t0") or (tok.get("timestamp", [None, None])[0] if isinstance(tok.get("timestamp"), (list, tuple)) else None)
                        t1 = tok.get("t1") or (tok.get("timestamp", [None, None])[1] if isinstance(tok.get("timestamp"), (list, tuple)) else None)
                        if w and t0 is not None and t1 is not None:
                            words.append({"word": str(w), "t0": int(round(float(t0) * 1000)), "t1": int(round(float(t1) * 1000))})
                    elif isinstance(tok, (list, tuple)) and len(tok) >= 2:
                        # (word, start, end) or (word, (start,end))
                        if len(tok) == 3:
                            w, s, e = tok[0], tok[1], tok[2]
                        elif len(tok) == 2 and isinstance(tok[1], (list, tuple)):
                            w, (s, e) = tok[0], tok[1]
                        else:
                            continue
                        words.append({"word": str(w), "t0": int(round(float(s) * 1000)), "t1": int(round(float(e) * 1000))})
            except Exception:
                words = []
        # fallback duration estimate: use audio length when model doesn't report it
        duration = float(len(data) / 16000) if 'data' in locals() else 0.0
        segments: list[dict]
        if words:
            segments = _parakeet_words_to_segments(words)
        else:
            # single segment spanning full audio when word timings unavailable
            segments = [{"text": text, "t0": 0, "t1": int(round(duration * 1000))}] if text else []
        result = {
            "id": job_id,
            "text": text,
            "language": language or "auto",
            "model": model_name,
            "backend": "transformers",
            "device": torch_device,
            "compute_type": compute_type,
            "segments": segments,
            "words": words,
            "duration": duration,
            "process_time": round(elapsed, 2),
        }
        if duration > 0 and elapsed > 0:
            result["realtime_factor"] = round(duration / elapsed, 1)
        return result
    finally:
        if temp_wav:
            try:
                os.unlink(temp_wav)
            except OSError:
                pass


def load_model(model_name: str | None = None) -> None:
    """Load a specific model into the cache. If model_name is None, load WHISPER_MODEL."""
    global _device_info

    target = model_name or WHISPER_MODEL

    if is_parakeet_model(target):
        if target not in _parakeet_models:
            if not _device_info:
                device, compute_type, cc = _detect_device()
                _device_info.update({"device": device, "compute_type": compute_type, "compute_capability": cc})
            load_parakeet_model(target)
        return

    if is_crisper_model(target):
        if target not in _crisper_models:
            if not _device_info:
                device, compute_type, cc = _detect_device()
                _device_info.update({"device": device, "compute_type": compute_type, "compute_capability": cc})
            load_crisper_model(target)
        return

    if target in _models:
        logger.info("Model %s already loaded", target)
        return

    if not _device_info:
        device, compute_type, cc = _detect_device()
        _device_info.update({"device": device, "compute_type": compute_type, "compute_capability": cc})

    device = _device_info.get("device", "cpu")
    compute_type = _device_info.get("compute_type", "int8")

    logger.info("Loading model %s on %s (%s)...", target, device, compute_type)

    try:
        model = WhisperModel(
            target,
            device=device,
            compute_type=compute_type,
        )
    except RuntimeError as e:
        if "out of memory" in str(e).lower() and device == "cuda":
            logger.warning(
                "GPU OOM with %s on %s, falling back to CPU int8", target, compute_type
            )
            device, compute_type, cc = "cpu", "int8", 0
            _device_info = {"device": device, "compute_type": compute_type, "compute_capability": cc}
            model = WhisperModel(
                target,
                device=device,
                compute_type=compute_type,
            )
        else:
            raise

    batch_size = 16 if device == "cuda" else 1
    batched = BatchedInferencePipeline(model=model)
    _models[target] = (model, batched)
    logger.info("Model %s loaded on %s/%s (batch_size=%d).", target, device, compute_type, batch_size)


def get_progress(job_id: str) -> float | None:
    """Thread-safe read of real transcription progress for a job.
    Returns a float 0.0-1.0, or None if no progress data is available."""
    return _progress.get(job_id)


def _transcribe_sync(
    audio_path: str,
    language: str,
    job_id: str,
    model_name: str | None = None,
    mode: str = "verbatim",
) -> dict:
    """Synchronous transcription function that runs entirely in a thread.
    Consumes the segment generator inside the thread so the asyncio event loop
    is never blocked. Updates real progress in _progress dict during iteration."""
    target = model_name or WHISPER_MODEL

    if is_parakeet_model(target):
        if target not in _parakeet_models:
            load_model(target)
        return _parakeet_executor.submit(
            _transcribe_parakeet_sync, audio_path, language, job_id, target
        ).result()

    if is_crisper_model(target):
        if target not in _crisper_models:
            load_model(target)
        # Run on the dedicated crisper thread: the model was created there and
        # the upstream ct2 recovery primitives are affine to that thread.
        return _crisper_executor.submit(
            _transcribe_crisper_sync, audio_path, language, job_id, target, mode
        ).result()

    device = _device_info.get("device", "cpu")

    # Get or load the model
    if target not in _models:
        load_model(target)
    _model, batched = _models[target]

    start = time.monotonic()
    segments, info = batched.transcribe(
        audio_path,
        language=language or "en",
        batch_size=16 if device == "cuda" else 1,
        vad_filter=True,
        vad_parameters=dict(
            min_silence_duration_ms=500,
            speech_pad_ms=200,
        ),
        word_timestamps=False,
        condition_on_previous_text=True,
    )
    elapsed = time.monotonic() - start

    logger.info("Transcription done in %.2fs device=%s model=%s", elapsed, device, target)

    result = {"id": job_id, "segments": []}
    result["language"] = info.language
    result["device"] = _device_info.get("device", "unknown")
    result["compute_type"] = _device_info.get("compute_type", "unknown")
    result["model"] = target

    audio_duration = info.duration  # total audio duration in seconds
    full_text_parts = []
    total_duration = 0.0
    max_end = 0.0

    for seg in segments:
        text = seg.text.strip()
        if not text:
            max_end = max(max_end, seg.end)
            if audio_duration and audio_duration > 0:
                _progress[job_id] = min(1.0, max_end / audio_duration)
            continue
        full_text_parts.append(text)
        duration = seg.end - seg.start
        total_duration += duration
        max_end = max(max_end, seg.end)

        result["segments"].append({
            "text": text,
            "t0": int(seg.start * 1000),
            "t1": int(seg.end * 1000),
        })

        # Update real progress based on audio position
        if audio_duration and audio_duration > 0:
            _progress[job_id] = min(1.0, max_end / audio_duration)

    result["text"] = " ".join(full_text_parts)
    result["duration"] = total_duration
    result["process_time"] = round(elapsed, 2)

    if total_duration > 0:
        result["realtime_factor"] = round(total_duration / elapsed, 1)

    return result


async def transcribe_audio(
    audio_path: str,
    language: str = None,
    job_id: str = "unknown",
    model_name: str | None = None,
    mode: str = "verbatim",
) -> dict:
    """Transcribe audio in a thread. Never blocks the asyncio event loop.
    Optionally specify model_name to use a specific Whisper model, and mode
    (verbatim|intended) for CrisperWhisper models (ignored by faster-whisper).
    Returns the result dict once both inference and segment iteration complete."""
    target = model_name or WHISPER_MODEL
    if is_crisper_model(target) or is_parakeet_model(target):
        # CrisperWhisper and Parakeet do not report incremental audio-position
        # progress; signal "working, progress unknown" instead of a stuck 0%.
        _progress[job_id] = None
    else:
        _progress[job_id] = 0.0
    try:
        result = await asyncio.to_thread(
            _transcribe_sync,
            audio_path,
            language or "en",
            job_id,
            model_name,
            mode,
        )
        _progress[job_id] = 1.0
        return result
    finally:
        _progress.pop(job_id, None)
