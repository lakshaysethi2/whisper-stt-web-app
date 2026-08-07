"""GPU worker dispatch for transcription jobs.

When WHISPER_WORKER_URLS is set (comma-separated base URLs of reachable
whisper worker instances, e.g. http://host.docker.internal:8564), the app
forwards transcription jobs for GPU-friendly models to those workers and
falls back to local CPU transcription whenever no worker is reachable or a
worker fails. With no workers configured the app behaves exactly as before
(always local).

The workers are plain instances of this same app (the GPU Dockerfile) —
dispatch reuses their existing /api/transcribe + /api/transcribe/status
job queue, so the result shape is identical and no worker-side code change
is needed.
"""
import asyncio
import logging
import os
import time

import httpx

from app.transcriber import _progress

logger = logging.getLogger(__name__)

# Comma-separated worker base URLs (empty = dispatch disabled, local only).
WHISPER_WORKER_URLS = [
    u.strip().rstrip("/")
    for u in os.getenv("WHISPER_WORKER_URLS", "").split(",")
    if u.strip()
]

# Models we may dispatch to the laptops' small-VRAM GPUs. Everything else
# transcribes locally on CPU. The laptop GPUs (Maxwell/Pascal) run float32
# (no fp16/int8 kernels on CC 5.0/6.1): tiny/base/small fit ~2 GB VRAM;
# larger models would OOM the worker and waste the upload round trip.
WHISPER_GPU_MODELS = {
    m.strip()
    for m in os.getenv("WHISPER_GPU_MODELS", "tiny,base,small,crisperwhisper-small").split(",")
    if m.strip()
}

# Worker request timeouts. The upload (home uplink through a reverse tunnel)
# gets a generous read budget; health/poll calls are short.
HEALTH_TIMEOUT = httpx.Timeout(connect=3.0, read=5.0, write=5.0, pool=3.0)
UPLOAD_TIMEOUT = httpx.Timeout(connect=10.0, read=3600.0, write=3600.0, pool=10.0)
POLL_TIMEOUT = httpx.Timeout(connect=5.0, read=15.0, write=15.0, pool=5.0)
POLL_INTERVAL_SECONDS = 1.2
# Consecutive poll failures (worker stopped responding) before giving up.
MAX_POLL_FAILURES = 10

_round_robin_index = 0


def should_dispatch(model_name: str) -> bool:
    """True when a job for this model should be attempted on a GPU worker."""
    return bool(WHISPER_WORKER_URLS) and model_name in WHISPER_GPU_MODELS


def _worker_order() -> list[str]:
    """Workers rotated round-robin so both laptops share the load."""
    global _round_robin_index
    if not WHISPER_WORKER_URLS:
        return []
    i = _round_robin_index % len(WHISPER_WORKER_URLS)
    _round_robin_index = (i + 1) % len(WHISPER_WORKER_URLS)
    return WHISPER_WORKER_URLS[i:] + WHISPER_WORKER_URLS[:i]


class RemoteTranscriptionError(RuntimeError):
    """A worker was unreachable, failed, or stopped responding mid-job."""


async def _check_worker(base: str, transport=None) -> bool:
    try:
        async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT, transport=transport) as client:
            r = await client.get(f"{base}/health")
        return r.status_code == 200
    except Exception:
        return False


async def _upload_and_poll(
    base: str, audio_path: str, language: str, model_name: str, mode: str, job_id: str,
    transport=None,
) -> dict:
    """Upload audio to one worker and poll its job to completion.

    Returns the worker's transcript result dict. Raises
    RemoteTranscriptionError on any worker-side failure so the caller can
    fall back to local CPU transcription.
    """
    if not await _check_worker(base, transport=transport):
        raise RemoteTranscriptionError(f"worker {base} unreachable")

    async with httpx.AsyncClient(timeout=UPLOAD_TIMEOUT, transport=transport) as client:
        with open(audio_path, "rb") as f:
            files = {"file": (os.path.basename(audio_path), f, "application/octet-stream")}
            data = {"language": language or "", "model": model_name, "mode": mode}
            r = await client.post(f"{base}/api/transcribe", files=files, data=data)
        if r.status_code != 200:
            raise RemoteTranscriptionError(
                f"worker {base} rejected upload (HTTP {r.status_code}): {r.text[:200]}"
            )
        worker_job_id = r.json()["job_id"]

    async with httpx.AsyncClient(timeout=POLL_TIMEOUT, transport=transport) as client:
        failures = 0
        while True:
            await asyncio.sleep(POLL_INTERVAL_SECONDS)
            try:
                r = await client.get(f"{base}/api/transcribe/status/{worker_job_id}")
                r.raise_for_status()
                failures = 0
            except Exception as e:
                failures += 1
                logger.warning(
                    "Poll worker %s job %s failed (%d/%d): %s",
                    base, worker_job_id, failures, MAX_POLL_FAILURES, e,
                )
                if failures >= MAX_POLL_FAILURES:
                    raise RemoteTranscriptionError(
                        f"worker {base} stopped responding"
                    ) from e
                continue

            st = r.json()
            status = st.get("status")
            if status == "processing":
                progress = st.get("progress")
                if isinstance(progress, (int, float)):
                    # Mirror the worker's real progress onto our job.
                    _progress[job_id] = min(0.99, float(progress))
                # Otherwise the worker reports "working, progress unknown" —
                # leave ours unknown too.
                continue
            if status == "completed":
                result = st.get("result")
                if not isinstance(result, dict) or not result:
                    raise RemoteTranscriptionError(
                        f"worker {base} returned empty result for job {worker_job_id}"
                    )
                return result
            if status == "failed":
                raise RemoteTranscriptionError(
                    f"worker {base} failed job {worker_job_id}: {st.get('error', 'unknown error')}"
                )
            if status == "interrupted":
                raise RemoteTranscriptionError(
                    f"worker {base} restarted mid-job (interrupted)"
                )
            raise RemoteTranscriptionError(
                f"worker {base} unexpected status {status!r}"
            )


async def transcribe_remote(
    audio_path: str,
    language: str,
    model_name: str,
    mode: str,
    job_id: str,
    _transport=None,  # test hook: httpx.MockTransport
) -> dict:
    """Transcribe via the configured GPU workers (round-robin).

    Raises RemoteTranscriptionError when every worker fails; the caller falls
    back to local CPU transcription in that case.
    """
    # Unknown until a worker reports real progress -> UI shows "working".
    _progress[job_id] = None
    start = time.monotonic()
    last_err: Exception | None = None
    try:
        for base in _worker_order():
            try:
                result = await _upload_and_poll(
                    base, audio_path, language, model_name, mode, job_id,
                    transport=_transport,
                )
                result = dict(result)
                # The worker's result carries its own job id — ours is the one
                # the user is polling.
                result["id"] = job_id
                logger.info(
                    "Remote transcription complete via %s job=%s in %.1fs",
                    base, job_id, time.monotonic() - start,
                )
                return result
            except Exception as e:
                last_err = e
                logger.warning(
                    "Worker %s failed for job %s: %s", base, job_id, e,
                )
        raise RemoteTranscriptionError(
            f"all workers failed for job {job_id}: {last_err}"
        )
    finally:
        _progress.pop(job_id, None)
