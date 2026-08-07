"""GPU worker dispatch for transcription jobs.

Nodes (GPU workers) come from app/nodes.py — config/nodes.yaml or the
WHISPER_WORKER_URLS env — as a list of {name, url}. When nodes are
configured, the app forwards transcription jobs for GPU-friendly models to
them; whether a job falls back to local CPU transcription is governed by
WHISPER_ALLOW_LOCAL_CPU (see app/config.py — default false on this stack).
With no nodes configured the app behaves exactly as before (always local).

Each job can also name a specific node ("Automatic", a node name, a 1-based
index, or the built-in "CPU" node). An explicitly chosen node is used
exclusively — if it fails, the job fails with a message naming the node
(the caller decides that policy; see app.main).

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

from app.nodes import NODES, CPU_NODE
from app.transcriber import _progress

logger = logging.getLogger(__name__)

# Base URLs of the configured nodes (empty = dispatch disabled, local only).
WHISPER_WORKER_URLS = [n["url"] for n in NODES]

# Display names for the nodes, same order as WHISPER_WORKER_URLS.
WHISPER_WORKER_NAMES = [n["name"] for n in NODES]

# Models we may dispatch to the workers' small-VRAM GPUs. The worker GPUs
# (Maxwell/Pascal) run float32 (no fp16/int8 kernels on CC 5.0/6.1):
# tiny/base/small fit ~2 GB VRAM; larger models would OOM the worker and
# waste the upload round trip. CrisperWhisper needs the torch backend the
# worker image deliberately excludes, so it is never auto-dispatched.
# An explicitly chosen node overrides this list (see should_dispatch).
WHISPER_GPU_MODELS = {
    m.strip()
    for m in os.getenv("WHISPER_GPU_MODELS", "tiny,base,small").split(",")
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

# base url -> device reported by the worker's /health ("cuda"/"cpu").
_worker_devices: dict[str, str] = {}


# ---------------------------------------------------------------------------
# Worker selectors
# ---------------------------------------------------------------------------


def worker_index(worker: str | None) -> int | None:
    """Resolve a worker selector to an index into WHISPER_WORKER_URLS.

    None = automatic. -1 = unknown name/index (caller rejects with 400).
    Accepts "auto"/"" (automatic), a display name (case-insensitive), or a
    1-based index into the worker list.
    """
    w = (worker or "").strip()
    if not w or w.lower() == "auto":
        return None
    if w.isdigit():
        i = int(w)
        if 1 <= i <= len(WHISPER_WORKER_URLS):
            return i - 1
        return -1
    low = w.lower()
    for i, name in enumerate(WHISPER_WORKER_NAMES):
        if name.lower() == low:
            return i
    return -1


def validate_worker(worker: str | None) -> str:
    """Validate a worker selector from a request.

    Returns "auto", CPU_NODE ("CPU"), or the worker's display NAME.
    Raises HTTPException(400) for an unknown name/index.
    """
    from fastapi import HTTPException
    idx = worker_index(worker)
    if idx is None:
        return "auto"
    if idx < 0:
        if (worker or "").strip().lower() == CPU_NODE.lower():
            return CPU_NODE
        raise HTTPException(
            400,
            f"Unknown node: '{worker}'. Choose Automatic, {CPU_NODE}, or one of: "
            + ", ".join(WHISPER_WORKER_NAMES),
        )
    return WHISPER_WORKER_NAMES[idx]


def should_dispatch(model_name: str, worker: str = "auto") -> bool:
    """True when a job should be attempted on a GPU worker.

    An explicitly chosen node always routes there (the user decides);
    automatic dispatch only sends models that fit the workers' VRAM. The
    CPU_NODE selection never dispatches.
    """
    if not WHISPER_WORKER_URLS:
        return False
    if worker == CPU_NODE:
        return False
    if worker and worker != "auto":
        return True
    return model_name in WHISPER_GPU_MODELS


def _name_for(base: str) -> str:
    try:
        i = WHISPER_WORKER_URLS.index(base)
    except ValueError:
        return base
    if i < len(WHISPER_WORKER_NAMES):
        return WHISPER_WORKER_NAMES[i]
    return f"Worker {i + 1}"


def worker_label(base: str) -> str:
    """Display label for a worker, e.g. "GPU Node B (GPU)" — device appended
    when the worker reports a CUDA device (cached from health checks)."""
    name = _name_for(base)
    if _worker_devices.get(base) == "cuda" and "gpu" not in name.lower():
        return f"{name} (GPU)"
    return name


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


# ---------------------------------------------------------------------------
# Worker client
# ---------------------------------------------------------------------------


async def _check_worker(base: str, transport=None) -> bool:
    try:
        async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT, transport=transport) as client:
            r = await client.get(f"{base}/health")
        if r.status_code == 200:
            try:
                _worker_devices[base] = (r.json().get("device") or "cpu")
            except Exception:
                pass
            return True
    except Exception:
        pass
    return False


async def worker_statuses(transport=None) -> list[dict]:
    """Live online/offline + device for every configured worker."""
    async def _status(base: str) -> dict:
        online = await _check_worker(base, transport=transport)
        return {
            "name": _name_for(base),
            "url": base,
            "online": online,
            "device": _worker_devices.get(base),
        }
    return await asyncio.gather(*(_status(b) for b in WHISPER_WORKER_URLS))


async def _upload_and_poll(
    base: str, audio_path: str, language: str, model_name: str, mode: str, job_id: str,
    transport=None, on_update=None,
) -> dict:
    """Upload audio to one worker and poll its job to completion.

    Returns the worker's transcript result dict. Raises
    RemoteTranscriptionError on any worker-side failure so the caller can
    fall back to local CPU transcription (or fail the job for an explicit
    node choice).
    """
    label = worker_label(base)

    if not await _check_worker(base, transport=transport):
        raise RemoteTranscriptionError(f"{label} is unreachable")
    if on_update is not None:
        await on_update("dispatched", label)

    async with httpx.AsyncClient(timeout=UPLOAD_TIMEOUT, transport=transport) as client:
        with open(audio_path, "rb") as f:
            files = {"file": (os.path.basename(audio_path), f, "application/octet-stream")}
            data = {"language": language or "", "model": model_name, "mode": mode}
            r = await client.post(f"{base}/api/transcribe", files=files, data=data)
        if r.status_code != 200:
            raise RemoteTranscriptionError(
                f"{label} rejected the upload (HTTP {r.status_code}): {r.text[:200]}"
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
                        f"{label} stopped responding"
                    ) from e
                continue

            st = r.json()
            status = st.get("status")
            if status == "processing":
                progress = st.get("progress")
                if isinstance(progress, (int, float)):
                    # Mirror the worker's real progress onto our job.
                    _progress[job_id] = min(0.99, float(progress))
                    if on_update is not None:
                        await on_update("transcribing", label)
                # Otherwise the worker reports "working, progress unknown" —
                # leave ours unknown too.
                continue
            if status == "completed":
                result = st.get("result")
                if not isinstance(result, dict) or not result:
                    raise RemoteTranscriptionError(
                        f"{label} returned an empty result for job {worker_job_id}"
                    )
                return result
            if status == "failed":
                raise RemoteTranscriptionError(
                    f"{label} failed the job: {st.get('error', 'unknown error')}"
                )
            if status == "interrupted":
                raise RemoteTranscriptionError(f"{label} restarted mid-job (interrupted)")
            raise RemoteTranscriptionError(f"{label} returned unexpected status {status!r}")


async def transcribe_remote(
    audio_path: str,
    language: str,
    model_name: str,
    mode: str,
    job_id: str,
    worker: str = "auto",
    on_update=None,      # async (stage: str, node_label: str) -> None
    _transport=None,     # test hook: httpx.MockTransport
) -> dict:
    """Transcribe via the GPU workers.

    worker="auto" (default): round-robin over the configured workers.
    worker=<name/index>: try ONLY that worker. In both cases a failure
    raises RemoteTranscriptionError; for automatic mode the caller falls
    back to local CPU, for an explicit node the caller fails the job.
    """
    # Unknown until a worker reports real progress -> UI shows "working".
    _progress[job_id] = None
    start = time.monotonic()
    last_err: Exception | None = None
    try:
        idx = worker_index(worker)
        if idx is not None and idx >= 0:
            base = WHISPER_WORKER_URLS[idx]
            result = await _upload_and_poll(
                base, audio_path, language, model_name, mode, job_id,
                transport=_transport, on_update=on_update,
            )
            return _finalize_result(result, job_id, base, start)

        for base in _worker_order():
            try:
                result = await _upload_and_poll(
                    base, audio_path, language, model_name, mode, job_id,
                    transport=_transport, on_update=on_update,
                )
                return _finalize_result(result, job_id, base, start)
            except Exception as e:
                last_err = e
                logger.warning("Worker %s failed for job %s: %s", base, job_id, e)
        raise RemoteTranscriptionError(
            f"all workers failed for job {job_id}: {last_err}"
        )
    finally:
        _progress.pop(job_id, None)


def _finalize_result(result: dict, job_id: str, base: str, start: float) -> dict:
    result = dict(result)
    # The worker's result carries its own job id — ours is the one the user polls.
    result["id"] = job_id
    logger.info(
        "Remote transcription complete via %s job=%s in %.1fs",
        worker_label(base), job_id, time.monotonic() - start,
    )
    return result
