"""Tests for GPU worker dispatch (app/remote.py) and its integration into the
job flow: dispatch to a worker when configured and reachable, fall back to
local CPU transcription when the worker is unreachable or fails, and keep the
normal job lifecycle (status.json, /j/{job_id}) either way.
"""
import asyncio
import json
import time
from pathlib import Path
from unittest.mock import patch

import httpx
from fastapi.testclient import TestClient

import app.remote as remote
from app.main import app, _jobs

client = TestClient(app)


def _make_async_mock(result_dict):
    async def _mock(*args, **kwargs):
        return result_dict
    return _mock


def _make_raise_async_mock(exc):
    async def _mock(*args, **kwargs):
        raise exc
    return _mock


def _wait_for_status(job_id, wanted, tries=60):
    for _ in range(tries):
        time.sleep(0.1)
        r = client.get(f"/api/transcribe/status/{job_id}")
        if r.status_code == 200 and r.json().get("status") in wanted:
            return r.json()
    return None


REMOTE_RESULT = {
    "id": "w" * 32,
    "text": "hello from the laptop GPU",
    "language": "en",
    "device": "cuda",
    "compute_type": "float32",
    "model": "small",
    "segments": [{"text": "hello from the laptop GPU", "t0": 0, "t1": 1200}],
    "duration": 1.2,
    "process_time": 0.3,
}


class _FakeWorker:
    """In-process fake of a worker app (health + transcribe + status)."""

    def __init__(self, device="cuda", fail_transcribe=False, down_after_polls=0,
                 complete_after_polls=3):
        self.device = device
        self.fail_transcribe = fail_transcribe
        self.down_after_polls = down_after_polls
        self.complete_after_polls = complete_after_polls
        self.jobs: dict[str, dict] = {}
        self.poll_count = 0

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/health":
            return httpx.Response(200, json={"status": "ok", "device": self.device})
        if path == "/api/transcribe":
            if self.fail_transcribe:
                return httpx.Response(500, json={"detail": "gpu oom"})
            wid = "f" * 32
            self.jobs[wid] = {"status": "processing", "progress": 0.1,
                              "created_at": time.time()}
            return httpx.Response(200, json={"job_id": wid, "status": "processing",
                                             "progress": 0.1})
        if path.startswith("/api/transcribe/status/"):
            wid = path.rsplit("/", 1)[-1]
            job = self.jobs.get(wid)
            if job is None:
                return httpx.Response(404, json={"detail": "Job not found"})
            self.poll_count += 1
            if self.down_after_polls and self.poll_count > self.down_after_polls:
                return httpx.Response(503, json={"detail": "service down"})
            if job["status"] == "processing":
                if self.poll_count >= self.complete_after_polls:
                    job["status"] = "completed"
                    job["result"] = dict(REMOTE_RESULT)
                    job["progress"] = 1.0
                return httpx.Response(200, json=job)
            if job["status"] == "completed":
                return httpx.Response(200, json=job)
        return httpx.Response(404, json={"detail": "nope"})


def _transport_for(*workers: _FakeWorker) -> httpx.MockTransport:
    by_host = {f"http://worker{i}:{8000 + i}": w for i, w in enumerate(workers)}

    def handler(request: httpx.Request) -> httpx.Response:
        base = f"{request.url.scheme}://{request.url.host}:{request.url.port}"
        w = by_host.get(base)
        if w is None:
            return httpx.Response(404)
        return w.handle(request)
    return httpx.MockTransport(handler)


def _audio_file(tmp_path: Path) -> str:
    p = tmp_path / "input.wav"
    p.write_bytes(b"\x00" * 1024)
    return str(p)


# ---------------------------------------------------------------------------
# Dispatch decision
# ---------------------------------------------------------------------------


def test_should_dispatch_false_without_workers():
    with patch.object(remote, "WHISPER_WORKER_URLS", []):
        assert remote.should_dispatch("base") is False


def test_should_dispatch_only_gpu_models():
    with patch.object(remote, "WHISPER_WORKER_URLS", ["http://w:8564"]), \
         patch.object(remote, "WHISPER_GPU_MODELS", {"tiny", "base", "small"}):
        assert remote.should_dispatch("base") is True
        assert remote.should_dispatch("small") is True
        assert remote.should_dispatch("large-v3-turbo") is False


def test_worker_order_round_robin():
    with patch.object(remote, "WHISPER_WORKER_URLS", ["a", "b", "c"]):
        remote._round_robin_index = 0
        assert remote._worker_order() == ["a", "b", "c"]
        assert remote._worker_order() == ["b", "c", "a"]
        assert remote._worker_order() == ["c", "a", "b"]
        assert remote._worker_order() == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# transcribe_remote client behaviour (in-process fake worker)
# ---------------------------------------------------------------------------


def test_transcribe_remote_happy_path(tmp_path):
    w = _FakeWorker()
    transport = _transport_for(w)
    with patch.object(remote, "WHISPER_WORKER_URLS", ["http://worker0:8000"]):
        result = asyncio.run(remote.transcribe_remote(
            _audio_file(tmp_path), "en", "small", "verbatim", "j" * 32,
            _transport=transport,
        ))
    assert result["text"] == REMOTE_RESULT["text"]
    assert result["device"] == "cuda"
    assert result["id"] == "j" * 32  # rewritten to OUR job id


def test_transcribe_remote_skips_unreachable_worker(tmp_path):
    w = _FakeWorker()
    transport = _transport_for(w)  # only http://worker0:8000 exists
    with patch.object(remote, "WHISPER_WORKER_URLS", ["http://worker1:8001", "http://worker0:8000"]):
        result = asyncio.run(remote.transcribe_remote(
            _audio_file(tmp_path), "en", "small", "verbatim", "j" * 32,
            _transport=transport,
        ))
    assert result["id"] == "j" * 32


def test_transcribe_remote_raises_when_worker_fails(tmp_path):
    w = _FakeWorker(fail_transcribe=True)
    transport = _transport_for(w)
    with patch.object(remote, "WHISPER_WORKER_URLS", ["http://worker0:8000"]):
        try:
            asyncio.run(remote.transcribe_remote(
                _audio_file(tmp_path), "en", "small", "verbatim", "j" * 32,
                _transport=transport,
            ))
            assert False, "expected RemoteTranscriptionError"
        except remote.RemoteTranscriptionError as e:
            assert "failed" in str(e) or "rejected" in str(e)


def test_transcribe_remote_raises_when_worker_goes_down(tmp_path):
    w = _FakeWorker(down_after_polls=3, complete_after_polls=99)
    transport = _transport_for(w)
    with patch.object(remote, "WHISPER_WORKER_URLS", ["http://worker0:8000"]), \
         patch.object(remote, "MAX_POLL_FAILURES", 2):
        try:
            asyncio.run(remote.transcribe_remote(
                _audio_file(tmp_path), "en", "small", "verbatim", "j" * 32,
                _transport=transport,
            ))
            assert False, "expected RemoteTranscriptionError"
        except remote.RemoteTranscriptionError:
            pass


# ---------------------------------------------------------------------------
# Integration: job lifecycle with dispatch + fallback
# ---------------------------------------------------------------------------


def test_job_completes_through_remote_when_dispatched():
    with patch("app.main.should_dispatch", return_value=True), \
         patch("app.main.transcribe_remote", side_effect=_make_async_mock(dict(REMOTE_RESULT))), \
         patch("app.main.transcribe_audio", side_effect=_make_async_mock(
             {"id": "x", "text": "should not be used"})):
        r = client.post("/api/transcribe",
                        files={"file": ("a.wav", b"\x00" * 16, "audio/wav")},
                        data={"language": "en", "model": "base", "mode": "verbatim"})
        assert r.status_code == 200
        job_id = r.json()["job_id"]
        st = _wait_for_status(job_id, ("completed", "failed"))
        assert st and st["status"] == "completed"
        assert st["result"]["text"] == REMOTE_RESULT["text"]


def test_job_falls_back_to_local_when_worker_fails():
    """Remote dispatch raising must not fail the job — local CPU takes over."""
    with patch("app.main.should_dispatch", return_value=True), \
         patch("app.main.transcribe_remote",
               side_effect=_make_raise_async_mock(remote.RemoteTranscriptionError("all down"))), \
         patch("app.main.transcribe_audio", side_effect=_make_async_mock({
             "id": "x", "text": "local fallback result", "language": "en",
             "device": "cpu", "segments": [], "duration": 0.0, "process_time": 0.1,
         })):
        r = client.post("/api/transcribe",
                        files={"file": ("a.wav", b"\x00" * 16, "audio/wav")},
                        data={"language": "en", "model": "base", "mode": "verbatim"})
        assert r.status_code == 200
        job_id = r.json()["job_id"]
        st = _wait_for_status(job_id, ("completed", "failed"))
        assert st and st["status"] == "completed"
        assert st["result"]["text"] == "local fallback result"


def test_job_local_path_when_model_not_dispatched():
    """Models outside the GPU whitelist never hit a worker."""
    with patch("app.main.should_dispatch", return_value=False), \
         patch("app.main.transcribe_audio", side_effect=_make_async_mock({
             "id": "x", "text": "local result", "language": "en",
             "device": "cpu", "segments": [], "duration": 0.0, "process_time": 0.1,
         })):
        r = client.post("/api/transcribe",
                        files={"file": ("a.wav", b"\x00" * 16, "audio/wav")},
                        data={"language": "en", "model": "large-v3-turbo", "mode": "verbatim"})
        assert r.status_code == 200
        job_id = r.json()["job_id"]
        st = _wait_for_status(job_id, ("completed", "failed"))
        assert st and st["status"] == "completed"
        assert st["result"]["text"] == "local result"
