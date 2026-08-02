"""Retention tests: transcripts survive, recordings are deleted earlier.

Covers the split-retention contract:
- startup cleanup (config.cleanup_all_jobs) is retention-aware, NOT a wipe
- recordings (input audio) are deleted sooner than transcripts (status.json)
- completed/failed jobs keep status.json and drop the input audio
- persisted jobs are reloaded on (simulated) restart
- /api/transcribe/status exposes honest expiry fields
- /api/upload/config exposes retention settings
"""
import asyncio as _asyncio
import json
import os
import time
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import app.config as config
import app.main as app_main
from app.main import app, _jobs

client = TestClient(app)


def _make_async_mock(result_dict):
    async def _mock(*args, **kwargs):
        return result_dict
    return _mock


def _make_job_dir(root: Path, job_id: str, created_at: float,
                  with_status: bool = True, with_input: bool = True) -> Path:
    d = root / job_id
    d.mkdir(parents=True, exist_ok=True)
    if with_input:
        (d / "input.wav").write_bytes(b"\x00" * 16)
    if with_status:
        (d / "status.json").write_text(json.dumps({
            "status": "completed",
            "created_at": created_at,
            "result": {"text": "hello", "id": job_id},
        }))
    return d


def _wait_for_status(job_id: str, wanted: tuple[str, ...], tries: int = 50):
    for _ in range(tries):
        _asyncio.sleep(0.1)
        r = client.get(f"/api/transcribe/status/{job_id}")
        if r.status_code == 200 and r.json().get("status") in wanted:
            return r.json()
    return None


# ---------------------------------------------------------------------------
# Configuration-level retention contract
# ---------------------------------------------------------------------------


def test_transcripts_kept_at_least_one_hour():
    """Requirement: transcripts survive >= 1 hour (default is 2h)."""
    assert config.JOB_RETENTION_SECONDS >= 3600
    assert config.AUDIO_RETENTION_SECONDS < config.JOB_RETENTION_SECONDS


def test_cleanup_all_jobs_preserves_fresh_jobs(tmp_path):
    """Startup cleanup must NOT wipe fresh jobs (status + audio intact)."""
    now = time.time()
    d = _make_job_dir(tmp_path, "a" * 32, created_at=now - 60)
    with patch.object(config, "WORK_DIR", tmp_path):
        config.cleanup_all_jobs()
    assert d.exists()
    assert (d / "status.json").exists()
    assert (d / "input.wav").exists()


def test_cleanup_all_jobs_removes_expired_jobs(tmp_path):
    """Job dirs older than JOB_RETENTION_SECONDS are removed at startup."""
    now = time.time()
    old = now - config.JOB_RETENTION_SECONDS - 1000
    d = _make_job_dir(tmp_path, "b" * 32, created_at=old)
    with patch.object(config, "WORK_DIR", tmp_path):
        config.cleanup_all_jobs()
    assert not d.exists()


def test_cleanup_all_jobs_deletes_stale_audio_keeps_status(tmp_path):
    """Audio older than AUDIO_RETENTION_SECONDS is dropped; transcript kept."""
    now = time.time()
    d = _make_job_dir(tmp_path, "c" * 32, created_at=now - 60)
    input_f = d / "input.wav"
    old = now - config.AUDIO_RETENTION_SECONDS - 100
    os.utime(input_f, (old, old))
    with patch.object(config, "WORK_DIR", tmp_path):
        config.cleanup_all_jobs()
    assert d.exists()
    assert (d / "status.json").exists()
    assert not input_f.exists()


def test_cleanup_all_jobs_preserves_chunks_dir(tmp_path):
    """The chunked-upload sessions directory is never touched."""
    chunks = tmp_path / "chunks"
    (chunks / "sess1").mkdir(parents=True, exist_ok=True)
    (chunks / "sess1" / "meta.json").write_text("{}")
    with patch.object(config, "WORK_DIR", tmp_path):
        config.cleanup_all_jobs()
    assert chunks.exists()
    assert (chunks / "sess1").exists()


def test_cleanup_job_audio_keeps_status(tmp_path):
    """Post-completion cleanup removes the recording but keeps status.json."""
    d = _make_job_dir(tmp_path, "d" * 32, created_at=time.time())
    (d / "input.mp3").write_bytes(b"x")
    with patch.object(config, "WORK_DIR", tmp_path):
        config.cleanup_job_audio("d" * 32)
    assert d.exists()
    assert (d / "status.json").exists()
    assert not (d / "input.mp3").exists()


def test_job_created_at_prefers_status_json(tmp_path):
    d = _make_job_dir(tmp_path, "e" * 32, created_at=12345.0)
    assert config.job_created_at(d) == 12345.0
    d2 = tmp_path / ("f" * 32)
    d2.mkdir()
    assert abs(config.job_created_at(d2) - time.time()) < 5


# ---------------------------------------------------------------------------
# Periodic sweep (shared helper _cleanup_expired_jobs)
# ---------------------------------------------------------------------------


def test_cleanup_expired_jobs_sweep(tmp_path):
    """Sweep removes expired dirs, drops stale audio, keeps fresh transcripts,
    and never touches running jobs or chunks."""
    now = time.time()
    fresh = _make_job_dir(tmp_path, "aa" * 16, created_at=now - 60)
    stale_audio = _make_job_dir(tmp_path, "bb" * 16, created_at=now - 60)
    input_f = stale_audio / "input.wav"
    old = now - config.AUDIO_RETENTION_SECONDS - 100
    os.utime(input_f, (old, old))
    expired = _make_job_dir(tmp_path, "cc" * 16, created_at=now - config.JOB_RETENTION_SECONDS - 100)
    running = _make_job_dir(tmp_path, "dd" * 16, created_at=now - config.JOB_RETENTION_SECONDS - 100)
    chunks = tmp_path / "chunks"
    (chunks / "sess1").mkdir(parents=True, exist_ok=True)

    with patch.object(app_main, "WORK_DIR", tmp_path):
        removed = app_main._cleanup_expired_jobs(now, running_ids={"dd" * 16})

    assert removed == ["cc" * 16]
    assert not (tmp_path / ("cc" * 16)).exists()
    assert (tmp_path / ("dd" * 16)).exists()          # running jobs never cleaned
    assert fresh.exists() and (fresh / "status.json").exists()
    assert (fresh / "input.wav").exists()           # fresh audio kept
    assert stale_audio.exists() and (stale_audio / "status.json").exists()
    assert not input_f.exists()                     # stale audio dropped
    assert chunks.exists()                          # chunks untouched


# ---------------------------------------------------------------------------
# End-to-end: completed job keeps status on disk, survives (simulated) restart
# ---------------------------------------------------------------------------


def test_completed_job_persists_status_and_deletes_audio():
    fast_result = {
        "text": "retention result", "language": "en",
        "duration": 0.5, "process_time": 0.05,
        "segments": [], "id": "x",
        "device": "cpu", "compute_type": "int8",
    }
    with patch("app.main.transcribe_audio", side_effect=_make_async_mock(fast_result)):
        resp = client.post(
            "/api/transcribe",
            files={"file": ("test.wav", b"X" * 1024, "audio/wav")},
            data={"language": "en", "model": "base"},
        )
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        status = _wait_for_status(job_id, ("completed", "failed"))
        assert status is not None and status["status"] == "completed"

        # Honest expiry fields present on the status response.
        assert status["retention_seconds"] == config.JOB_RETENTION_SECONDS
        assert status["expires_at"] > time.time()
        assert 0 < status["expires_in_seconds"] <= status["retention_seconds"]

        # Job dir keeps status.json but the recording is already gone.
        job_dir = app_main.WORK_DIR / job_id
        assert job_dir.exists()
        assert (job_dir / "status.json").exists()
        assert [p for p in job_dir.iterdir() if p.name.startswith("input")] == []

        # Simulated restart: run the startup sequence — retention-aware cleanup
        # must NOT wipe the job, then reload persisted jobs from disk.
        config.cleanup_all_jobs()
        _jobs.pop(job_id, None)
        loaded = app_main._load_persisted_jobs()
        assert job_id in loaded
        _jobs[job_id] = loaded[job_id]
        r = client.get(f"/api/transcribe/status/{job_id}")
        assert r.status_code == 200
        assert r.json()["status"] == "completed"
        assert r.json()["result"]["text"] == "retention result"
        _jobs.pop(job_id, None)


def test_failed_job_status_includes_expiry_and_keeps_status():
    async def _fail(*args, **kwargs):
        raise RuntimeError("boom")

    with patch("app.main.transcribe_audio", side_effect=_fail):
        resp = client.post(
            "/api/transcribe",
            files={"file": ("test.wav", b"X" * 1024, "audio/wav")},
            data={"language": "en", "model": "base"},
        )
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]

        status = _wait_for_status(job_id, ("completed", "failed"))
        assert status is not None and status["status"] == "failed"
        assert status["error"] == "boom"
        assert "expires_at" in status
        assert "retention_seconds" in status
        assert status["retention_seconds"] == config.JOB_RETENTION_SECONDS

        job_dir = app_main.WORK_DIR / job_id
        assert job_dir.exists()
        assert (job_dir / "status.json").exists()
        assert [p for p in job_dir.iterdir() if p.name.startswith("input")] == []
        _jobs.pop(job_id, None)


def test_upload_config_includes_retention():
    """Frontend needs the retention windows to state honest expiry."""
    resp = client.get("/api/upload/config")
    assert resp.status_code == 200
    data = resp.json()
    assert data["job_retention_seconds"] == config.JOB_RETENTION_SECONDS
    assert data["audio_retention_seconds"] == config.AUDIO_RETENTION_SECONDS


def test_processing_status_also_reports_expiry():
    """Expiry fields are additive on every status response, including processing."""
    from app.main import _jobs

    slow_result = {
        "text": "slow", "language": "en",
        "duration": 1.0, "process_time": 10.0,
        "segments": [], "id": "x",
        "device": "cpu", "compute_type": "int8",
    }
    async def _slow(*args, **kwargs):
        await _asyncio.sleep(10)
        return slow_result

    with patch("app.main.transcribe_audio", side_effect=_slow):
        resp = client.post(
            "/api/transcribe",
            files={"file": ("test.wav", b"X" * 1024, "audio/wav")},
            data={"language": "en", "model": "base"},
        )
        assert resp.status_code == 200
        job_id = resp.json()["job_id"]
        _asyncio.sleep(0.3)
        r = client.get(f"/api/transcribe/status/{job_id}")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "processing"
        assert data["expires_at"] > time.time()
        assert data["retention_seconds"] == config.JOB_RETENTION_SECONDS
        _jobs.pop(job_id, None)
