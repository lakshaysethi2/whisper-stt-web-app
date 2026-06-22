import json
import math
import shutil
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from app.main import app, CHUNK_DIR, CHUNK_SIZE, MAX_CHUNKS

client = TestClient(app)


def _cleanup_upload(upload_id: str):
    d = CHUNK_DIR / upload_id
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)


import asyncio as _asyncio


def _run_sync(coro):
    """Helper: run a coroutine in a fresh event loop (for setup, NOT for mocks)."""
    loop = _asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


def _make_async_mock(result_dict):
    """Return an async side_effect that resolves immediately to result_dict."""
    async def _mock(*args, **kwargs):
        return result_dict
    return _mock


def _make_slow_async_mock(result_dict, delay=10):
    """Return an async side_effect that sleeps then resolves to result_dict."""
    async def _mock(*args, **kwargs):
        await _asyncio.sleep(delay)
        return result_dict
    return _mock


def test_chunked_upload_flow():
    """Start -> chunks -> finish (mocked transcription). Verifies async behaviour."""
    upload_id = None
    try:
        payload = b"AA" * 512
        size = len(payload)
        resp = client.post(
            "/api/upload/start",
            data={"filename": "test.wav", "size": size, "total_chunks": math.ceil(size / CHUNK_SIZE)},
        )
        assert resp.status_code == 200
        upload_id = resp.json()["upload_id"]
        total_chunks = math.ceil(size / CHUNK_SIZE)

        resp = client.post(
            f"/api/upload/chunk/{upload_id}",
            data={"chunk_index": 0},
            files={"file": ("test.wav.part0", payload, "audio/wav")},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["received"] == 1
        assert body["total"] == total_chunks

        mock_result = {
            "text": "hello world", "language": "en",
            "duration": 2.0, "process_time": 0.1,
            "segments": [], "id": "x",
            "device": "cpu", "compute_type": "int8",
        }

        with patch("app.main.transcribe_audio",
                   side_effect=_make_async_mock(mock_result)):
            resp = client.post(f"/api/upload/finish/{upload_id}", params={"language": "en"}, timeout=2.0)
            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "processing"
            job_id = data["job_id"]

            # Poll for completion
            final = None
            for _ in range(50):
                _asyncio.sleep(0.1)
                r = client.get(f"/api/transcribe/status/{job_id}")
                if r.status_code == 200 and r.json().get("status") == "completed":
                    final = r.json()
                    break
            assert final is not None
            assert final["result"]["text"] == "hello world"

            # Cleanup job entry
            from app.main import _jobs
            _jobs.pop(job_id, None)
    finally:
        if upload_id:
            _cleanup_upload(upload_id)


def test_missing_chunks_returns_detail_and_preserves_session():
    """finish with missing chunks returns detail and does NOT delete the session."""
    with patch("app.main.CHUNK_SIZE", 10), patch("app.main.MAX_CHUNKS", 100):
        upload_id = None
        try:
            resp = client.post(
                "/api/upload/start",
                data={"filename": "test.wav", "size": 30, "total_chunks": math.ceil(30 / 10)},
            )
            upload_id = resp.json()["upload_id"]
            total_chunks = math.ceil(30 / 10)

            # Upload chunks 0 and 2, skip chunk 1
            for i in [0, 2]:
                client.post(
                    f"/api/upload/chunk/{upload_id}",
                    data={"chunk_index": i},
                    files={"file": (f"test.wav.part{i}", b"X" * 10, "audio/wav")},
                )

            resp = client.post(f"/api/upload/finish/{upload_id}", data={})
            assert resp.status_code == 400
            detail = resp.json()["detail"]
            assert "upload_id" in detail
            assert detail["total_chunks"] == total_chunks
            assert 1 in detail["missing"]

            # Session must still exist on disk so client can upload missing chunks
            assert (CHUNK_DIR / upload_id).exists()
        finally:
            if upload_id:
                _cleanup_upload(upload_id)


def test_invalid_language_preserves_session():
    """Invalid language code rejects finish without deleting the session."""
    with patch("app.main.CHUNK_SIZE", 10), patch("app.main.MAX_CHUNKS", 100):
        upload_id = None
        try:
            resp = client.post(
                "/api/upload/start",
                data={"filename": "test.wav", "size": 20, "total_chunks": math.ceil(20 / 10)},
            )
            upload_id = resp.json()["upload_id"]
            total_chunks = math.ceil(20 / 10)

            for i in range(total_chunks):
                client.post(
                    f"/api/upload/chunk/{upload_id}",
                    data={"chunk_index": i},
                    files={"file": (f"test.wav.part{i}", b"X" * 10, "audio/wav")},
                )

            resp = client.post(f"/api/upload/finish/{upload_id}", params={"language": "!!!"})
            assert resp.status_code == 400
            assert "Invalid language" in resp.json()["detail"]

            # Session must still exist
            assert (CHUNK_DIR / upload_id).exists()
        finally:
            if upload_id:
                _cleanup_upload(upload_id)


def test_size_mismatch_rejected():
    """Assembled size must match declared size."""
    with patch("app.main.CHUNK_SIZE", 10), patch("app.main.MAX_CHUNKS", 100):
        upload_id = None
        try:
            # Declare size=20 but actually upload 30 bytes total
            resp = client.post(
                "/api/upload/start",
                data={"filename": "test.wav", "size": 20, "total_chunks": math.ceil(20 / 10)},
            )
            upload_id = resp.json()["upload_id"]
            total_chunks = math.ceil(20 / 10)

            # Upload first chunk at correct size, second at correct size for its index
            # but the last chunk gets extra data to cause a mismatch
            for i in range(total_chunks):
                expected = min(10, 20 - i * 10)
                client.post(
                    f"/api/upload/chunk/{upload_id}",
                    data={"chunk_index": i},
                    files={"file": (f"test.wav.part{i}", b"X" * expected, "audio/wav")},
                )

            # Tamper: overwrite chunk 0 with larger data directly
            chunk0 = CHUNK_DIR / upload_id / "0.part"
            chunk0.write_bytes(b"X" * 20)

            with patch("app.main.transcribe_audio"):
                resp = client.post(f"/api/upload/finish/{upload_id}", data={"language": "en"})
                assert resp.status_code == 400
                assert "size mismatch" in resp.json()["detail"].lower()
        finally:
            if upload_id:
                _cleanup_upload(upload_id)


def test_invalid_total_chunks_rejected():
    """total_chunks must match expected count for declared size."""
    with patch("app.main.CHUNK_SIZE", 10), patch("app.main.MAX_CHUNKS", 100):
        resp = client.post(
            "/api/upload/start",
            data={"filename": "test.wav", "size": 100, "total_chunks": 999},
        )
        assert resp.status_code == 400
        assert "Invalid total_chunks" in resp.json()["detail"]


def test_chunk_wrong_size_rejected():
    """Each chunk must match its expected byte length."""
    with patch("app.main.CHUNK_SIZE", 10), patch("app.main.MAX_CHUNKS", 100):
        upload_id = None
        try:
            # size=25, total_chunks=ceil(25/10)=3. Chunk 0 expects 10 bytes.
            resp = client.post(
                "/api/upload/start",
                data={"filename": "test.wav", "size": 25, "total_chunks": math.ceil(25 / 10)},
            )
            assert resp.status_code == 200
            upload_id = resp.json()["upload_id"]

            # Upload 15 bytes for chunk 0 (expected 10) — should be rejected
            resp = client.post(
                f"/api/upload/chunk/{upload_id}",
                data={"chunk_index": 0},
                files={"file": ("test.wav.part0", b"X" * 15, "audio/wav")},
            )
            assert resp.status_code == 400
            assert "Invalid chunk size" in resp.json()["detail"]

            # Upload correct 10 bytes for chunk 0 — should succeed
            resp = client.post(
                f"/api/upload/chunk/{upload_id}",
                data={"chunk_index": 0},
                files={"file": ("test.wav.part0", b"X" * 10, "audio/wav")},
            )
            assert resp.status_code == 200
        finally:
            if upload_id:
                _cleanup_upload(upload_id)


def test_chunk_too_large():
    """Reject chunks exceeding CHUNK_MAX_SIZE."""
    with patch("app.main.CHUNK_SIZE", 10), patch("app.main.MAX_CHUNKS", 100), \
         patch("app.main.CHUNK_MAX_SIZE", 1024):
        upload_id = None
        try:
            resp = client.post(
                "/api/upload/start",
                data={"filename": "big.wav", "size": 200, "total_chunks": math.ceil(200 / 10)},
            )
            assert resp.status_code == 200
            upload_id = resp.json()["upload_id"]

            oversized = b"\x00" * 1025
            resp = client.post(
                f"/api/upload/chunk/{upload_id}",
                data={"chunk_index": 0},
                files={"file": ("big.wav.part0", oversized, "audio/wav")},
            )
            assert resp.status_code == 413
        finally:
            if upload_id:
                _cleanup_upload(upload_id)


def test_concurrent_finish_guard():
    """Second finish call while first is in-progress returns 409."""
    from app.main import _finish_locks

    with patch("app.main.CHUNK_SIZE", 10), patch("app.main.MAX_CHUNKS", 100):
        upload_id = None
        try:
            resp = client.post(
                "/api/upload/start",
                data={"filename": "test.wav", "size": 10, "total_chunks": 1},
            )
            upload_id = resp.json()["upload_id"]

            client.post(
                f"/api/upload/chunk/{upload_id}",
                data={"chunk_index": 0},
                files={"file": ("test.wav.part0", b"X" * 10, "audio/wav")},
            )

            _finish_locks.add(upload_id)
            try:
                resp = client.post(f"/api/upload/finish/{upload_id}", data={})
                assert resp.status_code == 409
            finally:
                _finish_locks.discard(upload_id)
        finally:
            if upload_id:
                _cleanup_upload(upload_id)


def test_upload_config():
    """GET /api/upload/config returns expected keys."""
    resp = client.get("/api/upload/config")
    assert resp.status_code == 200
    data = resp.json()
    assert "chunk_size" in data
    assert data["chunk_size"] == CHUNK_SIZE
    assert "max_chunk_size" in data
    assert "max_file_size" in data
    assert "max_chunks" in data
    assert "direct_upload_threshold" in data
    assert "allowed_extensions" in data
    assert isinstance(data["allowed_extensions"], list)


# ---------------------------------------------------------------------------
# Async transcription (fire-and-poll) tests
# ---------------------------------------------------------------------------


def test_finish_returns_immediately_with_job_id():
    """/finish should not block on transcription; it must return a job_id fast."""
    with patch("app.main.CHUNK_SIZE", 10), patch("app.main.MAX_CHUNKS", 100):
        from app.main import _jobs
        upload_id = None
        try:
            resp = client.post(
                "/api/upload/start",
                data={"filename": "test.wav", "size": 10, "total_chunks": 1},
            )
            upload_id = resp.json()["upload_id"]
            for i in range(1):
                client.post(
                    f"/api/upload/chunk/{upload_id}",
                    data={"chunk_index": i},
                    files={"file": (f"test.wav.part{i}", b"X" * 10, "audio/wav")},
                )

            slow_result = {
                "text": "slow result", "language": "en",
                "duration": 1.0, "process_time": 10.0,
                "segments": [], "id": "x",
                "device": "cpu", "compute_type": "int8",
            }

            with patch("app.main.transcribe_audio",
                       side_effect=_make_slow_async_mock(slow_result, delay=10)):
                resp = client.post(f"/api/upload/finish/{upload_id}", data={}, timeout=2.0)
                assert resp.status_code == 200, resp.text
                body = resp.json()
                assert "job_id" in body
                assert body["status"] == "processing"
                job_id = body["job_id"]

                # /finish MUST have removed the upload session (chunks consumed).
                assert not (CHUNK_DIR / upload_id).exists()

                # Status endpoint should report processing.
                _asyncio.sleep(0.5)
                status = None
                for _ in range(5):
                    r = client.get(f"/api/transcribe/status/{job_id}")
                    if r.status_code == 200:
                        status = r.json()
                        break
                assert status is not None
                assert status["status"] == "processing"
                assert "elapsed_seconds" in status

                # Cleanup the running background task + job entry for this test.
                _jobs.pop(job_id, None)
        finally:
            if upload_id:
                _cleanup_upload(upload_id)


def test_status_returns_completed_with_result():
    """/status returns completed+result once transcription finishes."""
    with patch("app.main.CHUNK_SIZE", 10), patch("app.main.MAX_CHUNKS", 100):
        from app.main import _jobs
        upload_id = None
        try:
            resp = client.post(
                "/api/upload/start",
                data={"filename": "test.wav", "size": 10, "total_chunks": 1},
            )
            upload_id = resp.json()["upload_id"]
            client.post(
                f"/api/upload/chunk/{upload_id}",
                data={"chunk_index": 0},
                files={"file": ("test.wav.part0", b"X" * 10, "audio/wav")},
            )

            fast_result = {
                "text": "fast result", "language": "en",
                "duration": 0.5, "process_time": 0.05,
                "segments": [], "id": "x",
                "device": "cpu", "compute_type": "int8",
            }

            with patch("app.main.transcribe_audio",
                       side_effect=_make_async_mock(fast_result)):
                resp = client.post(f"/api/upload/finish/{upload_id}", data={}, timeout=2.0)
                assert resp.status_code == 200
                job_id = resp.json()["job_id"]

                # Poll up to 5s for completion.
                status = None
                for _ in range(50):
                    _asyncio.sleep(0.1)
                    r = client.get(f"/api/transcribe/status/{job_id}")
                    if r.status_code == 200 and r.json().get("status") in ("completed", "failed"):
                        status = r.json()
                        break

                assert status is not None, "job never completed"
                assert status["status"] == "completed"
                assert status["progress"] == 1.0
                assert status["result"]["text"] == "fast result"
        finally:
            if upload_id:
                _cleanup_upload(upload_id)


def test_status_returns_404_for_unknown_job():
    resp = client.get("/api/transcribe/status/zzzzzzzz")
    assert resp.status_code == 404


def test_finish_preserves_session_on_missing_chunks():
    """Sanity: existing behaviour for missing chunks still preserved."""
    with patch("app.main.CHUNK_SIZE", 10), patch("app.main.MAX_CHUNKS", 100):
        upload_id = None
        try:
            resp = client.post(
                "/api/upload/start",
                data={"filename": "test.wav", "size": 30, "total_chunks": 3},
            )
            upload_id = resp.json()["upload_id"]
            for i in [0, 2]:
                client.post(
                    f"/api/upload/chunk/{upload_id}",
                    data={"chunk_index": i},
                    files={"file": (f"test.wav.part{i}", b"X" * 10, "audio/wav")},
                )
            resp = client.post(f"/api/upload/finish/{upload_id}", data={})
            assert resp.status_code == 400
            assert "Missing chunks" in resp.text or "missing" in resp.text.lower()
            assert (CHUNK_DIR / upload_id).exists()
        finally:
            if upload_id:
                _cleanup_upload(upload_id)
