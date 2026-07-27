# User Requirements

## Bookmarkable Job Resume Links

### Problem
Transcription of large files takes a long time. Users may close the browser while waiting. They need a unique link they can save and reopen later to see the same job progress/result.

### Requirements
1. After a transcription job is created (upload finishes, server returns `job_id`):
   - Browser URL updates to `/j/{job_id}`
   - A "Save this link" control is displayed (copyable URL button)
   - Works on mobile and desktop

2. Reopening `/j/{job_id}` (new tab, after close, on phone):
   - App detects job_id from URL
   - Fetches status from `/api/transcribe/status/{job_id}`
   - Shows appropriate UI: progress (processing), result (completed), error (failed), or expired message (not found)
   - Resumes polling for in-progress jobs

3. Job IDs use full uuid4 hex (128 bits entropy) — not trivially guessable

4. Completed/failed jobs retained for configurable period (`JOB_RETENTION_SECONDS`, default 2 hours)
   - Running jobs are never cleaned up

5. Job status+result persisted to disk as `status.json` under the job directory
   - Survives container restart for completed/failed jobs (best-effort)
   - Running jobs after restart show as expired

6. Both direct upload and chunked upload paths produce bookmarkable job links

## Real Progress Tracking (Bug Fix)

### Problem
- Progress was fake: `min(0.9, elapsed/60.0)` always shows 90% at 60 seconds regardless of actual transcription progress
- Segment generator from faster-whisper was iterated on the asyncio event loop, blocking health/status endpoints for multi-hour files

### Fix
1. Moved entire segment iteration into `to_thread` so event loop is never blocked
2. Real progress: `seg.end / info.duration` (audio position processed / total audio duration)
3. When no real progress data yet (model loading / VAD analyzing), show honest "working, progress unknown"
4. UI shows "working, % unknown" when `progress_note` is "working", else shows real percentage
