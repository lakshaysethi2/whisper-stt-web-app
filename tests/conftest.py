import atexit
import os
import shutil
import tempfile

# Isolate job storage from any real deployment WORK_DIR (/tmp/whisper-stt).
# Must be set BEFORE app.config is imported by the test modules, so this file
# only sets environment variables at module level (no fixtures involved).
_TEST_WORK_DIR = tempfile.mkdtemp(prefix="whisper-stt-tests-")
os.environ["WORK_DIR"] = _TEST_WORK_DIR

# The legacy tests exercise the local-CPU transcription path directly (they
# mock transcribe_audio); the captain's no-CPU default is false, so allow it
# here and have the dispatch tests patch it false where the rule is under
# test. Must be set before app.config is imported.
os.environ["WHISPER_ALLOW_LOCAL_CPU"] = "true"

atexit.register(shutil.rmtree, _TEST_WORK_DIR, ignore_errors=True)
