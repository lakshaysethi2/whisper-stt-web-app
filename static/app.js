(() => {
  "use strict";

  let mediaRecorder = null;
  let audioChunks = [];
  let recordingStartTime = 0;
  let timerInterval = null;
  let micPermission = null;

  const $ = (s) => document.querySelector(s);

  const els = {
    recordBtn: $("#record-btn"),
    timer: $("#timer"),
    recordHint: $("#record-hint"),
    fileInput: $("#file-input"),
    fileDrop: $("#file-drop"),
    fileDropText: $("#file-drop-text"),
    fileName: $("#file-name"),
    transcribeFileBtn: $("#transcribe-file-btn"),
    language: $("#language-select"),
    status: $("#status"),
    result: $("#result"),
    resultText: $("#result-text"),
    resultLang: $("#result-lang"),
    resultDuration: $("#result-duration"),
    segmentsWrapper: $("#segments-wrapper"),
    segments: $("#segments"),
    copyBtn: $("#copy-btn"),
    downloadBtn: $("#download-btn"),
    modelBadge: $("#model-badge"),
  };

  const uploadConfig = {
    chunkSize: 5 * 1024 * 1024,
    directUploadThreshold: 50 * 1024 * 1024,
  };

  init();

  async function init() {
    if ("serviceWorker" in navigator) {
      navigator.serviceWorker.register("/static/sw.js").catch(() => {});
    }

    try {
      const [modelsResult, configResult] = await Promise.allSettled([
        fetch("/api/models"),
        fetch("/api/upload/config"),
      ]);

      if (modelsResult.status === "fulfilled" && modelsResult.value.ok) {
        const modelsData = await modelsResult.value.json();
        if (els.modelBadge) els.modelBadge.textContent = modelsData.current;
      }

      if (configResult.status === "fulfilled" && configResult.value.ok) {
        const cfg = await configResult.value.json();
        uploadConfig.chunkSize = cfg.chunk_size || uploadConfig.chunkSize;
        uploadConfig.directUploadThreshold = cfg.direct_upload_threshold || uploadConfig.directUploadThreshold;
      }
    } catch {}

    setupRecording();
    setupFileUpload();
    setupActions();
  }

  // --- Recording ---

  function setupRecording() {
    els.recordBtn.addEventListener("click", toggleRecording);
  }

  async function toggleRecording() {
    if (mediaRecorder && mediaRecorder.state === "recording") {
      stopRecording();
      return;
    }
    await startRecording();
  }

  async function startRecording() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      showToast("Recording not supported in this browser");
      return;
    }

    let stream;
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          sampleRate: 16000,
        },
      });
    } catch (err) {
      if (err.name === "NotAllowedError") {
        showToast("Microphone permission denied. Please allow mic access in your browser settings.");
      } else if (err.name === "NotFoundError") {
        showToast("No microphone found on this device");
      } else {
        showToast("Could not access microphone: " + err.message);
      }
      return;
    }

    const mimeType = MediaRecorder.isTypeSupported("audio/webm;codecs=opus")
      ? "audio/webm;codecs=opus"
      : MediaRecorder.isTypeSupported("audio/webm")
        ? "audio/webm"
        : "audio/mp4";

    try {
      mediaRecorder = new MediaRecorder(stream, { mimeType });
    } catch {
      mediaRecorder = new MediaRecorder(stream);
    }

    audioChunks = [];

    mediaRecorder.ondataavailable = (e) => {
      if (e.data.size > 0) audioChunks.push(e.data);
    };

    mediaRecorder.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      const blob = new Blob(audioChunks, { type: mediaRecorder.mimeType });
      if (blob.size > 0) {
        const ext = mediaRecorder.mimeType.includes("mp4") ? "m4a" : "webm";
        const file = new File([blob], `recording.${ext}`, { type: mediaRecorder.mimeType });
        uploadFile(file);
      }
    };

    mediaRecorder.onerror = () => {
      showToast("Recording error occurred");
      resetRecordingUI();
    };

    mediaRecorder.start(500);
    recordingStartTime = Date.now();
    els.recordBtn.classList.add("recording");
    els.timer.classList.remove("hidden");
    els.recordHint.textContent = "Tap to stop";
    timerInterval = setInterval(updateTimer, 100);
  }

  function stopRecording() {
    if (mediaRecorder && mediaRecorder.state === "recording") {
      mediaRecorder.stop();
    }
    resetRecordingUI();
  }

  function resetRecordingUI() {
    els.recordBtn.classList.remove("recording");
    clearInterval(timerInterval);
    els.timer.classList.add("hidden");
    els.timer.textContent = "00:00.0";
    els.recordHint.textContent = "Tap to start recording";
  }

  function updateTimer() {
    const elapsed = (Date.now() - recordingStartTime) / 1000;
    els.timer.textContent = formatTime(elapsed);
  }

  function formatTime(secs) {
    const m = Math.floor(secs / 60);
    const s = Math.floor(secs % 60);
    const d = Math.floor((secs % 1) * 10);
    return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}.${d}`;
  }

  // --- File Upload ---

  function setupFileUpload() {
    els.fileDrop.addEventListener("click", () => els.fileInput.click());

    els.fileDrop.addEventListener("dragover", (e) => {
      e.preventDefault();
      e.stopPropagation();
      els.fileDrop.classList.add("dragover");
    });

    els.fileDrop.addEventListener("dragleave", (e) => {
      e.preventDefault();
      e.stopPropagation();
      els.fileDrop.classList.remove("dragover");
    });

    els.fileDrop.addEventListener("drop", (e) => {
      e.preventDefault();
      e.stopPropagation();
      els.fileDrop.classList.remove("dragover");
      if (e.dataTransfer.files.length) selectFile(e.dataTransfer.files[0]);
    });

    els.fileInput.addEventListener("change", () => {
      if (els.fileInput.files.length) selectFile(els.fileInput.files[0]);
    });
  }

  function selectFile(file) {
    const ext = file.name.split(".").pop().toLowerCase();
    const allowed = [
      "mp3", "wav", "flac", "ogg", "m4a", "aac", "wma", "opus", // Audio
      "mp4", "webm", "avi", "mov", "mkv", "flv", "wmv", "mpeg", "mpg", "3gp", "m4v", "asf" // Video
    ];
    if (!allowed.includes(ext)) {
      showToast(`Unsupported format: .${ext}`);
      return;
    }
    const sizeMB = (file.size / (1024 * 1024)).toFixed(1);
    els.fileDropText.textContent = "File selected";
    els.fileName.textContent = `${file.name} (${sizeMB} MB)`;
    els.fileName.classList.remove("hidden");
    els.transcribeFileBtn.classList.remove("hidden");
    els.transcribeFileBtn.onclick = () => uploadFile(file);
  }

  // --- Upload & Transcribe ---

  const MAX_CONCURRENT_CHUNKS = 4;
  const MAX_CHUNK_RETRIES = 3;
  const RETRY_BASE_DELAY_MS = 1000;

  function uploadFile(file) {
    els.transcribeFileBtn.disabled = true;
    els.recordBtn.disabled = true;

    if (file.size <= uploadConfig.directUploadThreshold) {
      return directUpload(file);
    }
    return chunkedUpload(file);
  }

  function directUpload(file) {
    const form = new FormData();
    form.append("file", file);
    if (els.language.value) form.append("language", els.language.value);

    showStatus("Uploading: 0%...");

    return new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();

      xhr.upload.addEventListener("progress", (e) => {
        if (e.lengthComputable) {
          const percent = Math.round((e.loaded / e.total) * 100);
          showStatus(`Uploading: ${percent}%...`);
        }
      });

      xhr.onload = () => {
        if (xhr.status >= 200 && xhr.status < 300) {
          try {
            const data = JSON.parse(xhr.responseText);
            resolve(data);
          } catch (err) {
            reject(new Error("Invalid response from server"));
          }
        } else {
          try {
            const err = JSON.parse(xhr.responseText);
            reject(new Error(errorMessage(err, `Server error: ${xhr.statusText}`)));
          } catch (err) {
            reject(new Error(`Server error: ${xhr.status} ${xhr.statusText}`));
          }
        }
      };

      xhr.onerror = () => {
        reject(new Error("Network error or connection lost"));
      };

      xhr.onabort = () => {
        reject(new Error("Upload aborted"));
      };

      xhr.upload.onloadend = () => {
        showStatus("Transcribing... This may take a moment.");
      };

      xhr.open("POST", "/api/transcribe");
      xhr.send(form);
    })
    .then((data) => {
      showResult(data);
    })
    .catch((err) => {
      showToast(err.message);
      hideStatus();
    })
    .finally(() => {
      resetUploadUI();
    });
  }

  async function uploadChunkWithRetry(uploadId, chunkIndex, chunk, fileName, totalChunks) {
    for (let attempt = 0; attempt <= MAX_CHUNK_RETRIES; attempt++) {
      try {
        const form = new FormData();
        form.append("chunk_index", chunkIndex);
        form.append("file", chunk, `${fileName}.part${chunkIndex}`);

        const res = await fetch(`/api/upload/chunk/${uploadId}`, { method: "POST", body: form });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(errorMessage(err, `Chunk ${chunkIndex + 1} failed: ${res.statusText}`));
        }
        return await res.json();
      } catch (err) {
        if (attempt === MAX_CHUNK_RETRIES) throw err;
        const delay = RETRY_BASE_DELAY_MS * Math.pow(2, attempt);
        await new Promise((r) => setTimeout(r, delay));
      }
    }
  }

  async function chunkedUpload(file) {
    const totalChunks = Math.ceil(file.size / uploadConfig.chunkSize);
    const sizeMB = (file.size / (1024 * 1024)).toFixed(1);

    const startForm = new FormData();
    startForm.append("filename", file.name);
    startForm.append("size", file.size);
    startForm.append("total_chunks", totalChunks);

    let uploadId;
    try {
      const res = await fetch("/api/upload/start", { method: "POST", body: startForm });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(errorMessage(err, `Start failed: ${res.statusText}`));
      }
      const data = await res.json();
      uploadId = data.upload_id;
    } catch (err) {
      showToast(err.message);
      resetUploadUI();
      return;
    }

    let completedChunks = 0;
    let nextChunkIndex = 0;

    async function worker() {
      while (nextChunkIndex < totalChunks) {
        const i = nextChunkIndex++;
        const start = i * uploadConfig.chunkSize;
        const end = Math.min(start + uploadConfig.chunkSize, file.size);
        const chunk = file.slice(start, end);

        await uploadChunkWithRetry(uploadId, i, chunk, file.name, totalChunks);
        completedChunks++;
        showStatus(`Uploading ${sizeMB} MB file: ${completedChunks}/${totalChunks} chunks...`);
      }
    }

    try {
      const workers = [];
      const concurrency = Math.min(MAX_CONCURRENT_CHUNKS, totalChunks);
      for (let w = 0; w < concurrency; w++) {
        workers.push(worker());
      }
      await Promise.all(workers);
    } catch (err) {
      showToast(err.message);
      resetUploadUI();
      return;
    }

    const finishForm = new FormData();
    if (els.language.value) finishForm.append("language", els.language.value);

    try {
      showStatus("Transcribing... This may take a moment.");
      const res = await fetch(`/api/upload/finish/${uploadId}`, { method: "POST", body: finishForm });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(errorMessage(err, `Transcription failed: ${res.statusText}`));
      }
      const data = await res.json();
      showResult(data);
    } catch (err) {
      showToast(err.message);
    } finally {
      resetUploadUI();
    }
  }

  function resetUploadUI() {
    els.transcribeFileBtn.disabled = false;
    els.recordBtn.disabled = false;
    resetFileInput();
    hideStatus();
  }

  // --- Results ---

  function showResult(data) {
    hideStatus();
    els.result.classList.remove("hidden");
    els.resultText.textContent = data.text || "(no speech detected)";
    els.resultLang.textContent = data.language || "unknown";
    els.resultDuration.textContent = data.duration ? `${data.duration.toFixed(1)}s` : "";

    els.segments.innerHTML = "";
    if (data.segments && data.segments.length > 1) {
      els.segmentsWrapper.classList.remove("hidden");
      data.segments.forEach((seg) => {
        const div = document.createElement("div");
        div.className = "segment";
        const start = seg.t0 != null ? seg.t0 / 1000 : 0;
        div.innerHTML = `<span class="seg-time">${formatTime(start)}</span><span class="seg-text">${escapeHtml(seg.text || "")}</span>`;
        els.segments.appendChild(div);
      });
    } else {
      els.segmentsWrapper.classList.add("hidden");
    }

    els.result.scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  // --- Actions ---

  function setupActions() {
    els.copyBtn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(els.resultText.textContent);
        els.copyBtn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"/></svg> Copied`;
        setTimeout(() => {
          els.copyBtn.innerHTML = `<svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><rect width="14" height="14" x="8" y="8" rx="2" ry="2"/><path d="M4 16c-1.1 0-2-.9-2-2V4c0-1.1.9-2 2-2h10c1.1 0 2 .9 2 2"/></svg> Copy`;
        }, 2000);
      } catch {
        showToast("Copy failed");
      }
    });

    els.downloadBtn.addEventListener("click", () => {
      const blob = new Blob([els.resultText.textContent], { type: "text/plain" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "transcription.txt";
      a.click();
      URL.revokeObjectURL(url);
    });
  }

  // --- Utilities ---

  function errorMessage(err, fallback) {
    if (typeof err.detail === "string") return err.detail;
    if (err.detail?.message) return err.detail.message;
    return fallback;
  }

  function showStatus(msg) {
    els.status.classList.remove("hidden");
    els.status.innerHTML = `<div class="spinner"></div><span>${escapeHtml(msg)}</span>`;
  }

  function hideStatus() {
    els.status.classList.add("hidden");
    els.status.innerHTML = "";
  }

  function resetFileInput() {
    els.fileInput.value = "";
    els.fileDropText.textContent = "Drop audio/video file or tap to browse";
    els.fileName.classList.add("hidden");
    els.transcribeFileBtn.classList.add("hidden");
  }

  function showToast(msg) {
    const existing = document.querySelector(".toast");
    if (existing) existing.remove();
    const toast = document.createElement("div");
    toast.className = "toast";
    toast.textContent = msg;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
  }

  function escapeHtml(str) {
    const d = document.createElement("div");
    d.textContent = str;
    return d.innerHTML;
  }
})();