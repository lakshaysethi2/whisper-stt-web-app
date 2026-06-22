#!/usr/bin/env python3
"""Benchmark transcription speed across model configs."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def get_gpu_info():
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(", ")
            return {"name": parts[0], "total_mb": int(parts[1]), "used_mb": int(parts[2])}
    except Exception:
        pass
    return None


def benchmark_config(model_name, compute_type, batch_size, audio_path, language="en", use_batched=True, device=None):
    from faster_whisper import WhisperModel, BatchedInferencePipeline

    gpu_before = get_gpu_info()
    if device is None:
        device = "cuda"

    print(f"  Loading {model_name} ({compute_type}, batch={batch_size}, batched={use_batched}, device={device})...")
    load_start = time.perf_counter()
    try:
        model = WhisperModel(model_name, device=device, compute_type=compute_type)
    except ValueError:
        print(f"  {device} {compute_type} not supported, falling back to CPU int8")
        device = "cpu"
        model = WhisperModel(model_name, device="cpu", compute_type="int8")
    load_time = time.perf_counter() - load_start

    gpu_after = get_gpu_info()

    if use_batched:
        pipeline = BatchedInferencePipeline(model=model)
        transcribe_fn = lambda: pipeline.transcribe(
            audio_path,
            language=language,
            batch_size=batch_size,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
            word_timestamps=False,
            condition_on_previous_text=True,
        )
    else:
        transcribe_fn = lambda: model.transcribe(
            audio_path,
            language=language,
            vad_filter=True,
            vad_parameters=dict(min_silence_duration_ms=500, speech_pad_ms=200),
            word_timestamps=False,
            condition_on_previous_text=True,
        )

    print(f"  Transcribing...")
    transcribe_start = time.perf_counter()
    segments, info = transcribe_fn()
    text_parts = []
    total_duration = 0.0
    for seg in segments:
        t = seg.text.strip()
        if t:
            text_parts.append(t)
            total_duration += seg.end - seg.start
    transcribe_time = time.perf_counter() - transcribe_start

    vram_used = None
    if gpu_before and gpu_after:
        vram_used = gpu_after["used_mb"] - gpu_before["used_mb"]

    del model
    if use_batched:
        del pipeline

    return {
        "model": model_name,
        "compute_type": compute_type,
        "batch_size": batch_size,
        "batched": use_batched,
        "load_time_s": round(load_time, 2),
        "transcribe_time_s": round(transcribe_time, 2),
        "audio_duration_s": round(total_duration, 2),
        "realtime_factor": round(total_duration / transcribe_time, 1) if transcribe_time > 0 else 0,
        "text_length": len(" ".join(text_parts)),
        "vram_delta_mb": vram_used,
        "gpu": gpu_after["name"] if gpu_after else None,
    }


def main():
    parser = argparse.ArgumentParser(description="Benchmark Whisper transcription speed")
    parser.add_argument("audio", help="Path to audio file (WAV recommended)")
    parser.add_argument("--language", default="en", help="Language code (default: en)")
    parser.add_argument("--configs", nargs="*", help="Specific configs to test, e.g.: large-v3-turbo:int8:16")
    args = parser.parse_args()

    audio_path = os.path.abspath(args.audio)
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)

    gpu = get_gpu_info()
    if not gpu:
        print("Warning: No NVIDIA GPU detected. Benchmarks will use CPU (very slow).")
    else:
        print(f"GPU: {gpu['name']} ({gpu['total_mb']} MB VRAM)")

    if args.configs:
        configs = []
        for c in args.configs:
            parts = c.split(":")
            if len(parts) == 3:
                configs.append({"model": parts[0], "compute_type": parts[1], "batch_size": int(parts[2])})
            else:
                print(f"Warning: Invalid config format '{c}', expected model:compute_type:batch_size. Skipping.")
        if not configs:
            print("No valid configs. Exiting.")
            sys.exit(1)
    else:
        gpu = get_gpu_info()
        if gpu and "940" in gpu.get("name", ""):
            configs = [
                {"model": "tiny", "compute_type": "int8", "batch_size": 1, "use_batched": False, "device": "cpu"},
                {"model": "tiny", "compute_type": "float32", "batch_size": 1, "use_batched": False, "device": "cuda"},
                {"model": "base", "compute_type": "int8", "batch_size": 1, "use_batched": False, "device": "cpu"},
                {"model": "base.en", "compute_type": "int8", "batch_size": 1, "use_batched": False, "device": "cpu"},
                {"model": "base", "compute_type": "float32", "batch_size": 1, "use_batched": False, "device": "cuda"},
                {"model": "small.en", "compute_type": "int8", "batch_size": 1, "use_batched": False, "device": "cpu"},
                {"model": "tiny", "compute_type": "int8", "batch_size": 1, "use_batched": True, "device": "cpu"},
                {"model": "base.en", "compute_type": "int8", "batch_size": 8, "use_batched": True, "device": "cpu"},
            ]
        else:
            configs = [
                {"model": "large-v3-turbo", "compute_type": "int8", "batch_size": 16},
                {"model": "large-v3-turbo", "compute_type": "float16", "batch_size": 16},
                {"model": "large-v3-turbo", "compute_type": "int8", "batch_size": 1},
                {"model": "large-v3-turbo", "compute_type": "float16", "batch_size": 1},
                {"model": "base", "compute_type": "int8", "batch_size": 16},
                {"model": "base.en", "compute_type": "int8", "batch_size": 16},
                {"model": "small", "compute_type": "int8", "batch_size": 16},
                {"model": "small.en", "compute_type": "int8", "batch_size": 16},
                {"model": "tiny", "compute_type": "int8", "batch_size": 16},
            ]

    print(f"\nAudio: {audio_path}")
    print(f"Language: {args.language}")
    print(f"Configs to test: {len(configs)}")
    print("=" * 70)

    results = []
    for i, cfg in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] {cfg['model']} / {cfg['compute_type']} / batch={cfg['batch_size']}")
        try:
            result = benchmark_config(
                cfg["model"], cfg["compute_type"], cfg["batch_size"],
                audio_path, args.language,
                use_batched=cfg.get("use_batched", True),
                device=cfg.get("device"),
            )
            results.append(result)
            print(f"  -> {result['transcribe_time_s']}s ({result['realtime_factor']}x realtime)")
        except Exception as e:
            print(f"  -> FAILED: {e}")
            results.append({**cfg, "error": str(e)})

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Model':<20} {'Compute':<10} {'Batch':<8} {'Time':<10} {'Speed':<12} {'VRAM':<10}")
    print("-" * 70)
    for r in sorted(results, key=lambda x: x.get("transcribe_time_s", 999)):
        if "error" in r:
            print(f"{r['model']:<20} {r['compute_type']:<10} {r['batch_size']:<8} FAILED")
        else:
            vram = f"{r['vram_delta_mb']}MB" if r["vram_delta_mb"] is not None else "N/A"
            print(f"{r['model']:<20} {r['compute_type']:<10} {r['batch_size']:<8} {r['transcribe_time_s']:<10}s {r['realtime_factor']:<12}x {vram:<10}")

    winner = min((r for r in results if "error" not in r), key=lambda x: x["transcribe_time_s"], default=None)
    if winner:
        print(f"\nFastest: {winner['model']} / {winner['compute_type']} / batch={winner['batch_size']} -> {winner['transcribe_time_s']}s ({winner['realtime_factor']}x)")

    out = Path("benchmark_results.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nFull results saved to {out}")


if __name__ == "__main__":
    main()
