#!/usr/bin/env python3
"""API load test for the /api/transcribe endpoint."""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

import aiohttp


async def transcribe_one(client, url, audio_path, language="en"):
    filename = os.path.basename(audio_path)
    start = time.perf_counter()
    try:
        with open(audio_path, "rb") as f:
            data = aiohttp.FormData()
            data.add_field("file", f, filename=filename, content_type="audio/wav")
            data.add_field("language", language)
            async with client.post(url, data=data) as resp:
                body = await resp.json()
                elapsed = time.perf_counter() - start
                return {"status": resp.status, "time_s": round(elapsed, 3), "text_length": len(body.get("text", "")), "error": None}
    except Exception as e:
        elapsed = time.perf_counter() - start
        return {"status": 0, "time_s": round(elapsed, 3), "text_length": 0, "error": str(e)}


async def run_load_test(base_url, audio_path, concurrency, total_requests, language="en"):
    import aiohttp

    url = f"{base_url}/api/transcribe"
    print(f"Target: {url}")
    print(f"Audio: {audio_path}")
    print(f"Concurrency: {concurrency}, Total requests: {total_requests}")
    print("=" * 60)

    semaphore = asyncio.Semaphore(concurrency)
    results = []

    async def bounded_transcribe(client, idx):
        async with semaphore:
            print(f"  Request {idx+1}/{total_requests} started...")
            result = await transcribe_one(client, url, audio_path, language)
            results.append(result)
            status = "OK" if result["status"] == 200 else f"ERR({result['status']})"
            print(f"  Request {idx+1}/{total_requests} done: {status} in {result['time_s']}s")
            return result

    wall_start = time.perf_counter()
    async with aiohttp.ClientSession() as client:
        tasks = [bounded_transcribe(client, i) for i in range(total_requests)]
        await asyncio.gather(*tasks)
    wall_time = time.perf_counter() - wall_start

    print("\n" + "=" * 60)
    print("LOAD TEST RESULTS")
    print("=" * 60)

    successful = [r for r in results if r["status"] == 200]
    failed = [r for r in results if r["status"] != 200]
    times = [r["time_s"] for r in successful]

    print(f"Total requests: {total_requests}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    print(f"Wall time: {wall_time:.2f}s")

    if successful:
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        p50 = sorted(times)[len(times) // 2]
        p95 = sorted(times)[int(len(times) * 0.95)]
        p99 = sorted(times)[int(len(times) * 0.99)]
        throughput = len(successful) / wall_time

        print(f"\nLatency (successful requests):")
        print(f"  Avg:  {avg_time:.3f}s")
        print(f"  Min:  {min_time:.3f}s")
        print(f"  Max:  {max_time:.3f}s")
        print(f"  P50:  {p50:.3f}s")
        print(f"  P95:  {p95:.3f}s")
        print(f"  P99:  {p99:.3f}s")
        print(f"\nThroughput: {throughput:.2f} req/s")

    if failed:
        print(f"\nFailed requests:")
        for i, f in enumerate(failed[:10]):
            print(f"  #{i+1}: status={f['status']}, error={f['error']}")

    out = Path("load_test_results.json")
    out.write_text(json.dumps({"wall_time_s": round(wall_time, 2), "concurrency": concurrency,
                               "total_requests": total_requests, "results": results}, indent=2))
    print(f"\nFull results saved to {out}")


def main():
    parser = argparse.ArgumentParser(description="Load test the transcription API")
    parser.add_argument("--url", default="http://localhost:8000", help="Server base URL")
    parser.add_argument("audio", help="Path to audio file for transcription")
    parser.add_argument("-c", "--concurrency", type=int, default=1, help="Concurrent requests (default: 1)")
    parser.add_argument("-n", "--requests", type=int, default=5, help="Total requests (default: 5)")
    parser.add_argument("--language", default="en", help="Language code (default: en)")
    args = parser.parse_args()

    audio_path = os.path.abspath(args.audio)
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found: {audio_path}")
        sys.exit(1)

    asyncio.run(run_load_test(args.url, audio_path, args.concurrency, args.requests, args.language))


if __name__ == "__main__":
    main()
