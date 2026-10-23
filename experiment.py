#!/usr/bin/env python
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Measure the latency impact of early UUID-based multimodal cache lookups.

Usage (with .venv activated):
    ./experiment.py          # start server if needed, benchmark
    ./experiment.py start    # run the server in the foreground
    ./experiment.py stop     # kill the server
"""

import argparse
import functools
import http.server
import json
import os
import random
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

import cv2
import httpx
import psutil

from vllm.assets.base import VLLM_S3_BUCKET_URL

MODEL = "Qwen/Qwen3-VL-2B-Instruct"
SERVE_ARGS = [
    "--max-model-len", "16384",
    "--gpu-memory-utilization", "0.85",
    "--limit-mm-per-prompt", json.dumps({"video": 1, "image": 0}),
    "--mm-processor-kwargs", json.dumps({"fps": 2, "max_pixels": 230400}),
    "--max-num-seqs", "4",
]  # fmt: skip
STARTUP_TIMEOUT_S = 900

SOURCE_URL = f"{VLLM_S3_BUCKET_URL}/multimodal_asset/slow_traffic_small.mp4"
ASSET_DIR = Path(__file__).parent
ASSET_PATH = ASSET_DIR / "asset.mp4"
SERVER_LOG_PATH = ASSET_DIR / "experiment_server.log"
TARGET_SECONDS = 30

NUM_QUESTIONS_PER_BATCH = 10
WARMUP_QUESTION = "Tell me about this video."
QUESTIONS = [
    "Are there cars in this video?",
    "Is it nighttime in this video?",
    "Is there a road in this video?",
    "Are there any boats in this video?",
    "Is the camera filming from above?",
    "Is it snowing in this video?",
    "Are there trees in this video?",
    "Is there an airplane in this video?",
    "Is the traffic moving?",
    "Is this video filmed indoors?",
    "Are there pedestrians in this video?",
    "Is there a bus in this video?",
    "Is it raining in this video?",
    "Are there traffic lights in this video?",
    "Is there a bicycle in this video?",
    "Is the sky visible in this video?",
    "Are there buildings in this video?",
    "Is there a truck in this video?",
    "Is this video in black and white?",
    "Is the road wet?",
    "Are there any animals in this video?",
    "Is there a motorcycle in this video?",
    "Is the camera shaking?",
    "Are there more than five vehicles visible?",
    "Is this a highway?",
    "Is there a bridge in this video?",
    "Are there any people running?",
    "Is there a red car in this video?",
    "Is this filmed in a desert?",
    "Are there road markings visible?",
    "Is there a traffic jam?",
    "Is any vehicle turning?",
    "Is there text visible in this video?",
    "Are there street lamps in this video?",
    "Is this video filmed at sunset?",
    "Is there a parking lot in this video?",
    "Are the vehicles driving on the right side?",
    "Is there a police car in this video?",
    "Is there water visible in this video?",
    "Are there mountains in this video?",
]


def serve_command(port: int) -> list[str]:
    vllm_bin = Path(sys.executable).parent / "vllm"
    return [str(vllm_bin), "serve", MODEL, "--port", str(port), *SERVE_ARGS]


def is_ready(base_url: str) -> bool:
    try:
        return httpx.get(f"{base_url}/health", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


def ensure_server(base_url: str, port: int) -> None:
    """Start the server in the background unless one is already serving."""
    if is_ready(base_url):
        print(f"Server already running at {base_url}")
        return

    print(f"Starting server in the background, logging to {SERVER_LOG_PATH}")
    with open(SERVER_LOG_PATH, "w") as log:
        proc = subprocess.Popen(
            serve_command(port),
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    deadline = time.monotonic() + STARTUP_TIMEOUT_S
    while not is_ready(base_url):
        if proc.poll() is not None:
            sys.exit(f"Server exited with code {proc.returncode}, see log")
        if time.monotonic() > deadline:
            sys.exit(f"Server not ready after {STARTUP_TIMEOUT_S}s, see log")
        time.sleep(2)
    print(f"Server ready (pid {proc.pid})")


def find_server_processes() -> list[psutil.Process]:
    procs = []
    for proc in psutil.process_iter(["pid", "cmdline"]):
        cmdline = proc.info["cmdline"] or []
        if proc.pid == os.getpid():
            continue
        if any(arg.endswith("vllm") for arg in cmdline) and (
            "serve" in cmdline and MODEL in cmdline
        ):
            procs.append(proc)
    return procs


def stop_server() -> None:
    procs = find_server_processes()
    if not procs:
        print("No server running")
        return
    for proc in procs:
        print(f"Stopping pid {proc.pid}")
        proc.terminate()
    _, alive = psutil.wait_procs(procs, timeout=30)
    for proc in alive:
        print(f"Killing pid {proc.pid}")
        proc.kill()


def prepare_asset() -> Path:
    """Download the test video and loop it to TARGET_SECONDS."""
    if ASSET_PATH.exists():
        return ASSET_PATH

    source_path = ASSET_DIR / "asset_source.mp4"
    urllib.request.urlretrieve(SOURCE_URL, source_path)

    cap = cv2.VideoCapture(str(source_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    source_path.unlink()

    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(ASSET_PATH), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    target_frames = int(TARGET_SECONDS * fps)
    for i in range(target_frames):
        writer.write(frames[i % len(frames)])
    writer.release()

    print(
        f"Prepared {ASSET_PATH} ({target_frames} frames, {fps:.1f} fps, "
        f"{width}x{height}, source was {len(frames) / fps:.1f}s)"
    )
    return ASSET_PATH


class CountingHandler(http.server.SimpleHTTPRequestHandler):
    """Counts video fetches instead of logging each request."""

    fetches = 0

    def do_GET(self) -> None:
        type(self).fetches += 1
        super().do_GET()

    def log_message(self, format, *args) -> None:
        pass


def serve_asset_dir() -> str:
    """Serve ASSET_DIR over HTTP so the server fetches the video like a URL."""
    handler = functools.partial(CountingHandler, directory=str(ASSET_DIR))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{httpd.server_address[1]}/{ASSET_PATH.name}"


def send(
    client: httpx.Client,
    video_url: str,
    uuid: str,
    question: str,
    skip_early_mm_lookup: bool,
) -> tuple[float, str]:
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": video_url},
                        "uuid": uuid,
                    },
                    {"type": "text", "text": f"{question} Answer yes or no."},
                ],
            }
        ],
        "max_tokens": 1,
        "temperature": 0.0,
        "skip_early_mm_lookup": skip_early_mm_lookup,
    }
    start = time.perf_counter()
    response = client.post("/v1/chat/completions", json=payload)
    latency = time.perf_counter() - start
    response.raise_for_status()
    return latency, response.json()["choices"][0]["message"]["content"]


def run_batch(
    client: httpx.Client,
    video_url: str,
    uuid: str,
    questions: list[str],
    skip_early_mm_lookup: bool,
) -> list[float]:
    label = "skip_early_mm_lookup" if skip_early_mm_lookup else "early_lookup"
    print(f"\n--- {label} ---")
    latencies = []
    print(f"{'latency':>11}  {'fetches':>7}  {'answer':8}  question")
    for question in questions:
        fetches_before = CountingHandler.fetches
        latency, answer = send(client, video_url, uuid, question, skip_early_mm_lookup)
        latencies.append(latency)
        fetches = CountingHandler.fetches - fetches_before
        print(f"{latency * 1000:8.1f} ms  {fetches:7}  {answer!r:8}  {question}")
    return latencies


def summarize(name: str, latencies: list[float]) -> None:
    ms = sorted(x * 1000 for x in latencies)
    print(
        f"{name:22} mean {statistics.mean(ms):8.1f}  "
        f"median {statistics.median(ms):8.1f}  "
        f"min {ms[0]:8.1f}  max {ms[-1]:8.1f}  (ms)"
    )


def benchmark(base_url: str, port: int) -> None:
    ensure_server(base_url, port)

    uuid = f"video-{time.time_ns()}"
    print(f"UUID: {uuid}")

    prepare_asset()
    video_url = serve_asset_dir()
    print(f"Video URL: {video_url}")

    # Distinct prompts avoid full-prompt prefix cache hits across requests.
    questions = random.sample(QUESTIONS, 2 * NUM_QUESTIONS_PER_BATCH)
    assert WARMUP_QUESTION not in questions

    with httpx.Client(base_url=base_url, timeout=600) as client:
        latency, answer = send(
            client, video_url, uuid, WARMUP_QUESTION, skip_early_mm_lookup=True
        )
        print(f"\nWarmup (cache miss): {latency * 1000:.1f} ms  {answer!r}")

        early = run_batch(
            client,
            video_url,
            uuid,
            questions[:NUM_QUESTIONS_PER_BATCH],
            skip_early_mm_lookup=False,
        )
        skipped = run_batch(
            client,
            video_url,
            uuid,
            questions[NUM_QUESTIONS_PER_BATCH:],
            skip_early_mm_lookup=True,
        )

    print()
    summarize("early_lookup", early)
    summarize("skip_early_mm_lookup", skipped)
    speedup = statistics.mean(skipped) / statistics.mean(early)
    print(f"Mean speedup from early lookup: {speedup:.2f}x")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "mode", nargs="?", default="run", choices=["run", "start", "stop"]
    )
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    base_url = f"http://localhost:{args.port}"

    if args.mode == "start":
        os.execv(serve_command(args.port)[0], serve_command(args.port))
    elif args.mode == "stop":
        stop_server()
    else:
        benchmark(base_url, args.port)


if __name__ == "__main__":
    main()
