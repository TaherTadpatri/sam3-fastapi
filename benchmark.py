"""Benchmark utility for SAM 3 FastAPI inference service."""

import sys
import time
import base64
import io
import json
import numpy as np
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def generate_benchmark_image(width=512, height=512) -> str:
    """Generate a test image with several geometric shapes."""
    img = Image.new("RGB", (width, height), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    # Draw a blue circle
    draw.ellipse((50, 50, 200, 200), fill=(30, 144, 255))
    # Draw a green rectangle
    draw.rectangle((250, 80, 450, 220), fill=(34, 139, 34))
    # Draw a red circle
    draw.ellipse((150, 280, 350, 480), fill=(220, 20, 60))

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def run_benchmarks(num_runs=5):
    print("=" * 65)
    print("        SAM 3 FASTAPI PERFORMANCE BENCHMARK SUITE")
    print("=" * 65)

    img_b64 = generate_benchmark_image()

    # 1. Check GPU and Health
    health_resp = client.get("/v1/health")
    gpu_data = health_resp.json()
    print(f"Device:           {gpu_data.get('device')}")
    print(f"GPU Model:        {gpu_data.get('gpu_name', 'N/A')}")
    print(f"Total VRAM:       {gpu_data.get('vram_total_mb', 0)} MB")
    print(f"Current VRAM:     {gpu_data.get('vram_allocated_mb', 0)} MB")
    print("-" * 65)

    # 2. Benchmark Text Concept Prompt
    print(f"\n[1] Benchmarking Concept Prompt ('circle', {num_runs} runs)...")
    concept_payload = {
        "image_base64": img_b64,
        "prompt_text": "circle",
        "threshold": 0.10,
        "mask_format": "rle",
    }
    concept_times = []
    for i in range(num_runs):
        t0 = time.perf_counter()
        resp = client.post("/v1/segment/json", json=concept_payload)
        client_elapsed = (time.perf_counter() - t0) * 1000
        data = resp.json()
        infer_time = data["latency_ms"]
        concept_times.append(infer_time)
        print(f"  Run #{i+1}: Server Infer = {infer_time:.2f} ms | Client E2E = {client_elapsed:.2f} ms | Status = {data['results']['status']}")

    print(f"  -> Concept Mean Infer Latency: {np.mean(concept_times):.2f} ms (Min: {np.min(concept_times):.2f} ms)")

    # 3. Benchmark Box Prompt
    print(f"\n[2] Benchmarking Bounding Box Prompt ({num_runs} runs)...")
    box_payload = {
        "image_base64": img_b64,
        "boxes": [[50.0, 50.0, 200.0, 200.0]],
        "threshold": 0.10,
        "mask_format": "rle",
    }
    box_times = []
    for i in range(num_runs):
        t0 = time.perf_counter()
        resp = client.post("/v1/segment/json", json=box_payload)
        client_elapsed = (time.perf_counter() - t0) * 1000
        data = resp.json()
        infer_time = data["latency_ms"]
        box_times.append(infer_time)
        print(f"  Run #{i+1}: Server Infer = {infer_time:.2f} ms | Client E2E = {client_elapsed:.2f} ms | Status = {data['results']['status']}")

    print(f"  -> Box Mean Infer Latency:     {np.mean(box_times):.2f} ms (Min: {np.min(box_times):.2f} ms)")

    # 4. Compare Payload Sizes and Serialization Times
    print(f"\n[3] Comparing Output Mask Formats (Payload Size & Serialization Overhead)...")
    for fmt in ["rle", "polygon", "base64_png", "binary_mask"]:
        p = dict(concept_payload)
        p["mask_format"] = fmt
        t0 = time.perf_counter()
        resp = client.post("/v1/segment/json", json=p)
        elapsed = (time.perf_counter() - t0) * 1000
        payload_size_kb = len(resp.content) / 1024
        print(f"  Format: {fmt:<12} | Payload: {payload_size_kb:8.2f} KB | Total Latency: {elapsed:7.2f} ms")

    print("\n" + "=" * 65)
    print("                    BENCHMARK COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    runs = 3
    if len(sys.argv) > 1:
        runs = int(sys.argv[1])
    run_benchmarks(runs)
