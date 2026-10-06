"""Integration test suite for SAM 3 FastAPI service."""

import base64
import io
import time
import numpy as np
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def create_synthetic_image(width=400, height=400) -> Image.Image:
    """Create a synthetic test image with a red circle on white background."""
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.ellipse((100, 100, 300, 300), fill=(255, 0, 0))
    return img


def image_to_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


def image_to_base64(img: Image.Image) -> str:
    raw_bytes = image_to_bytes(img)
    return base64.b64encode(raw_bytes).decode("utf-8")


def test_health():
    print("\n--- Testing GET /v1/health ---")
    resp = client.get("/v1/health")
    assert resp.status_code == 200, f"Health check failed: {resp.text}"
    data = resp.json()
    print("Health response:", data)
    assert data["status"] == "ok"
    assert "device" in data
    print("Health check passed.")


def test_info():
    print("\n--- Testing GET /v1/info ---")
    resp = client.get("/v1/info")
    assert resp.status_code == 200, f"Info check failed: {resp.text}"
    data = resp.json()
    print(f"Info response: model={data['model_id']}, precision={data['precision']}, params={data['num_parameters']:,}")
    assert "supported_models" in data
    assert "supported_formats" in data
    print("Info check passed.")


def test_multipart_segmentation():
    print("\n--- Testing POST /v1/segment (Multipart) ---")
    img = create_synthetic_image()
    img_bytes = image_to_bytes(img)

    t0 = time.perf_counter()
    resp = client.post(
        "/v1/segment",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        data={"prompt_text": "red circle", "threshold": "0.05"},
    )
    total_elapsed_ms = (time.perf_counter() - t0) * 1000

    assert resp.status_code == 200, f"Multipart segmentation failed: {resp.text}"
    data = resp.json()
    print(f"Multipart segmentation result: {data}")
    print(f"Client elapsed: {total_elapsed_ms:.2f} ms")

    # Verify top-level response fields
    assert "object_name" in data
    assert data["object_name"] == "red circle"
    assert "content_type" in data
    assert data["content_type"] == "image/png"
    assert "content type" in data
    assert data["content type"] == "image/png"
    assert "latency_ms" in data
    assert isinstance(data["latency_ms"], (int, float))
    assert "mask_url" in data
    assert data["mask_url"] is not None
    assert "/static/masks/" in data["mask_url"]

    # Verify results object
    assert "results" in data
    results = data["results"]
    assert results["status"] == "success"
    assert "message" in results
    assert "bbox" in results
    assert isinstance(results["bbox"], list) and len(results["bbox"]) == 4
    assert results["label"] == "red circle"
    assert "score" in results
    assert isinstance(results["score"], float) and results["score"] > 0.0

    # Verify that mask_url is static and downloadable
    mask_resp = client.get(data["mask_url"])
    assert mask_resp.status_code == 200, f"Failed to download mask from {data['mask_url']}"
    assert "image/png" in mask_resp.headers.get("content-type", "")
    assert len(mask_resp.content) > 0
    print(f"[✓] Successfully downloaded mask ({len(mask_resp.content)} bytes) from static URL: {data['mask_url']}")
    print("Multipart test passed.")


def test_json_segmentation():
    print("\n--- Testing POST /v1/segment/json ---")
    img = create_synthetic_image()
    img_b64 = image_to_base64(img)

    t0 = time.perf_counter()
    payload = {
        "image_base64": img_b64,
        "prompt_text": "red circle",
        "threshold": 0.05,
    }
    resp = client.post("/v1/segment/json", json=payload)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    assert resp.status_code == 200, f"JSON segmentation failed: {resp.text}"
    data = resp.json()
    print(f"JSON response: {data}")
    print(f"Client elapsed: {elapsed_ms:.2f} ms")

    assert data["object_name"] == "red circle"
    assert data["content_type"] == "image/png"
    assert data["content type"] == "image/png"
    assert "latency_ms" in data
    assert data["mask_url"] is not None

    results = data["results"]
    assert results["status"] == "success"
    assert results["label"] == "red circle"
    assert results["score"] > 0.5
    assert len(results["bbox"]) == 4

    # Verify mask download
    mask_resp = client.get(data["mask_url"])
    assert mask_resp.status_code == 200
    assert "image/png" in mask_resp.headers.get("content-type", "")
    print("JSON segmentation test passed.")


def test_box_prompt():
    print("\n--- Testing Bounding Box Prompt ---")
    img = create_synthetic_image()
    img_b64 = image_to_base64(img)

    payload = {
        "image_base64": img_b64,
        "boxes": [[100.0, 100.0, 300.0, 300.0]],
        "threshold": 0.05,
    }
    resp = client.post("/v1/segment/json", json=payload)
    assert resp.status_code == 200, f"Box prompt failed: {resp.text}"
    data = resp.json()
    print(f"Box prompt response: {data}")
    assert data["results"]["status"] == "success"
    assert data["results"]["bbox"] is not None
    assert data["mask_url"] is not None
    print("Box prompt test passed.")


def test_root_segment_endpoint():
    print("\n--- Testing Root POST /segment (Alias) ---")
    img = create_synthetic_image()
    img_bytes = image_to_bytes(img)
    resp = client.post(
        "/segment",
        files={"file": ("test.jpg", img_bytes, "image/jpeg")},
        data={"prompt_text": "red circle", "threshold": "0.05"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["results"]["status"] == "success"
    assert data["mask_url"] is not None
    print("Root /segment alias test passed.")


if __name__ == "__main__":
    print("Starting SAM 3 Test Suite...")
    test_health()
    test_info()
    test_multipart_segmentation()
    test_json_segmentation()
    test_box_prompt()
    test_root_segment_endpoint()
    print("\n================ ALL TESTS PASSED SUCCESSFULLY! ================\n")
