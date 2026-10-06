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
        data={"prompt_text": "red circle", "threshold": "0.05", "mask_format": "rle"},
    )
    total_elapsed_ms = (time.perf_counter() - t0) * 1000

    assert resp.status_code == 200, f"Multipart segmentation failed: {resp.text}"
    data = resp.json()
    print(f"Multipart segmentation result: success={data['success']}, detections={data['num_detections']}, infer_time={data['inference_time_ms']}ms, client_elapsed={total_elapsed_ms:.2f}ms")
    assert data["num_detections"] > 0, "Expected at least 1 detection for synthetic circle"
    top_det = data["detections"][0]
    assert "box" in top_det
    assert "mask_rle" in top_det
    print(f"Top detection: score={top_det['score']}, box={top_det['box']}, mask_rle_size={top_det['mask_rle']['size']}")
    print("Multipart test passed.")


def test_json_segmentation_formats():
    print("\n--- Testing POST /v1/segment/json with multiple mask formats ---")
    img = create_synthetic_image()
    img_b64 = image_to_base64(img)

    for fmt in ["rle", "polygon", "base64_png"]:
        t0 = time.perf_counter()
        payload = {
            "image_base64": img_b64,
            "prompt_text": "red circle",
            "threshold": 0.05,
            "mask_format": fmt,
        }
        resp = client.post("/v1/segment/json", json=payload)
        elapsed_ms = (time.perf_counter() - t0) * 1000
        assert resp.status_code == 200, f"JSON segmentation failed for format={fmt}: {resp.text}"
        data = resp.json()
        print(f"Format '{fmt}': detections={data['num_detections']}, server_infer={data['inference_time_ms']}ms, total_client={elapsed_ms:.2f}ms")

        if data["num_detections"] > 0:
            det = data["detections"][0]
            if fmt == "rle":
                assert det["mask_rle"] is not None
            elif fmt == "polygon":
                assert det["mask_polygons"] is not None
            elif fmt == "base64_png":
                assert det["mask_base64"] is not None
    print("JSON formats test passed.")


def test_box_prompt():
    print("\n--- Testing Bounding Box Prompt ---")
    img = create_synthetic_image()
    img_b64 = image_to_base64(img)

    payload = {
        "image_base64": img_b64,
        "boxes": [[100.0, 100.0, 300.0, 300.0]],
        "threshold": 0.05,
        "mask_format": "rle",
    }
    resp = client.post("/v1/segment/json", json=payload)
    assert resp.status_code == 200, f"Box prompt failed: {resp.text}"
    data = resp.json()
    print(f"Box prompt: detections={data['num_detections']}, inference_time={data['inference_time_ms']}ms")
    print("Box prompt test passed.")


if __name__ == "__main__":
    print("Starting SAM 3 Test Suite...")
    test_health()
    test_info()
    test_multipart_segmentation()
    test_json_segmentation_formats()
    test_box_prompt()
    print("\n================ ALL TESTS PASSED SUCCESSFULLY! ================\n")
