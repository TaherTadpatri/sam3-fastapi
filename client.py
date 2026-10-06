"""Client SDK and CLI utility for SAM 3 FastAPI service.

Supports segmenting images, retrieving highest-confidence mask metadata,
and downloading static mask files directly from the server.
"""

import os
import sys
import argparse
import base64
import json
import time
from typing import Optional, List, Dict, Any, Union

import requests
import numpy as np
import cv2
from PIL import Image, ImageDraw


class SAM3Client:
    """Python client for interacting with SAM 3 FastAPI service."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")

    def health(self) -> Dict[str, Any]:
        """Check system and GPU health."""
        url = f"{self.base_url}/v1/health"
        resp = requests.get(url)
        resp.raise_for_status()
        return resp.json()

    def info(self) -> Dict[str, Any]:
        """Get model architecture and optimization metadata."""
        url = f"{self.base_url}/v1/info"
        resp = requests.get(url)
        resp.raise_for_status()
        return resp.json()

    def segment(
        self,
        image: Union[str, bytes, Image.Image],
        prompt_text: Optional[str] = None,
        boxes: Optional[List[List[float]]] = None,
        threshold: float = 0.35,
        mask_threshold: float = 0.50,
        refine_box_to_mask: bool = True,
        method: str = "multipart",
    ) -> Dict[str, Any]:
        """Send an image to SAM 3 for segmentation and return highest-confidence result.

        Args:
            image: File path string, raw image bytes, or PIL Image instance.
            prompt_text: Text concept prompt (e.g. 'dog', 'red circle').
            boxes: Bounding box prompts in [[x1, y1, x2, y2], ...] format.
            threshold: Confidence score threshold (default: 0.35).
            mask_threshold: Mask binarization threshold (default: 0.50).
            refine_box_to_mask: Refine bounding box to tight boundary of mask.
            method: 'multipart' (Form file upload) or 'json' (Base64 payload).

        Returns:
            Dict containing object_name, content_type, latency_ms, mask_url, results.
        """
        # Resolve raw bytes from image input
        if isinstance(image, str):
            if not os.path.exists(image):
                raise FileNotFoundError(f"Input image not found: {image}")
            with open(image, "rb") as f:
                img_bytes = f.read()
            filename = os.path.basename(image)
        elif isinstance(image, bytes):
            img_bytes = image
            filename = "image.jpg"
        elif isinstance(image, Image.Image):
            import io
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=95)
            img_bytes = buf.getvalue()
            filename = "image.jpg"
        else:
            raise TypeError("image must be a file path string, bytes, or PIL Image")

        if method == "multipart":
            url = f"{self.base_url}/v1/segment"
            data: Dict[str, Any] = {
                "threshold": threshold,
                "mask_threshold": mask_threshold,
                "refine_box_to_mask": str(refine_box_to_mask).lower(),
            }
            if prompt_text:
                data["prompt_text"] = prompt_text
            if boxes:
                data["boxes"] = json.dumps(boxes)

            files = {"file": (filename, img_bytes, "image/jpeg")}
            resp = requests.post(url, data=data, files=files)
        else:
            url = f"{self.base_url}/v1/segment/json"
            b64_str = base64.b64encode(img_bytes).decode("utf-8")
            payload: Dict[str, Any] = {
                "image_base64": b64_str,
                "threshold": threshold,
                "mask_threshold": mask_threshold,
                "refine_box_to_mask": refine_box_to_mask,
            }
            if prompt_text:
                payload["prompt_text"] = prompt_text
            if boxes:
                payload["boxes"] = boxes

            resp = requests.post(url, json=payload)

        if resp.status_code != 200:
            raise RuntimeError(f"Server error ({resp.status_code}): {resp.text}")

        return resp.json()

    def download_mask(self, mask_url: str, save_path: str) -> str:
        """Download static mask PNG from server mask_url and save to disk."""
        if not mask_url:
            raise ValueError("Cannot download mask: mask_url is None")

        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        resp = requests.get(mask_url)
        resp.raise_for_status()

        with open(save_path, "wb") as f:
            f.write(resp.content)
        return save_path


def segment_and_save(
    image_path: str,
    prompt_text: Optional[str] = None,
    boxes: Optional[List[List[float]]] = None,
    threshold: float = 0.35,
    mask_threshold: float = 0.50,
    refine_box_to_mask: bool = True,
    method: str = "multipart",
    server_url: str = "http://localhost:8000",
    output_dir: str = "output",
    download_mask: bool = True,
) -> Dict[str, Any]:
    """Segment image via server, download highest confidence static mask, and create overlays."""
    client = SAM3Client(base_url=server_url)

    print(f"\n[+] Requesting SAM 3 segmentation from: {server_url}")
    print(f"    Image:        {image_path}")
    print(f"    Prompt:       {prompt_text or '(None)'}")
    print(f"    Boxes:        {boxes or '(None)'}")
    print(f"    Threshold:    {threshold}")
    print(f"    Method:       {method}")

    t0 = time.perf_counter()
    response = client.segment(
        image=image_path,
        prompt_text=prompt_text,
        boxes=boxes,
        threshold=threshold,
        mask_threshold=mask_threshold,
        refine_box_to_mask=refine_box_to_mask,
        method=method,
    )
    client_elapsed_ms = (time.perf_counter() - t0) * 1000

    object_name = response.get("object_name")
    content_type = response.get("content_type")
    latency_ms = response.get("latency_ms", 0.0)
    mask_url = response.get("mask_url")
    results = response.get("results", {})

    status_str = results.get("status")
    message_str = results.get("message")
    bbox = results.get("bbox")
    label = results.get("label")
    score = results.get("score")

    print(f"\n[✓] Response received from server:")
    print(f"    Object Name:  {object_name}")
    print(f"    Content Type: {content_type}")
    print(f"    Server Time:  {latency_ms:.1f} ms (Total Client: {client_elapsed_ms:.1f} ms)")
    print(f"    Status:       {status_str} ({message_str})")
    print(f"    Label:        {label}")
    print(f"    Score:        {score}")
    print(f"    BBox:         {bbox}")
    print(f"    Mask URL:     {mask_url or '(None)'}")

    if not mask_url or status_str != "success":
        print(f"[!] No object detected above threshold={threshold}.")
        return response

    if not download_mask:
        return response

    os.makedirs(output_dir, exist_ok=True)

    # 1. Download highest-confidence static mask
    mask_save_path = os.path.join(output_dir, "highest_confidence_mask.png")
    client.download_mask(mask_url, mask_save_path)
    print(f"[✓] Downloaded static mask -> {mask_save_path}")

    # 2. Generate Cutout and Visual Overlay
    orig_img = Image.open(image_path).convert("RGB")
    orig_np = np.array(orig_img)
    h, w = orig_np.shape[:2]

    mask_raw = cv2.imread(mask_save_path, cv2.IMREAD_GRAYSCALE)
    if mask_raw is None:
        print("[!] Could not decode downloaded mask image.")
        return response

    binary_mask = (mask_raw > 127).astype(np.uint8)

    # RGBA cutout (transparent background)
    cutout = np.zeros((h, w, 4), dtype=np.uint8)
    cutout[:, :, :3] = orig_np
    cutout[:, :, 3] = binary_mask * 255
    cutout_path = os.path.join(output_dir, "cutout_highest.png")
    cv2.imwrite(cutout_path, cv2.cvtColor(cutout, cv2.COLOR_RGBA2BGRA))
    print(f"[✓] Saved RGBA cutout -> {cutout_path}")

    # Colored overlay (semi-transparent blend)
    overlay_np = orig_np.copy().astype(np.float32)
    color = (255, 59, 48)  # Distinct coral red
    color_mask = np.zeros_like(orig_np, dtype=np.float32)
    color_mask[binary_mask == 1] = color
    alpha = 0.45
    mask_idx = binary_mask == 1
    overlay_np[mask_idx] = overlay_np[mask_idx] * (1 - alpha) + color_mask[mask_idx] * alpha

    overlay_uint8 = np.clip(overlay_np, 0, 255).astype(np.uint8)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay_uint8, contours, -1, color, 2)

    overlay_img = Image.fromarray(overlay_uint8)
    draw = ImageDraw.Draw(overlay_img)

    if bbox and len(bbox) == 4:
        x1, y1, x2, y2 = bbox
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        lbl_text = f"{label or object_name}: {score:.2f}" if score is not None else f"{label or object_name}"
        lbl_y = max(0, y1 - 22)
        text_bbox = draw.textbbox((x1, lbl_y), lbl_text)
        draw.rectangle([text_bbox[0] - 3, text_bbox[1] - 3, text_bbox[2] + 3, text_bbox[3] + 3], fill=color)
        draw.text((x1, lbl_y), lbl_text, fill=(255, 255, 255))

    overlay_path = os.path.join(output_dir, "overlay.png")
    overlay_img.save(overlay_path)
    print(f"[✓] Saved visualization overlay -> {overlay_path}\n")

    response["downloaded_mask_path"] = mask_save_path
    response["cutout_path"] = cutout_path
    response["overlay_path"] = overlay_path

    return response


def create_demo_image(path: str = "demo_input.jpg") -> str:
    """Generate a sample image with shapes for demonstration."""
    img = Image.new("RGB", (640, 480), color=(245, 245, 245))
    draw = ImageDraw.Draw(img)
    # Red circle
    draw.ellipse((80, 80, 280, 280), fill=(220, 20, 60))
    # Blue rectangle
    draw.rectangle((340, 100, 560, 320), fill=(30, 144, 255))
    img.save(path, format="JPEG", quality=95)
    return path


def main():
    parser = argparse.ArgumentParser(
        description="SAM 3 FastAPI Client: Segment images and download static masks"
    )
    parser.add_argument("--image", "-i", type=str, default=None, help="Path to input image file")
    parser.add_argument(
        "--prompt", "-p", type=str, default="red circle", help="Text concept prompt (e.g. 'red circle', 'dog')"
    )
    parser.add_argument(
        "--boxes", "-b", type=str, default=None, help="Bounding box coordinates in JSON format: '[[x1,y1,x2,y2]]'"
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=0.20, help="Detection confidence threshold (default: 0.20)"
    )
    parser.add_argument(
        "--method", "-m", type=str, default="multipart", choices=["multipart", "json"],
        help="Upload method: 'multipart' (default) or 'json' (Base64)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="output", help="Directory to save downloaded mask and overlays (default: output)"
    )
    parser.add_argument(
        "--server", "-s", type=str, default="http://localhost:8000", help="SAM 3 server URL (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--no-download", action="store_true", help="Skip downloading mask file from static mask_url"
    )
    parser.add_argument(
        "--json-output", action="store_true", help="Output full JSON response to stdout"
    )
    args = parser.parse_args()

    image_path = args.image
    if not image_path:
        print("[*] No --image argument provided. Generating demo image 'demo_input.jpg'...")
        image_path = create_demo_image()

    parsed_boxes = None
    if args.boxes:
        parsed_boxes = json.loads(args.boxes)

    res = segment_and_save(
        image_path=image_path,
        prompt_text=args.prompt,
        boxes=parsed_boxes,
        threshold=args.threshold,
        method=args.method,
        server_url=args.server,
        output_dir=args.output,
        download_mask=not args.no_download,
    )

    if args.json_output:
        print("\n--- JSON OUTPUT ---")
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
