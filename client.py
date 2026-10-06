"""Client script for SAM 3 FastAPI service: segment images and save output masks."""

import os
import sys
import argparse
import base64
import io
import json
from typing import Optional, List, Dict, Any

import requests
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont


def decode_rle(rle: Dict[str, Any]) -> np.ndarray:
    """Decode bit-exact COCO Run-Length Encoded (RLE) mask back to a binary numpy array (H, W)."""
    h, w = rle["size"]
    counts = rle["counts"]
    flat = np.zeros(h * w, dtype=np.uint8)
    idx = 0
    val = 0
    for count in counts:
        if val == 1:
            flat[idx : idx + count] = 1
        idx += count
        val = 1 - val
    return flat.reshape((h, w), order="F")


def decode_mask_from_detection(det: Dict[str, Any], target_shape: tuple) -> np.ndarray:
    """Extract binary numpy mask (0 or 1) from detection dictionary regardless of format."""
    if det.get("mask_rle"):
        return decode_rle(det["mask_rle"])
    elif det.get("mask_base64"):
        raw_bytes = base64.b64decode(det["mask_base64"])
        nparr = np.frombuffer(raw_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        return (img > 127).astype(np.uint8)
    elif det.get("mask_binary"):
        return np.array(det["mask_binary"], dtype=np.uint8)
    elif det.get("mask_polygons"):
        mask = np.zeros(target_shape, dtype=np.uint8)
        for poly in det["mask_polygons"]:
            pts = np.array(poly, dtype=np.int32)
            cv2.fillPoly(mask, [pts], 1)
        return mask
    else:
        raise ValueError(f"No valid mask format found in detection {det.get('id')}")


def get_distinct_colors(n: int) -> List[tuple]:
    """Generate distinct visually pleasing RGB colors for each mask."""
    colors = [
        (255, 59, 48),    # Red
        (52, 199, 89),    # Green
        (0, 122, 255),    # Blue
        (255, 149, 0),    # Orange
        (175, 82, 222),   # Purple
        (255, 204, 0),    # Yellow
        (88, 86, 214),    # Indigo
        (255, 45, 85),    # Pink
        (90, 200, 250),   # Light Blue
        (0, 200, 180),    # Teal
    ]
    if n <= len(colors):
        return colors[:n]
    # Fallback to HSV color generation if > 10 objects
    extra = []
    for i in range(n):
        hue = int(180 * i / n)
        col = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2RGB)[0][0]
        extra.append(tuple(map(int, col)))
    return extra


def segment_and_save(
    image_path: str,
    prompt_text: Optional[str] = None,
    boxes: Optional[List[List[float]]] = None,
    threshold: float = 0.35,
    mask_threshold: float = 0.50,
    mask_format: str = "rle",
    refine_box_to_mask: bool = True,
    compute_obb: bool = True,
    box_nms_threshold: float = 0.50,
    filter_by_prompt_boxes: bool = True,
    box_type: str = "obb",
    server_url: str = "http://localhost:8000",
    output_dir: str = "output",
) -> Dict[str, Any]:
    """Send image to SAM 3 FastAPI service and save masks & visual overlays to disk."""
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Input image not found: {image_path}")

    os.makedirs(output_dir, exist_ok=True)

    # 1. Prepare Multipart Request
    url = f"{server_url.rstrip('/')}/v1/segment"
    data = {
        "threshold": threshold,
        "mask_threshold": mask_threshold,
        "mask_format": mask_format,
        "refine_box_to_mask": str(refine_box_to_mask).lower(),
        "compute_obb": str(compute_obb).lower(),
        "box_nms_threshold": str(box_nms_threshold),
        "filter_by_prompt_boxes": str(filter_by_prompt_boxes).lower(),
    }
    if prompt_text:
        data["prompt_text"] = prompt_text
    if boxes:
        data["boxes"] = json.dumps(boxes)

    print(f"\n[+] Sending request to: {url}")
    print(f"    Image:        {image_path}")
    print(f"    Prompt:       {prompt_text or '(None)'}")
    print(f"    Boxes:        {boxes or '(None)'}")
    print(f"    Threshold:    {threshold}")
    print(f"    Refine BBox:  {refine_box_to_mask}")
    print(f"    Compute OBB:  {compute_obb}")
    print(f"    NMS:          {box_nms_threshold}")

    with open(image_path, "rb") as f:
        files = {"file": (os.path.basename(image_path), f, "image/jpeg")}
        resp = requests.post(url, data=data, files=files)

    if resp.status_code != 200:
        raise RuntimeError(f"Server error ({resp.status_code}): {resp.text}")

    result = resp.json()
    detections = result.get("detections", [])
    num_det = len(detections)
    print(f"[✓] Segmentation successful!")
    print(f"    Model:          {result.get('model_id')}")
    print(f"    Inference time: {result.get('inference_time_ms'):.1f} ms")
    print(f"    Detections:     {num_det}")

    if num_det == 0:
        print(f"[!] No objects detected with threshold={threshold}. Try lowering threshold (e.g. -t 0.25).")
        return result

    # 2. Load Original Image for Overlays & Cutouts
    orig_img_rgb = Image.open(image_path).convert("RGB")
    orig_np = np.array(orig_img_rgb)
    h, w = orig_np.shape[:2]

    # Overlay canvas
    overlay_np = orig_np.copy().astype(np.float32)
    colors = get_distinct_colors(num_det)

    print(f"\n[+] Saving outputs to: {output_dir}/")

    for i, det in enumerate(detections):
        det_id = det["id"]
        score = det["score"]
        box = det["box"]
        obb = det.get("obb")
        color = colors[i]

        # Decode binary mask
        mask_binary = decode_mask_from_detection(det, (h, w))

        # A. Save standalone binary mask PNG (0 or 255)
        mask_filename = os.path.join(output_dir, f"mask_{det_id}.png")
        cv2.imwrite(mask_filename, (mask_binary * 255).astype(np.uint8))

        # B. Save RGBA cutout (transparent background with segmented object)
        cutout = np.zeros((h, w, 4), dtype=np.uint8)
        cutout[:, :, :3] = orig_np
        cutout[:, :, 3] = mask_binary * 255
        cutout_filename = os.path.join(output_dir, f"cutout_{det_id}.png")
        cv2.imwrite(cutout_filename, cv2.cvtColor(cutout, cv2.COLOR_RGBA2BGRA))

        # C. Blend colored mask onto overlay canvas (alpha = 0.40)
        color_mask = np.zeros_like(orig_np, dtype=np.float32)
        color_mask[mask_binary == 1] = color
        alpha = 0.40
        mask_indices = mask_binary == 1
        overlay_np[mask_indices] = (
            overlay_np[mask_indices] * (1 - alpha) + color_mask[mask_indices] * alpha
        )

        obb_str = ""
        if obb:
            obb_str = f", OBB=[center={obb['center']}, angle={obb['angle']}°]"
        print(f"    -> Saved mask #{det_id}: score={score:.3f}, box={box}{obb_str} -> {mask_filename}")

    # 3. Draw Bounding Boxes, Contours, and Labels on Overlay
    overlay_uint8 = np.clip(overlay_np, 0, 255).astype(np.uint8)

    # Draw crisp contour boundary lines around each mask
    for i, det in enumerate(detections):
        color = colors[i]
        mask_binary = decode_mask_from_detection(det, (h, w))
        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay_uint8, contours, -1, color, 2)

    # Draw oriented bounding box (OBB) polygons and center points
    for i, det in enumerate(detections):
        color = colors[i]
        obb = det.get("obb")
        if obb and box_type in ("obb", "both"):
            corners = np.array(obb["corners"], dtype=np.int32)
            cv2.polylines(overlay_uint8, [corners], isClosed=True, color=color, thickness=3)
            # Center grasp point
            cx, cy = int(round(obb["center"][0])), int(round(obb["center"][1]))
            cv2.circle(overlay_uint8, (cx, cy), 6, (0, 255, 0), -1)
            cv2.circle(overlay_uint8, (cx, cy), 7, (0, 0, 0), 2)

    overlay_img = Image.fromarray(overlay_uint8)
    draw = ImageDraw.Draw(overlay_img)

    for i, det in enumerate(detections):
        det_id = det["id"]
        score = det["score"]
        x1, y1, x2, y2 = det["box"]
        obb = det.get("obb")
        color = colors[i]

        # Draw axis-aligned bounding box (AABB) if requested
        if box_type in ("aabb", "both"):
            dash_color = color if box_type == "aabb" else (220, 220, 220)
            draw.rectangle([x1, y1, x2, y2], outline=dash_color, width=2)

        # Label position and text
        if obb and box_type in ("obb", "both"):
            label_x = min(pt[0] for pt in obb["corners"])
            label_y = min(pt[1] for pt in obb["corners"])
            label = f"#{det_id} {prompt_text or 'object'}: {score:.2f} ({obb['angle']}°)"
        else:
            label_x, label_y = x1, y1
            label = f"#{det_id} {prompt_text or 'object'}: {score:.2f}"

        label_y = max(0, label_y - 20)
        text_bbox = draw.textbbox((label_x, label_y), label)
        draw.rectangle([text_bbox[0] - 2, text_bbox[1] - 2, text_bbox[2] + 2, text_bbox[3] + 2], fill=color)
        draw.text((label_x, label_y), label, fill=(255, 255, 255))

    overlay_path = os.path.join(output_dir, "overlay.png")
    overlay_img.save(overlay_path)
    print(f"\n[✓] Saved combined visualization overlay: {overlay_path}")

    return result


def create_demo_image(path: str = "demo_input.jpg"):
    """Generate a quick demo image if none is provided."""
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
        description="SAM 3 FastAPI Client: Segment images and save masks/overlays"
    )
    parser.add_argument("--image", "-i", type=str, default=None, help="Path to input image file")
    parser.add_argument(
        "--prompt", "-p", type=str, default="circle", help="Text concept prompt (e.g. 'cat', 'red box')"
    )
    parser.add_argument(
        "--boxes", "-b", type=str, default=None, help="JSON bounding boxes: '[[x1,y1,x2,y2]]'"
    )
    parser.add_argument(
        "--threshold", "-t", type=float, default=0.35, help="Confidence threshold (default: 0.35)"
    )
    parser.add_argument(
        "--box-type", type=str, default="obb", choices=["obb", "aabb", "both"],
        help="Bounding box overlay style: 'obb' (snug rotated box), 'aabb' (upright box), or 'both' (default: obb)"
    )
    parser.add_argument(
        "--format", "-f", type=str, default="rle", choices=["rle", "polygon", "base64_png", "binary_mask"],
        help="Mask transmission format (default: rle)"
    )
    parser.add_argument(
        "--output", "-o", type=str, default="output", help="Directory to save masks (default: output)"
    )
    parser.add_argument(
        "--server", "-s", type=str, default="http://localhost:8000", help="Server URL (default: http://localhost:8000)"
    )
    args = parser.parse_args()

    image_path = args.image
    if not image_path:
        print("[*] No --image argument provided. Generating demo image 'demo_input.jpg'...")
        image_path = create_demo_image()

    parsed_boxes = None
    if args.boxes:
        parsed_boxes = json.loads(args.boxes)

    segment_and_save(
        image_path=image_path,
        prompt_text=args.prompt,
        boxes=parsed_boxes,
        threshold=args.threshold,
        box_type=args.box_type,
        mask_format=args.format,
        server_url=args.server,
        output_dir=args.output,
    )


if __name__ == "__main__":
    main()
