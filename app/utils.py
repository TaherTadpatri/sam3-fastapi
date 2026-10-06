"""High-speed image preprocessing and mask encoding utilities."""

import base64
import io
from typing import List, Dict, Any
import numpy as np
import cv2
from PIL import Image


def decode_image_bytes(image_bytes: bytes) -> Image.Image:
    """Decode raw image bytes into a PIL RGB Image."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return image


def decode_base64_image(base64_str: str) -> Image.Image:
    """Decode base64 (or data URI) string into a PIL RGB Image."""
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]
    image_bytes = base64.b64decode(base64_str)
    return decode_image_bytes(image_bytes)


def mask_to_rle(binary_mask: np.ndarray) -> Dict[str, Any]:
    """Convert binary mask (H, W) to bit-exact COCO Run-Length Encoding (RLE).
    
    Uses Fortran column-major order with alternating background/foreground counts.
    """
    pixels = binary_mask.astype(np.uint8).flatten(order="F")
    if len(pixels) == 0:
        return {"size": list(binary_mask.shape), "counts": []}
    where = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs = np.concatenate([[0], where, [len(pixels)]])
    counts = np.diff(runs).tolist()
    if pixels[0] == 1:
        counts = [0] + counts
        
    return {
        "size": [int(binary_mask.shape[0]), int(binary_mask.shape[1])],
        "counts": counts,
    }


def mask_to_polygons(binary_mask: np.ndarray, min_area: float = 10.0) -> List[List[List[float]]]:
    """Extract exterior boundary contours / polygons from binary mask using OpenCV."""
    mask_uint8 = (binary_mask * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    polygons: List[List[List[float]]] = []
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        # Squeeze to shape (N, 2)
        pts = contour.squeeze(axis=1)
        if len(pts.shape) == 2 and pts.shape[0] >= 3:
            polygons.append(pts.astype(float).tolist())
            
    return polygons


def mask_to_base64_png(binary_mask: np.ndarray) -> str:
    """Encode binary mask as a 1-channel grayscale PNG in base64."""
    mask_uint8 = (binary_mask * 255).astype(np.uint8)
    success, buffer = cv2.imencode(".png", mask_uint8)
    if not success:
        raise ValueError("Failed to encode mask to PNG")
    return base64.b64encode(buffer).decode("utf-8")


def format_mask(binary_mask: np.ndarray, mask_format: str) -> Dict[str, Any]:
    """Format binary mask into requested representation."""
    area = int(np.sum(binary_mask))
    res: Dict[str, Any] = {"area": area}
    
    if mask_format == "rle":
        res["mask_rle"] = mask_to_rle(binary_mask)
    elif mask_format == "polygon":
        res["mask_polygons"] = mask_to_polygons(binary_mask)
    elif mask_format == "base64_png":
        res["mask_base64"] = mask_to_base64_png(binary_mask)
    elif mask_format == "binary_mask":
        res["mask_binary"] = binary_mask.astype(int).tolist()
    else:
        # Default to RLE for network efficiency
        res["mask_rle"] = mask_to_rle(binary_mask)
        
    return res
