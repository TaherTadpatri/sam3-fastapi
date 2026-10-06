"""Pydantic schemas for SAM3 request and response models."""

from typing import List, Optional, Dict, Any, Literal
from pydantic import BaseModel, Field


MaskFormatType = Literal["rle", "polygon", "base64_png", "binary_mask"]


class SegmentJSONRequest(BaseModel):
    image_base64: str = Field(..., description="Base64 encoded input image (JPEG or PNG).")
    prompt_text: Optional[str] = Field(
        None, 
        description="Text concept prompt for SAM3 (e.g., 'dog', 'person wearing red hat', 'laptop')."
    )
    boxes: Optional[List[List[float]]] = Field(
        None, 
        description="Bounding box prompts in [x1, y1, x2, y2] format."
    )
    threshold: Optional[float] = Field(
        0.20, 
        ge=0.0, 
        le=1.0, 
        description="Detection confidence threshold (default: 0.20)."
    )
    mask_threshold: Optional[float] = Field(
        0.50, 
        ge=0.0, 
        le=1.0, 
        description="Binary threshold applied to predicted mask logits (default: 0.50)."
    )
    mask_format: MaskFormatType = Field(
        "rle", 
        description="Output mask serialization format: 'rle' (recommended for ultra-fast network transfer), 'polygon', 'base64_png', or 'binary_mask'."
    )
    model_override: Optional[Literal["facebook/sam3", "vil-uob/sam3-litetext-s0"]] = Field(
        None,
        description="Optional model override: switch dynamically between 'facebook/sam3' and 'vil-uob/sam3-litetext-s0'."
    )


class DetectionResult(BaseModel):
    id: int
    score: float
    box: List[float] = Field(..., description="Coordinates [x1, y1, x2, y2]")
    area: Optional[int] = Field(None, description="Mask area in pixels")
    mask_rle: Optional[Dict[str, Any]] = Field(None, description="COCO-style Run-Length Encoded mask")
    mask_polygons: Optional[List[List[List[float]]]] = Field(
        None, description="Contour polygons: list of [x, y] coordinates"
    )
    mask_base64: Optional[str] = Field(None, description="Base64 encoded 1-channel PNG mask")
    mask_binary: Optional[List[List[int]]] = Field(None, description="2D nested array (0 or 1)")


class SegmentResponse(BaseModel):
    success: bool = True
    model_id: str
    precision: str
    image_size: List[int] = Field(..., description="[Height, Width] of input image")
    num_detections: int
    presence_score: Optional[float] = Field(
        None, description="Sigmoid presence head logit for the concept in the frame"
    )
    detections: List[DetectionResult]
    inference_time_ms: float
    total_time_ms: float


class HealthResponse(BaseModel):
    status: str = "ok"
    device: str
    gpu_name: Optional[str] = None
    vram_allocated_mb: Optional[float] = None
    vram_reserved_mb: Optional[float] = None
    vram_total_mb: Optional[float] = None


class InfoResponse(BaseModel):
    model_id: str
    precision: str
    device: str
    num_parameters: int
    supported_models: List[str]
    supported_formats: List[str]
    cuda_optimizations: Dict[str, Any]
