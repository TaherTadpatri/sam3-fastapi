"""FastAPI Router and Endpoints for SAM 3 Segmentation Service."""

import json
import logging
import torch
from typing import Optional, List
from fastapi import APIRouter, File, UploadFile, Form, HTTPException, Query, status, Request

from app.config import settings
from app.schemas import (
    SegmentJSONRequest,
    SegmentResponse,
    HealthResponse,
    InfoResponse,
    MaskFormatType,
)
from app.utils import decode_image_bytes, decode_base64_image
from app.model_manager import ModelManager

logger = logging.getLogger("sam3.api")
router = APIRouter(tags=["SAM3 Segmentation"])


@router.post(
    "/segment",
    response_model=SegmentResponse,
    summary="Segment Image via File Upload (Multipart Form)",
    description="Upload an image file with an optional text concept prompt or bounding boxes.",
)
async def segment_image_multipart(
    request: Request,
    file: UploadFile = File(..., description="Image file (JPEG/PNG/WEBP)."),
    prompt_text: Optional[str] = Form(
        None, description="Text concept prompt (e.g. 'dog', 'sports car', 'person')."
    ),
    boxes: Optional[str] = Form(
        None, description="JSON string of bounding box coordinates: '[[x1, y1, x2, y2], ...]'"
    ),
    threshold: float = Form(
        0.35, ge=0.0, le=1.0, description="Detection confidence threshold (default: 0.35)."
    ),
    mask_threshold: float = Form(
        0.50, ge=0.0, le=1.0, description="Mask binarization threshold (default: 0.50)."
    ),
    mask_format: MaskFormatType = Form(
        "rle", description="Mask format: 'rle', 'polygon', 'base64_png', or 'binary_mask'."
    ),
    refine_box_to_mask: bool = Form(
        True, description="Refine bounding box to exact tight boundary of predicted mask."
    ),
    compute_obb: bool = Form(
        True, description="Compute oriented bounding box (OBB) hugging rotated objects."
    ),
    box_nms_threshold: Optional[float] = Form(
        0.50, description="Non-Maximum Suppression (NMS) IoU threshold (default: 0.50)."
    ),
    filter_by_prompt_boxes: bool = Form(
        True, description="When prompt boxes are given, only return prompt-matched detections."
    ),
    model_override: Optional[str] = Form(
        None, description="Optional override model: 'facebook/sam3' or 'vil-uob/sam3-litetext-s0'."
    ),
) -> SegmentResponse:
    try:
        image_bytes = await file.read()
        image = decode_image_bytes(image_bytes)
    except Exception as e:
        logger.error(f"Image decode failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not decode image file: {str(e)}",
        )

    parsed_boxes: Optional[List[List[float]]] = None
    if boxes:
        try:
            parsed_boxes = json.loads(boxes)
            if not isinstance(parsed_boxes, list):
                raise ValueError("boxes must be a JSON array of coordinates")
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid JSON format for boxes: {str(e)}",
            )

    manager = ModelManager.get_instance()
    base_url = str(request.base_url).rstrip("/")
    try:
        response = await manager.predict(
            image=image,
            prompt_text=prompt_text,
            boxes=parsed_boxes,
            threshold=threshold,
            mask_threshold=mask_threshold,
            mask_format=mask_format,
            refine_box_to_mask=refine_box_to_mask,
            compute_obb=compute_obb,
            box_nms_threshold=box_nms_threshold,
            filter_by_prompt_boxes=filter_by_prompt_boxes,
            model_override=model_override,
            base_url=base_url,
        )
        return response
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Segmentation inference failed: {str(e)}",
        )


@router.post(
    "/segment/json",
    response_model=SegmentResponse,
    summary="Segment Image via JSON Payload (Base64)",
    description="Send a base64 encoded image with text prompt and parameters in a JSON body.",
)
async def segment_image_json(
    request: Request,
    payload: SegmentJSONRequest,
) -> SegmentResponse:
    try:
        image = decode_base64_image(payload.image_base64)
    except Exception as e:
        logger.error(f"Base64 image decode failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not decode base64 image: {str(e)}",
        )

    manager = ModelManager.get_instance()
    base_url = str(request.base_url).rstrip("/")
    try:
        response = await manager.predict(
            image=image,
            prompt_text=payload.prompt_text,
            boxes=payload.boxes,
            threshold=payload.threshold or 0.35,
            mask_threshold=payload.mask_threshold or 0.50,
            mask_format=payload.mask_format,
            refine_box_to_mask=payload.refine_box_to_mask,
            compute_obb=payload.compute_obb,
            box_nms_threshold=payload.box_nms_threshold,
            filter_by_prompt_boxes=payload.filter_by_prompt_boxes,
            model_override=payload.model_override,
            base_url=base_url,
        )
        return response
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Segmentation inference failed: {str(e)}",
        )


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Service and GPU Health Check",
)
async def health_check() -> HealthResponse:
    manager = ModelManager.get_instance()
    gpu_stats = manager.get_gpu_stats()
    return HealthResponse(
        status="ok",
        device=str(manager.device),
        gpu_name=gpu_stats.get("gpu_name"),
        vram_allocated_mb=gpu_stats.get("vram_allocated_mb"),
        vram_reserved_mb=gpu_stats.get("vram_reserved_mb"),
        vram_total_mb=gpu_stats.get("vram_total_mb"),
    )


@router.get(
    "/info",
    response_model=InfoResponse,
    summary="Model Architecture and Optimization Information",
)
async def info() -> InfoResponse:
    manager = ModelManager.get_instance()
    active_id = manager.active_model_id or settings.MODEL_ID
    model, _ = manager.load_model(active_id)
    param_count = sum(p.numel() for p in model.parameters())

    compute_cap = torch.cuda.get_device_capability(0) if manager.device.type == "cuda" else (0, 0)
    has_tensor_cores = manager.device.type == "cuda" and compute_cap >= (7, 0) and manager.precision in ["fp16", "bf16"]

    return InfoResponse(
        model_id=active_id,
        precision=manager.precision,
        device=str(manager.device),
        num_parameters=param_count,
        supported_models=["facebook/sam3", "vil-uob/sam3-litetext-s0"],
        supported_formats=["rle", "polygon", "base64_png", "binary_mask"],
        cuda_optimizations={
            "tf32_enabled": settings.ENABLE_TF32 and compute_cap >= (8, 0),
            "cudnn_benchmark": settings.ENABLE_CUDNN_BENCHMARK and manager.device.type == "cuda",
            "tensor_cores_active": has_tensor_cores,
            "compute_capability": f"{compute_cap[0]}.{compute_cap[1]}" if manager.device.type == "cuda" else "N/A",
        },
    )


@router.post(
    "/warmup",
    summary="Trigger Pipeline Warmup",
)
async def trigger_warmup(
    model_id: Optional[str] = Query(None, description="Optional model to warm up.")
) -> dict:
    manager = ModelManager.get_instance()
    try:
        manager.warmup(model_id=model_id)
        return {"status": "success", "message": f"Warmup completed for {model_id or manager.active_model_id}"}
    except Exception as e:
        logger.exception("Warmup failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Warmup failed: {str(e)}"
        )
