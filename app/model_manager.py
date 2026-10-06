"""High-Performance Singleton Model Manager for SAM 3 with Memory-Safe Quantization."""

import gc
import asyncio
import time
import logging
from typing import Dict, Any, List, Optional, Tuple
import torch
import numpy as np
from PIL import Image
from transformers import Sam3Processor, Sam3Model, Sam3LiteTextModel

from app.config import settings
from app.schemas import DetectionResult, SegmentResponse
from app.utils import format_mask

logger = logging.getLogger("sam3.model_manager")


class ModelManager:
    """Singleton manager controlling SAM3 model lifecycle, quantization, and GPU execution."""
    _instance: Optional["ModelManager"] = None
    
    def __init__(self):
        self.device = torch.device(settings.DEVICE)
        self.torch_dtype = self._get_torch_dtype(settings.PRECISION)
        self.active_model: Optional[Any] = None
        self.active_processor: Optional[Sam3Processor] = None
        self.active_model_id: Optional[str] = None
        self.lock = asyncio.Lock()
        
        # Configure PyTorch CUDA runtime acceleration
        if self.device.type == "cuda":
            if settings.ENABLE_TF32:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            if settings.ENABLE_CUDNN_BENCHMARK:
                torch.backends.cudnn.benchmark = True
            logger.info("Configured CUDA optimizations: TF32=%s, cuDNN Benchmark=%s", 
                        settings.ENABLE_TF32, settings.ENABLE_CUDNN_BENCHMARK)

    @classmethod
    def get_instance(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @staticmethod
    def normalize_model_id(model_id: str) -> str:
        """Normalize aliases and common typos to full Hugging Face model IDs."""
        cleaned = (model_id or "").strip()
        cleaned_lower = cleaned.lower()
        if any(k in cleaned_lower for k in ["litetext", "lite-text", "sam3-lite"]):
            return "vil-uob/sam3-litetext-s0"
        if any(k in cleaned_lower for k in ["sam3", "facebook"]):
            return "facebook/sam3"
        return cleaned

    def _get_torch_dtype(self, precision_name: str) -> torch.dtype:
        if precision_name == "fp16":
            return torch.float16
        elif precision_name == "bf16":
            return torch.bfloat16
        return torch.float32

    def load_model(self, model_id: str) -> Tuple[Any, Sam3Processor]:
        """Load target model. Unloads previously active model to strictly conserve RAM and VRAM."""
        model_id = self.normalize_model_id(model_id)
        if self.active_model_id == model_id and self.active_model is not None and self.active_processor is not None:
            return self.active_model, self.active_processor
            
        # Unload prior model if different to guarantee zero RAM/VRAM leak
        if self.active_model is not None:
            logger.info(f"Unloading previous model '{self.active_model_id}' to preserve memory...")
            del self.active_model
            del self.active_processor
            self.active_model = None
            self.active_processor = None
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()

        logger.info(f"Loading processor for '{model_id}'...")
        try:
            # Try loading directly from local hub cache for instantaneous startup
            processor = Sam3Processor.from_pretrained(model_id, local_files_only=True)
        except Exception:
            processor = Sam3Processor.from_pretrained(model_id, local_files_only=settings.LOCAL_FILES_ONLY)
        
        logger.info(f"Loading model weights for '{model_id}' with precision={settings.PRECISION} on device={self.device}...")
        start_time = time.time()
        
        model_cls = Sam3LiteTextModel if "litetext" in model_id.lower() else Sam3Model
        try:
            # Try loading from local hub cache first
            model = model_cls.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                local_files_only=True
            )
        except Exception:
            model = model_cls.from_pretrained(
                model_id,
                torch_dtype=self.torch_dtype,
                local_files_only=settings.LOCAL_FILES_ONLY
            )
            
        model = model.to(self.device)
        model.eval()
        
        load_time = time.time() - start_time
        num_params = sum(p.numel() for p in model.parameters())
        logger.info(f"Successfully loaded '{model_id}' ({num_params:,} parameters) in {load_time:.2f}s")
        
        self.active_model = model
        self.active_processor = processor
        self.active_model_id = model_id
        return model, processor

    def warmup(self, model_id: Optional[str] = None):
        """Warm up CUDA kernels and JIT compilation paths with a dummy inference pass."""
        target_model_id = model_id or settings.MODEL_ID
        logger.info(f"Warming up pipeline for '{target_model_id}'...")
        model, processor = self.load_model(target_model_id)
        
        dummy_image = Image.new("RGB", (512, 512), color=(128, 128, 128))
        inputs = processor(images=dummy_image, text="warmup", return_tensors="pt")
        inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        for k, v in inputs.items():
            if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
                inputs[k] = v.to(self.torch_dtype)
                
        with torch.inference_mode():
            outputs = model(**inputs)
            _ = processor.post_process_instance_segmentation(
                outputs, 
                threshold=0.5, 
                mask_threshold=0.5, 
                target_sizes=inputs["original_sizes"].tolist()
            )
            
        if self.device.type == "cuda":
            torch.cuda.synchronize()
        logger.info(f"Pipeline warm-up complete for '{target_model_id}'.")

    @staticmethod
    def compute_box_iou(boxA: List[float], boxB: List[float]) -> float:
        """Compute Intersection-over-Union (IoU) between two bounding boxes [x1, y1, x2, y2]."""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        inter = max(0.0, xB - xA) * max(0.0, yB - yA)
        areaA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
        areaB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
        union = areaA + areaB - inter
        return inter / union if union > 0 else 0.0

    async def predict(
        self,
        image: Image.Image,
        prompt_text: Optional[str] = None,
        boxes: Optional[List[List[float]]] = None,
        threshold: float = 0.35,
        mask_threshold: float = 0.50,
        mask_format: str = "rle",
        refine_box_to_mask: bool = True,
        box_nms_threshold: Optional[float] = 0.50,
        filter_by_prompt_boxes: bool = True,
        model_override: Optional[str] = None,
    ) -> SegmentResponse:
        """Run high-accuracy SAM3 inference with box refinement and NMS."""
        async with self.lock:
            total_start = time.perf_counter()
            target_model_id = model_override or self.active_model_id or settings.MODEL_ID
            model, processor = self.load_model(target_model_id)
            
            orig_w, orig_h = image.size
            
            # Prepare kwargs for Sam3Processor
            proc_kwargs: Dict[str, Any] = {
                "images": image,
                "return_tensors": "pt",
            }
            
            # 1. Un-normalize input boxes if provided in normalized [0, 1] coordinates
            unnorm_prompt_boxes: List[List[float]] = []
            if boxes and len(boxes) > 0:
                for b in boxes:
                    if len(b) == 4:
                        if all(0.0 <= coord <= 1.0 for coord in b) and (orig_w > 1 and orig_h > 1):
                            # Un-normalize: [x1 * W, y1 * H, x2 * W, y2 * H]
                            unnorm_prompt_boxes.append([b[0] * orig_w, b[1] * orig_h, b[2] * orig_w, b[3] * orig_h])
                        else:
                            unnorm_prompt_boxes.append(b)
                proc_kwargs["input_boxes"] = [unnorm_prompt_boxes]

            # 2. Configure text prompt
            if prompt_text and prompt_text.strip():
                proc_kwargs["text"] = prompt_text.strip()
            elif not unnorm_prompt_boxes:
                # Only use fallback text prompt when NO boxes are supplied
                proc_kwargs["text"] = "object"

            inputs = processor(**proc_kwargs)
            inputs = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Cast floating inputs to match quantized / half precision
            for k, v in inputs.items():
                if isinstance(v, torch.Tensor) and v.dtype == torch.float32:
                    inputs[k] = v.to(self.torch_dtype)

            # Timed inference pass
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            infer_start = time.perf_counter()
            
            with torch.inference_mode():
                outputs = model(**inputs)
                
            if self.device.type == "cuda":
                torch.cuda.synchronize()
            infer_time_ms = (time.perf_counter() - infer_start) * 1000

            # 3. Post-process detections and masks
            results = processor.post_process_instance_segmentation(
                outputs,
                threshold=threshold,
                mask_threshold=mask_threshold,
                target_sizes=inputs["original_sizes"].tolist(),
            )
            
            # Extract presence score from SAM3's presence head
            presence_score: Optional[float] = None
            if hasattr(outputs, "presence_logits") and outputs.presence_logits is not None:
                presence_val = torch.sigmoid(outputs.presence_logits).squeeze().item()
                presence_score = round(float(presence_val), 4)

            # 4. Filter, Refine Bounding Boxes, and Format Detections
            detections: List[DetectionResult] = []
            if len(results) > 0 and len(results[0]["scores"]) > 0:
                res = results[0]
                scores_t = res["scores"].cpu()
                boxes_t = res["boxes"].cpu()
                masks_t = res["masks"].cpu()

                # A. Apply Non-Maximum Suppression (NMS) to eliminate duplicate overlapping boxes
                if box_nms_threshold is not None and box_nms_threshold > 0 and len(scores_t) > 1:
                    import torchvision
                    keep = torchvision.ops.nms(boxes_t.float(), scores_t.float(), iou_threshold=box_nms_threshold)
                    scores_t = scores_t[keep]
                    boxes_t = boxes_t[keep]
                    masks_t = masks_t[keep]

                scores = scores_t.numpy()
                boxes_out = boxes_t.numpy()
                masks = masks_t.numpy()

                raw_detections: List[Dict[str, Any]] = []
                for idx in range(len(scores)):
                    score_val = float(scores[idx])
                    box_coords = [round(float(c), 2) for c in boxes_out[idx].tolist()]
                    binary_mask = masks[idx].astype(np.uint8)
                    
                    # Compute pixel-exact tight bounding box from the predicted mask
                    y_indices, x_indices = np.where(binary_mask)
                    if len(x_indices) > 0:
                        tight_box = [
                            round(float(np.min(x_indices)), 2),
                            round(float(np.min(y_indices)), 2),
                            round(float(np.max(x_indices)), 2),
                            round(float(np.max(y_indices)), 2),
                        ]
                    else:
                        tight_box = box_coords

                    # If box refinement is enabled, use the tight mask boundary
                    final_box = tight_box if refine_box_to_mask else box_coords
                    mask_dict = format_mask(binary_mask, mask_format)

                    raw_detections.append({
                        "id": idx,
                        "score": round(score_val, 4),
                        "box": final_box,
                        "tight_box": tight_box,
                        "area": mask_dict.get("area"),
                        "mask_rle": mask_dict.get("mask_rle"),
                        "mask_polygons": mask_dict.get("mask_polygons"),
                        "mask_base64": mask_dict.get("mask_base64"),
                        "mask_binary": mask_dict.get("mask_binary"),
                        "binary_mask": binary_mask,
                    })

                # B. If input prompt boxes were provided, match detections to each prompt box
                if unnorm_prompt_boxes and len(unnorm_prompt_boxes) > 0 and filter_by_prompt_boxes:
                    matched_results: List[DetectionResult] = []
                    used_indices = set()
                    for p_idx, p_box in enumerate(unnorm_prompt_boxes):
                        best_match = None
                        best_iou = -1.0
                        best_idx = -1
                        for d_idx, d in enumerate(raw_detections):
                            if d_idx in used_indices:
                                continue
                            iou = self.compute_box_iou(d["box"], p_box)
                            if iou > best_iou:
                                best_iou = iou
                                best_match = d
                                best_idx = d_idx

                        if best_match is not None and (best_iou > 0.15 or len(raw_detections) == 1):
                            used_indices.add(best_idx)
                            matched_results.append(DetectionResult(
                                id=len(matched_results),
                                score=best_match["score"],
                                box=best_match["box"],
                                tight_box=best_match["tight_box"],
                                area=best_match["area"],
                                iou_with_prompt=round(best_iou, 4),
                                matched_prompt_index=p_idx,
                                mask_rle=best_match["mask_rle"],
                                mask_polygons=best_match["mask_polygons"],
                                mask_base64=best_match["mask_base64"],
                                mask_binary=best_match["mask_binary"],
                            ))
                    detections = matched_results
                else:
                    # Return all high-confidence detections
                    for idx, d in enumerate(raw_detections):
                        detections.append(DetectionResult(
                            id=idx,
                            score=d["score"],
                            box=d["box"],
                            tight_box=d["tight_box"],
                            area=d["area"],
                            mask_rle=d["mask_rle"],
                            mask_polygons=d["mask_polygons"],
                            mask_base64=d["mask_base64"],
                            mask_binary=d["mask_binary"],
                        ))

            total_time_ms = (time.perf_counter() - total_start) * 1000

            return SegmentResponse(
                success=True,
                model_id=target_model_id,
                precision=settings.PRECISION,
                image_size=[orig_h, orig_w],
                num_detections=len(detections),
                presence_score=presence_score,
                detections=detections,
                inference_time_ms=round(infer_time_ms, 2),
                total_time_ms=round(total_time_ms, 2),
            )

    def get_gpu_stats(self) -> Dict[str, Any]:
        """Return real-time VRAM allocation and device properties."""
        if self.device.type != "cuda":
            return {"device": "cpu"}
            
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        
        return {
            "device": "cuda",
            "gpu_name": torch.cuda.get_device_name(0),
            "vram_allocated_mb": round(allocated, 2),
            "vram_reserved_mb": round(reserved, 2),
            "vram_total_mb": round(total, 2),
        }
