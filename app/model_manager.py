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

    async def predict(
        self,
        image: Image.Image,
        prompt_text: Optional[str] = None,
        boxes: Optional[List[List[float]]] = None,
        threshold: float = 0.20,
        mask_threshold: float = 0.50,
        mask_format: str = "rle",
        model_override: Optional[str] = None,
    ) -> SegmentResponse:
        """Run SAM3 inference with async lock for GPU concurrency safety."""
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
            if prompt_text:
                proc_kwargs["text"] = prompt_text
            else:
                proc_kwargs["text"] = "object"
                
            if boxes and len(boxes) > 0:
                # Format: [batch_size, num_boxes, 4]
                proc_kwargs["input_boxes"] = [boxes]

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

            # Post-process detections and masks
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

            # Format detections
            detections: List[DetectionResult] = []
            if len(results) > 0:
                res = results[0]
                scores = res["scores"].cpu().numpy()
                boxes_out = res["boxes"].cpu().numpy()
                masks = res["masks"].cpu().numpy()

                for idx in range(len(scores)):
                    score_val = float(scores[idx])
                    box_coords = [round(float(c), 2) for c in boxes_out[idx].tolist()]
                    binary_mask = masks[idx].astype(np.uint8)
                    
                    mask_dict = format_mask(binary_mask, mask_format)
                    
                    det = DetectionResult(
                        id=idx,
                        score=round(score_val, 4),
                        box=box_coords,
                        area=mask_dict.get("area"),
                        mask_rle=mask_dict.get("mask_rle"),
                        mask_polygons=mask_dict.get("mask_polygons"),
                        mask_base64=mask_dict.get("mask_base64"),
                        mask_binary=mask_dict.get("mask_binary"),
                    )
                    detections.append(det)

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
