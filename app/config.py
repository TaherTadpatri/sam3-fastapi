"""Configuration management for SAM3 FastAPI service."""

import os
from typing import Literal
import torch
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Model Configuration
    # Options: "facebook/sam3", "vil-uob/sam3-litetext-s0"
    MODEL_ID: str = "facebook/sam3"
    
    # Precision / Quantization
    # "fp16" provides ~2.5x speedup and 50% memory reduction on NVIDIA Tensor Cores (RTX 3050)
    # with negligible loss in mAP accuracy.
    PRECISION: Literal["fp16", "bf16", "fp32"] = "fp16"
    
    # Device
    DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Performance & CUDA settings
    ENABLE_TF32: bool = True
    ENABLE_CUDNN_BENCHMARK: bool = True
    ENABLE_WARMUP: bool = True
    
    # Inference defaults
    DEFAULT_SCORE_THRESHOLD: float = 0.35
    DEFAULT_MASK_THRESHOLD: float = 0.50
    
    # Server configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "info"
    
    # Cache settings
    LOCAL_FILES_ONLY: bool = False

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


settings = Settings()
