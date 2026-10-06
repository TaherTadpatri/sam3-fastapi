# ==============================================================================
# SAM 3 High-Performance FastAPI Service Dockerfile
# Optimized for Universal Hardware Support:
# - NVIDIA GTX Series: Pascal (GTX 1060/1070/1080), Turing (GTX 1650/1660)
# - NVIDIA RTX Series: Turing (RTX 2060/2070/2080), Ampere (RTX 3050/3060/3070/3080/3090),
#                      Ada Lovelace (RTX 4060/4070/4080/4090)
# - Cloud & Data Center: T4, V100, A10, A100, H100, L4
# - Seamless CPU Fallback: Runs automatically on CPU if no GPU is attached
# ==============================================================================

FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

LABEL maintainer="SAM3 Team"
LABEL description="Production FastAPI Service for SAM 3 with universal GTX/RTX GPU and CPU support"

# Runtime Environment Configuration
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="6.1;7.0;7.5;8.0;8.6;8.9;9.0+PTX" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    DEVICE=auto \
    PRECISION=auto

# Install essential system dependencies (curl for healthchecks, libgl1/libglib for OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (PyTorch 2.6.0 is already pre-installed in the base image)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY app /app/app
COPY main.py /app/main.py

# Create directory for static mask files
RUN mkdir -p /app/static/masks

# Expose default HTTP service port
EXPOSE 8000

# Container healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/v1/health || exit 1

# Launch FastAPI service
CMD ["python", "main.py", "--host", "0.0.0.0", "--port", "8000"]
