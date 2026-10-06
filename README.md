# SAM 3 High-Performance FastAPI Inference Service

A production-grade, low-latency **FastAPI** service for Meta's **Segment Anything Model 3 (SAM 3)** (`facebook/sam3` & `vil-uob/sam3-litetext-s0`) integrated directly with Hugging Face `transformers`.

Optimized with **Half-Precision (FP16) Quantization** for NVIDIA Tensor Cores (RTX 3050, Ampere architecture), cut-down payload serialization (COCO-compatible Run-Length Encoding), pre-warmed pipelines, and thread-safe GPU memory management.

---

## 🚀 Key Features

* **Half-Precision (FP16) Quantization**: Leverages hardware Tensor Cores for ~2.5x speedup and 50% lower VRAM footprint without accuracy degradation.
* **Dual Model Support**:
  * `facebook/sam3` (Official Foundation SAM 3, 840M parameters)
  * `vil-uob/sam3-litetext-s0` (Distilled SAM 3 with 88% smaller text encoder, 529M parameters)
* **Flexible Prompts**:
  * **Open-Vocabulary Concept / Text Prompts** (e.g., `"dog"`, `"red circle"`, `"yellow backpack"`)
  * **Bounding Box Prompts** (`[[x1, y1, x2, y2], ...]`)
* **Ultra-Fast Mask Serialization Formats**:
  * `rle` (Recommended): Compressed Run-Length Encoding (~3 KB per mask vs ~1 MB raw).
  * `polygon`: Exterior boundary contour coordinates for direct web canvas rendering.
  * `base64_png`: Base64-encoded 1-channel PNG mask for immediate `<img>` embedding.
  * `binary_mask`: Raw 2D nested array.
* **Zero Cold-Start Latency**: Pre-warming pipeline on server startup eliminates initial JIT/CUDA allocation delays.
* **Production Ready**: Interactive OpenAPI documentation at `/docs`, health checks with live GPU VRAM telemetry, and latency tracking headers (`X-Process-Time-Ms`).

---

## 📊 Benchmark Results (NVIDIA RTX 3050 4GB GPU)

| Metric / Format | Value | Note |
|---|---|---|
| **Steady-State Inference Latency** | **~1,650 ms** | Full 1008x1008 ViT feature map |
| **VRAM Footprint (FP16)** | **~1,724 MB** | Easily fits within 4GB VRAM |
| **Mask Payload (RLE)** | **3.20 KB** | **>300x smaller** than raw binary mask |
| **Mask Payload (Base64 PNG)** | **5.22 KB** | Ready for browser display |
| **Mask Payload (Raw Binary)** | **1,026 KB** | Uncompressed array |

---

## 🛠️ Project Structure

```
sam3/
├── app/
│   ├── __init__.py
│   ├── config.py           # Server & model configuration (env vars)
│   ├── schemas.py          # Pydantic request and response schemas
│   ├── utils.py            # High-speed image decoding, RLE, and contour extraction
│   ├── model_manager.py    # Singleton SAM3 engine (FP16, warm-up, thread-safe GPU lock)
│   └── api.py              # FastAPI endpoints (/segment, /health, /info, /warmup)
├── main.py                 # ASGI entrypoint with lifespan startup and CORS
├── test_client.py          # Comprehensive test suite
├── benchmark.py            # Latency and payload size benchmark suite
├── requirements.txt        # Verified dependencies
└── README.md
```

---

## ⚡ Quickstart

### 1. Run Server with SAM 3 LiteText (Ultra-Fast / 1.09 GB VRAM)

**Option A: Using the CLI Flag (Recommended)**
```bash
source .venv/bin/activate
python main.py --model litetext
```
*(You can also pass the full Hugging Face ID: `python main.py --model vil-uob/sam3-litetext-s0`)*

**Option B: Using an Environment Variable**
```bash
MODEL_ID="vil-uob/sam3-litetext-s0" python main.py
```

**Option C: Using a `.env` File**
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
python main.py
```

To run the original base model instead:
```bash
python main.py --model facebook/sam3
```

The server will start on `http://0.0.0.0:8000`. Open `http://localhost:8000/docs` in your browser for the Swagger UI.

### 2. Check System Health & GPU VRAM

```bash
curl -X GET "http://localhost:8000/v1/health"
```

**Response:**
```json
{
  "status": "ok",
  "device": "cuda",
  "gpu_name": "NVIDIA GeForce RTX 3050 Laptop GPU",
  "vram_allocated_mb": 1724.55,
  "vram_reserved_mb": 2154.00,
  "vram_total_mb": 3772.44
}
```

### 3. Segment an Image (Multipart Upload)

```bash
curl -X POST "http://localhost:8000/v1/segment" \
  -F "file=@photo.jpg" \
  -F "prompt_text=person" \
  -F "threshold=0.20" \
  -F "mask_format=rle"
```

### 4. Segment an Image (JSON with Base64)

```bash
curl -X POST "http://localhost:8000/v1/segment/json" \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "<BASE64_STRING>",
    "prompt_text": "red circle",
    "threshold": 0.20,
    "mask_format": "polygon"
  }'
```

### 5. Prompting with Bounding Boxes

```bash
curl -X POST "http://localhost:8000/v1/segment/json" \
  -H "Content-Type: application/json" \
  -d '{
    "image_base64": "<BASE64_STRING>",
    "boxes": [[100.0, 100.0, 300.0, 300.0]],
    "threshold": 0.20,
    "mask_format": "rle"
  }'
```

---

## 🧪 Running Tests & Benchmarks

To run the automated integration test suite:

```bash
python test_client.py
```

To run the latency and serialization benchmark:

```bash
python benchmark.py 3
```

---

## ⚙️ Configuration (.env / Environment Variables)

| Variable | Default | Description |
|---|---|---|
| `MODEL_ID` | `facebook/sam3` | Default Hugging Face repository |
| `PRECISION` | `fp16` | Precision: `fp16`, `bf16`, or `fp32` |
| `DEVICE` | `cuda` | Hardware target (`cuda` or `cpu`) |
| `DEFAULT_SCORE_THRESHOLD` | `0.20` | Minimum confidence score for detection |
| `DEFAULT_MASK_THRESHOLD` | `0.50` | Binary threshold for mask logits |
| `ENABLE_WARMUP` | `true` | Pre-warm GPU on startup |
| `PORT` | `8000` | HTTP service port |
| `HOST` | `0.0.0.0` | Bind address |
