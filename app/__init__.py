"""SAM 3 FastAPI Service Package."""

__version__ = "1.0.0"

# Compatibility patch for PyTorch 2.6+ when running on CPU or unsupported devices
# In PyTorch 2.6, torch.accelerator.current_accelerator() raises a RuntimeError
# when no accelerator is attached, which can break transformers device resolution.
import torch

if hasattr(torch, "accelerator"):
    _orig_current_accelerator = getattr(torch.accelerator, "current_accelerator", None)
    if _orig_current_accelerator is not None:
        def _safe_current_accelerator():
            try:
                return _orig_current_accelerator()
            except Exception:
                return None
        torch.accelerator.current_accelerator = _safe_current_accelerator
