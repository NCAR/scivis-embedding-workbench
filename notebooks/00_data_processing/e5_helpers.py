"""
Helper functions for the ERA5 channel processing pipeline.

These functions must live at true module scope so they are importable by
ProcessPoolExecutor worker processes — marimo compiles cells into isolated
files, making notebook-level definitions unpicklable.

GPU acceleration
----------------
torch (already installed) handles batch bilinear resize on CUDA.
CuPy is optional: if installed, it provides a zero-copy DLPack bridge so
arrays go numpy → cupy (GPU) → torch (GPU) without an extra PCIe transfer.

Install CuPy to match your CUDA version:
  CUDA 12.x:  pip install cupy-cuda12x
  CUDA 11.x:  pip install cupy-cuda11x
  Check CUDA: nvidia-smi | head -1
"""
import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# CuPy — optional, loaded once at import time
# ---------------------------------------------------------------------------
def _try_cupy():
    try:
        import cupy as cp
        cp.zeros(1)          # verify a GPU is actually reachable
        return cp
    except Exception:
        return None

_CP = _try_cupy()            # cupy module, or None if unavailable / no GPU


# ---------------------------------------------------------------------------
# Device selection
# ---------------------------------------------------------------------------
def pick_device():
    """Return a torch.device: 'cuda' if available, else 'cpu'."""
    import torch
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Batch GPU resize
# ---------------------------------------------------------------------------
def batch_resize_torch(R_np, G_np, B_np, width, height, device):
    """
    Stack three (B, lat, lon) float32 arrays in [0,1] and resize to
    (width, height) using torch bilinear interpolation on *device*.

    CuPy path (if available + CUDA): numpy → cupy → DLPack → torch
      — avoids a CPU round-trip; data stays on GPU throughout.
    Fallback path: numpy → torch tensor → .to(device).

    Returns: numpy uint8 array of shape (B, height, width, 3).
    """
    import torch
    import torch.nn.functional as F

    if _CP is not None and device.type == "cuda":
        cp = _CP
        # Stack on GPU via CuPy, then hand to torch with zero-copy DLPack
        batch_cp = cp.stack(
            [cp.asarray(R_np), cp.asarray(G_np), cp.asarray(B_np)], axis=1
        ).astype(cp.float32)                          # (B, 3, lat, lon) on GPU
        batch = torch.utils.dlpack.from_dlpack(batch_cp.toDlpack())
    else:
        # CPU path: stack in numpy, move to device
        batch = torch.from_numpy(
            np.stack([R_np, G_np, B_np], axis=1).astype(np.float32)
        ).to(device)                                  # (B, 3, lat, lon)

    # Single GPU kernel: (B, 3, lat, lon) → (B, 3, height, width)
    resized = F.interpolate(batch, size=(height, width),
                            mode="bilinear", align_corners=False)

    # Clamp, convert to uint8, pull to CPU: (B, 3, H, W) → (B, H, W, 3)
    resized = (resized.clamp(0, 1) * 255).byte().cpu().numpy()
    return resized.transpose(0, 2, 3, 1)


# ---------------------------------------------------------------------------
# Worker: JPEG encode only (resize already done by batch_resize_torch)
# ---------------------------------------------------------------------------
def save_jpeg_worker(args):
    """
    Encode a pre-resized uint8 HWC array as JPEG and save.

    args = (rgb_hwc_uint8, out_path_str, quality)
    Returns the output path string on success.
    """
    from PIL import Image as _Image
    rgb_hwc, out_path, quality = args
    _Image.fromarray(rgb_hwc).save(out_path, "JPEG", quality=quality, optimize=True)
    return out_path


# ---------------------------------------------------------------------------
# Legacy worker kept for backwards compatibility
# ---------------------------------------------------------------------------
def save_rgb_worker(args):
    """
    Resize three float32 [0,1] arrays to JPEG and save (CPU-only, legacy).

    args = (r_2d, g_2d, b_2d, out_path_str, width, height, quality)
    Prefer save_jpeg_worker + batch_resize_torch for GPU-accelerated runs.
    """
    r, g, b, out_path, width, height, quality = args
    rgb = np.clip(np.stack([r, g, b], axis=-1), 0, 1)
    img = Image.fromarray((rgb * 255).astype(np.uint8))
    img.resize((width, height), resample=Image.BILINEAR).save(
        out_path, "JPEG", quality=quality, optimize=True
    )
    return out_path
