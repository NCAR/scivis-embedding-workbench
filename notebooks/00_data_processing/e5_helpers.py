"""
Helper functions for e5_channels.py that must be importable by worker processes.

ProcessPoolExecutor requires callables to be picklable, which means they must
be defined at true module scope in an importable module — not inside marimo
cell bodies or notebook-level code.
"""
import numpy as np
from PIL import Image


def save_rgb_worker(args):
    """
    Resize three float32 [0,1] arrays to JPEG and save.

    args = (r_2d, g_2d, b_2d, out_path_str, width, height, quality)
    All arrays are numpy float32, shape (lat, lon), values in [0, 1].
    Returns the output path string on success.
    """
    r, g, b, out_path, width, height, quality = args
    rgb = np.clip(np.stack([r, g, b], axis=-1), 0, 1)
    img = Image.fromarray((rgb * 255).astype(np.uint8))
    img.resize((width, height), resample=Image.BILINEAR).save(
        out_path, "JPEG", quality=quality, optimize=True
    )
    return out_path
