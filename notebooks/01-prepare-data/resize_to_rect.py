#!/usr/bin/env python3
"""
Resize a folder of square images to a rectangular aspect ratio.

Default target is 896×256 (W×H), matching the true geographic aspect ratio of
the ERA5 dataset (lon 260–330 = 70°, lat 15–35 = 20° → ratio 3.5:1).
Both dimensions are multiples of DINO patch size 16.

Usage:
    uv run python notebooks/01-prepare-data/resize_to_rect.py
    uv run python notebooks/01-prepare-data/resize_to_rect.py --src /path/to/src --dst /path/to/dst
"""

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

# Reuse existing image helpers
sys.path.insert(0, str(Path(__file__).parent))
from helpers.image_utils import image_to_jpeg_bytes, open_rgb_image, resize_image

_DEFAULT_SRC = Path("/Users/ncheruku/Documents/Work/sample_data/data/processed_rgb")
_DEFAULT_DST = Path("/Users/ncheruku/Documents/Work/sample_data/data/processed_rgb_rect")

IMAGE_EXTS = {".jpeg", ".jpg", ".png"}


def _process_one(args):
    src_path, dst_path, width, height, quality = args
    try:
        img = open_rgb_image(src_path)
        img = resize_image(img, width=width, height=height)
        data = image_to_jpeg_bytes(img, quality=quality)
        dst_path.write_bytes(data)
        return True, src_path.name, None
    except Exception as e:
        return False, src_path.name, str(e)


def main():
    ap = argparse.ArgumentParser(description="Resize images to a rectangular aspect ratio.")
    ap.add_argument("--src", type=Path, default=_DEFAULT_SRC, help="Source image directory")
    ap.add_argument("--dst", type=Path, default=_DEFAULT_DST, help="Destination directory")
    ap.add_argument("--width", type=int, default=896, help="Output width in pixels")
    ap.add_argument("--height", type=int, default=256, help="Output height in pixels")
    ap.add_argument("--quality", type=int, default=95, help="JPEG quality (1-95)")
    ap.add_argument("--workers", type=int, default=4, help="Parallel worker processes")
    args = ap.parse_args()

    src = args.src.resolve()
    dst = args.dst.resolve()

    if not src.is_dir():
        print(f"ERROR: source directory does not exist: {src}", file=sys.stderr)
        sys.exit(1)

    dst.mkdir(parents=True, exist_ok=True)

    image_files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if not image_files:
        print(f"No images found in {src}", file=sys.stderr)
        sys.exit(1)

    print(f"Source:  {src}  ({len(image_files)} images)")
    print(f"Dest:    {dst}")
    print(f"Target:  {args.width}×{args.height}  quality={args.quality}  workers={args.workers}")

    tasks = [
        (p, dst / p.name, args.width, args.height, args.quality)
        for p in image_files
    ]

    processed = 0
    errors = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_process_one, t): t[0].name for t in tasks}
        with tqdm(total=len(tasks), unit="img") as pbar:
            for fut in as_completed(futures):
                ok, name, err = fut.result()
                if ok:
                    processed += 1
                else:
                    errors.append((name, err))
                pbar.update(1)

    print(f"\nDone.  processed={processed}  errors={len(errors)}")
    if errors:
        for name, err in errors[:10]:
            print(f"  ERROR {name}: {err}")


if __name__ == "__main__":
    main()
