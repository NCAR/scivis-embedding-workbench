#!/usr/bin/env python3
"""
Stage a LanceDB experiment folder to Casper NVMe local scratch for I/O-intensive jobs.

Usage (inside a PBS job):
    python stage_to_nvme.py

Usage (standalone, e.g. for testing):
    python stage_to_nvme.py

Set SOURCE_DIR below to the folder you want to copy.
The script copies to /local_scratch/pbs.$PBS_JOBID (or /tmp/scivis_staging if
PBS_JOBID is not set), prints per-folder progress, and prints the destination
path to paste into the app's "Experiments DB path" field.

NOTE: /local_scratch/pbs.$PBS_JOBID is deleted when the job ends.
      Move any outputs you want to keep before the job finishes.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

# ── User config ───────────────────────────────────────────────────────────────
# Set this to the folder you want to stage (e.g. the LanceDB experiments dir)
SOURCE_DIR = "/glade/work/ncheruku/research/sample_data/data/lancedb/experiments/era5"
# ─────────────────────────────────────────────────────────────────────────────

def get_nvme_root() -> Path:
    job_id = os.environ.get("PBS_JOBID", "")
    if job_id:
        return Path(f"/local_scratch/pbs.{job_id}")
    fallback = Path("/tmp/scivis_staging")
    print(f"[stage] PBS_JOBID not set — using fallback: {fallback}", flush=True)
    return fallback


def check_space(src: Path, dst_root: Path) -> None:
    src_bytes = sum(f.stat().st_size for f in src.rglob("*") if f.is_file())
    src_gb = src_bytes / 1024**3
    result = subprocess.run(["df", "-BG", str(dst_root)], capture_output=True, text=True)
    print(f"[stage] Source size: {src_gb:.2f} GB", flush=True)
    print(f"[stage] NVMe available:\n{result.stdout}", flush=True)


def copy_with_progress(src: Path, dst: Path) -> None:
    # Group files by their immediate parent folder (one level below src)
    folders: dict[Path, list[Path]] = {}
    for f in sorted(src.rglob("*")):
        if not f.is_file():
            continue
        top = f.relative_to(src).parts[0]
        folders.setdefault(Path(top), []).append(f)

    total_bytes = sum(f.stat().st_size for f in src.rglob("*") if f.is_file())
    n_folders = len(folders)
    copied_bytes = 0

    print(f"[stage] {n_folders} folders  |  {total_bytes / 1024**3:.2f} GB total", flush=True)

    for i, (folder, files) in enumerate(folders.items(), 1):
        folder_bytes = sum(f.stat().st_size for f in files)
        for src_file in files:
            dst_file = dst / src_file.relative_to(src)
            dst_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src_file, dst_file)
        copied_bytes += folder_bytes
        pct = copied_bytes / total_bytes * 100
        print(
            f"  [{i}/{n_folders}] {folder}/  "
            f"{folder_bytes / 1024**2:.1f} MB  —  "
            f"{copied_bytes / 1024**3:.2f} / {total_bytes / 1024**3:.2f} GB  ({pct:.0f}%)",
            flush=True,
        )

    print("[stage] Done.", flush=True)


def main() -> None:
    src = Path(SOURCE_DIR)
    if not src.exists():
        print(f"[stage] ERROR: SOURCE_DIR does not exist: {src}", file=sys.stderr)
        sys.exit(1)

    nvme_root = get_nvme_root()
    dst = nvme_root / src.name

    print(f"[stage] Source : {src}", flush=True)
    print(f"[stage] Dest   : {dst}", flush=True)

    check_space(src, nvme_root)

    if dst.exists():
        print(f"[stage] Destination already exists, skipping copy.", flush=True)
    else:
        copy_with_progress(src, dst)

    print()
    print("=" * 60)
    print("  Paste this path into the app's 'Experiments DB path':")
    print(f"  {dst.parent}")
    print("=" * 60)


if __name__ == "__main__":
    main()
