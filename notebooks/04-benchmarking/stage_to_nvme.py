#!/usr/bin/env python3
"""
Stage a LanceDB experiment folder to Casper NVMe local scratch for I/O-intensive jobs.

Background
----------
Casper nodes expose fast NVMe storage at /local_scratch/pbs.$PBS_JOBID while a
job is running. Copying your LanceDB database there before launching the app
removes the Glade filesystem bottleneck for random-read-heavy workloads (vector
search, thumbnail fetching, etc.).

WARNING: /local_scratch/pbs.$PBS_JOBID is wiped when the job ends.
         Any outputs you generate must be moved back to Glade before the job
         finishes, or they will be lost.

How to use in a PBS job
-----------------------
Add the following lines to your PBS batch script, replacing the paths as needed:

    # Stage data to NVMe
    python /path/to/stage_to_nvme.py

    # Launch the app (use the printed "Experiments DB path" value)
    marimo run /path/to/app.py

    # (Optional) Move outputs back to Glade before job ends
    mv /local_scratch/pbs.$PBS_JOBID/output_data $SCRATCH/

How to test
-----------
Submit an interactive PBS job and run the script from there:

    qsub -I -l select=1:ncpus=4:mem=16GB -l walltime=01:00:00 -q casper -A <project>
    python stage_to_nvme.py

If PBS_JOBID is not set in the environment, the script queries qstat to find
your running job ID automatically. It errors out if no running job is found.

Configuration
-------------
Set SOURCE_DIR and GLADE_PROJECT_ROOT below.

SOURCE_DIR is the folder to stage (typically the lancedb directory).
GLADE_PROJECT_ROOT is the project root that SOURCE_DIR lives under.

The script mirrors SOURCE_DIR's path relative to GLADE_PROJECT_ROOT under the
NVMe root, so the destination becomes:
    /local_scratch/pbs.$PBS_JOBID/<SOURCE_DIR relative to GLADE_PROJECT_ROOT>

Preserving this relative path structure is required so that the dashboard's
source-path resolution can walk up the directory tree and locate the source
image DB using the relative path stored in each experiment's config.

The app's "Experiments DB path" field should be set to the staged lancedb
experiments subdirectory (printed at the end of the script).
"""

import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

# ── User config ───────────────────────────────────────────────────────────────
# Project root on Glade — SOURCE_DIR must be a subdirectory of this.
# The path of SOURCE_DIR relative to GLADE_PROJECT_ROOT is replicated under
# the NVMe root so the dashboard can resolve the source DB path correctly.
GLADE_PROJECT_ROOT = "/glade/work/ncheruku/research/sample_data"
SOURCE_DIR = "/glade/work/ncheruku/research/sample_data/data/lancedb/"
MAX_WORKERS = 8   # parallel copy threads; reduce if Glade I/O throttles
# ─────────────────────────────────────────────────────────────────────────────

def resolve_job_id() -> str:
    """Return PBS_JOBID from the environment, or look it up via qstat if not set."""
    job_id = os.environ.get("PBS_JOBID", "")
    if job_id:
        return job_id
    print("[stage] PBS_JOBID not in environment — querying qstat ...", flush=True)
    result = subprocess.run(
        ["qstat", "-u", os.environ["USER"], "-w"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print(f"[stage] ERROR: qstat failed: {result.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    # Parse lines like: 3384434.casper-pbs  ncheruku  ...  R  ...
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 10 and parts[9] == "R" and parts[1] == os.environ["USER"]:
            job_id = parts[0]
            print(f"[stage] Found running job: {job_id}", flush=True)
            return job_id
    print("[stage] ERROR: No running PBS job found for this user.", file=sys.stderr)
    sys.exit(1)


def get_nvme_root() -> Path:
    job_id = resolve_job_id()
    return Path(f"/local_scratch/pbs.{job_id}")


def check_space(src: Path, dst_root: Path) -> None:
    src_bytes = sum(f.stat().st_size for f in src.rglob("*") if f.is_file())
    src_gb = src_bytes / 1024**3
    result = subprocess.run(["df", "-BG", str(dst_root)], capture_output=True, text=True)
    print(f"[stage] Source size: {src_gb:.2f} GB", flush=True)
    print(f"[stage] NVMe available:\n{result.stdout}", flush=True)


def copy_with_progress(src: Path, dst: Path) -> None:
    all_files = sorted(f for f in src.rglob("*") if f.is_file())
    total_files = len(all_files)
    total_bytes = sum(f.stat().st_size for f in all_files)

    print(f"[stage] {total_files} files  |  {total_bytes / 1024**3:.2f} GB total", flush=True)

    done_count = 0

    def _copy_one(src_file: Path) -> int:
        dst_file = dst / src_file.relative_to(src)
        dst_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_file, dst_file)
        return src_file.stat().st_size

    with tqdm(
        total=total_bytes, unit="B", unit_scale=True,
        unit_divisor=1024, dynamic_ncols=True,
    ) as bar:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futures = {pool.submit(_copy_one, f): f for f in all_files}
            for fut in as_completed(futures):
                size = fut.result()
                done_count += 1
                bar.update(size)
                bar.set_postfix(files=f"{done_count}/{total_files}")

    print("[stage] Done.", flush=True)


def main() -> None:
    src = Path(SOURCE_DIR)
    project_root = Path(GLADE_PROJECT_ROOT)
    if not src.exists():
        print(f"[stage] ERROR: SOURCE_DIR does not exist: {src}", file=sys.stderr)
        sys.exit(1)
    try:
        rel = src.relative_to(project_root)
    except ValueError:
        print(
            f"[stage] ERROR: SOURCE_DIR ({src}) is not under GLADE_PROJECT_ROOT ({project_root})",
            file=sys.stderr,
        )
        sys.exit(1)

    nvme_root = get_nvme_root()
    # Mirror the Glade path structure under NVMe so resolve_source_path in the
    # dashboard can walk up the tree and find the source DB using the relative
    # path stored in each experiment's config.
    dst = nvme_root / rel

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
    print(f"  {dst}/experiments/era5")
    print("=" * 60)


if __name__ == "__main__":
    main()
