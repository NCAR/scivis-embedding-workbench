#!/usr/bin/env python3
"""
01_generate_ground_truth.py — GPU-accelerated exact nearest-neighbour ground truth.

Run this once on a node with an NVIDIA A100 GPU before running the latency
benchmark.  It produces two files consumed by 02_benchmark_ann_latency.py:

    results/universal_queries.npy   — shape (1_000, 768) float32
    results/ground_truth.pkl        — dict[exp_name → dict[query_idx → set[patch_id]]]

Methodology
-----------
1. Draw 1,000 universal query vectors from the 1h patch_embeddings table,
   restricted to images that exist in ALL 5 experiments (the 24h-aligned
   midnight images).  Saved with seed=42 for reproducibility.

2. For each experiment, load the full patch table (patch_id + embedding
   columns) into CPU RAM.  Normalize embeddings on the CPU, then stream
   them to the A100 in chunks for exact cosine similarity via matmul.

3. torch.topk(k=10) accumulates the global top-10 across all chunks using
   a rolling merge.  Resulting row indices are mapped back to patch_id
   strings (the unique per-row identifier in LanceDB — patch_index alone
   is 0-895 within each image and is NOT globally unique).

4. VRAM is freed between experiments via del + torch.cuda.empty_cache().

Dataset (3 years hourly ERA5, 2016-01-01 → 2018-12-31, 896 patches/image):

    Experiment   Images    Patches    ~CPU RAM
    24h          ~1,096    ~0.98M       ~3 GB
    12h          ~2,192    ~1.96M       ~6 GB
    6h           ~4,384    ~3.93M      ~12 GB
    3h           ~8,768    ~7.86M      ~24 GB
    1h          ~26,304   ~23.60M      ~73 GB   ← needs 256 GB node

Usage
-----
    uv run python notebooks/04-benchmarking/01_generate_ground_truth.py
"""

from __future__ import annotations

import math
import pickle
import time
from pathlib import Path

import lancedb
import numpy as np
import pyarrow.compute as pc
import torch
import torch.nn.functional as F

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

# NCAR Casper
PROJECT_ROOT = Path("/glade/work/ncheruku/research/sample_data")
# Local Mac (uncomment to switch):
# PROJECT_ROOT = Path("/Users/ncheruku/Documents/Work/sample_data")

DB_URI = PROJECT_ROOT / "data" / "lancedb" / "experiments" / "era5"

# Experiments — same order as the benchmark script.
EXPERIMENTS = [
    ("24h", "dinov3_rect_24h"),
    ("12h", "dinov3_rect_12h"),
    ("6h",  "dinov3_rect_6h"),
    ("3h",  "dinov3_rect_3h"),
    ("1h",  "dinov3_rect_1h"),
]

N_QUERY_VECTORS = 1_000    # universal query set size
RANDOM_SEED     = 42
TOP_K           = 10

# GPU chunk size: number of DB rows processed per matmul call.
# At 2 M rows × 768 dims × float32: ~6 GB VRAM for the chunk tensor,
# ~8 GB for the 1000 × 2M similarity matrix → ~14 GB total.  Safe for A100.
CHUNK_SIZE = 2_000_000

RESULTS_DIR    = Path(__file__).parent / "results"
QUERIES_NPY    = RESULTS_DIR / "universal_queries.npy"
GROUND_TRUTH_PKL = RESULTS_DIR / "ground_truth.pkl"


# ── HELPERS ───────────────────────────────────────────────────────────────────

def sample_intersection_vectors(
    base_patch_tbl: lancedb.table.Table,
    smallest_img_tbl: lancedb.table.Table,
    n: int,
    seed: int,
) -> np.ndarray:
    """Draw n patch embeddings guaranteed to exist in every experiment.

    The 24h experiment contains only midnight (00:00 UTC) images.  Every
    other experiment is a strict superset, so 24h image_ids form the safe
    intersection.  Vectors are fetched from the 1h base table.

    Parameters
    ----------
    base_patch_tbl   : patch_embeddings table of the 1h experiment
    smallest_img_tbl : image_embeddings table of the 24h experiment
    n                : number of vectors to return
    seed             : RNG seed for reproducibility

    Returns
    -------
    np.ndarray of shape (n, 768), dtype float32
    """
    anchor_ids = (
        smallest_img_tbl.to_lance()
        .scanner(columns=["image_id"])
        .to_table()
        .column("image_id")
        .to_pylist()
    )
    print(f"  Intersection image_ids : {len(anchor_ids):,}  "
          "(images present in all 5 experiments)")

    anchor_patches = (
        base_patch_tbl.to_lance()
        .scanner(
            columns=["embedding"],
            filter=pc.field("image_id").isin(anchor_ids),
        )
        .to_table()
    )
    embs = np.array(
        anchor_patches.column("embedding").to_pylist(), dtype=np.float32
    )
    print(f"  Patch pool             : {len(embs):,}")

    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(embs), size=min(n, len(embs)), replace=False)
    vecs = embs[chosen]
    print(f"  Query vectors sampled  : {len(vecs):,}  shape={vecs.shape}")
    return vecs


def load_patch_table(
    patch_tbl: lancedb.table.Table,
) -> tuple[np.ndarray, list[str]]:
    """Load embedding and patch_id columns from a LanceDB patch table.

    Returns
    -------
    embs      : np.ndarray of shape (N, 768), dtype float32
    patch_ids : list of N patch_id strings (globally unique per row)

    Note
    ----
    patch_index (0-895) is the within-image patch position and is NOT
    globally unique — patch 42 from image A ≠ patch 42 from image B.
    patch_id is the correct unique identifier for recall computation.
    """
    n_rows = patch_tbl.count_rows()
    mem_gb = n_rows * 768 * 4 / 1024**3
    print(f"  Loading {n_rows:,} rows  (~{mem_gb:.1f} GB RAM) …")

    t0 = time.perf_counter()
    arrow_tbl = (
        patch_tbl.to_lance()
        .scanner(columns=["patch_id", "embedding"])
        .to_table()
    )
    patch_ids = arrow_tbl.column("patch_id").to_pylist()
    embs = np.array(
        arrow_tbl.column("embedding").to_pylist(), dtype=np.float32
    )
    del arrow_tbl
    print(f"  Table loaded in {time.perf_counter() - t0:.1f}s")
    return embs, patch_ids


def compute_topk_cosine_gpu(
    db_embs_norm: np.ndarray,
    query_norm: torch.Tensor,
    k: int,
    chunk_size: int,
    device: torch.device,
) -> np.ndarray:
    """Exact top-k cosine similarity via chunked GPU matmul.

    Parameters
    ----------
    db_embs_norm : CPU numpy array (N, D), already L2-normalised float32
    query_norm   : GPU tensor (Q, D), already L2-normalised float32
    k            : number of nearest neighbours to return
    chunk_size   : number of DB rows per GPU matmul call
    device       : CUDA device

    Returns
    -------
    np.ndarray of shape (Q, k) — row indices into db_embs_norm
    """
    n_db   = db_embs_norm.shape[0]
    n_q    = query_norm.shape[0]
    n_chunks = math.ceil(n_db / chunk_size)

    # Initialise running best with sentinel similarity of -2.0
    # (cosine similarity is bounded in [-1, 1])
    best_sims = torch.full((n_q, k), fill_value=-2.0, device=device)
    best_idxs = torch.zeros((n_q, k), dtype=torch.long, device=device)

    for c in range(n_chunks):
        c_start = c * chunk_size
        c_end   = min(c_start + chunk_size, n_db)

        # Stream chunk to GPU
        chunk = torch.from_numpy(db_embs_norm[c_start:c_end]).to(device)
        sims  = torch.matmul(query_norm, chunk.T)       # (Q, chunk_size)

        k_eff = min(k, c_end - c_start)
        c_top_vals, c_top_local = torch.topk(sims, k=k_eff, dim=1)
        c_top_global = c_top_local + c_start

        # Merge chunk candidates with running best
        cat_vals = torch.cat([best_sims, c_top_vals],   dim=1)
        cat_idxs = torch.cat([best_idxs, c_top_global], dim=1)
        merged   = torch.topk(cat_vals, k=k, dim=1)
        best_sims = merged.values
        best_idxs = cat_idxs.gather(1, merged.indices)

        del chunk, sims, c_top_vals, c_top_local, c_top_global
        del cat_vals, cat_idxs, merged
        torch.cuda.empty_cache()

        if (c + 1) % 5 == 0 or c == n_chunks - 1:
            pct = 100 * (c + 1) / n_chunks
            print(f"    chunk {c+1:3d}/{n_chunks}  "
                  f"({c_end:,}/{n_db:,} rows, {pct:.0f}%)")

    return best_idxs.cpu().numpy()   # (Q, k) global row indices into db


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── GPU check ─────────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        raise RuntimeError(
            "No CUDA GPU detected.  This script requires an NVIDIA GPU (A100).\n"
            "Run on a GPU node: e.g.  qsub -l gpu=1  or  srun --gres=gpu:1 …"
        )
    device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
    vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"GPU : {gpu_name}  ({vram_gb:.0f} GB VRAM)")
    print()

    # ── Step 1: universal query vectors ───────────────────────────────────────
    print("=" * 64)
    print("Step 1 — Sampling universal query vectors …")
    _base_db    = lancedb.connect(str(DB_URI / "dinov3_rect_1h"))
    _base_patch = _base_db.open_table("patch_embeddings")
    _24h_db     = lancedb.connect(str(DB_URI / "dinov3_rect_24h"))
    _24h_img    = _24h_db.open_table("image_embeddings")

    query_vectors = sample_intersection_vectors(
        _base_patch, _24h_img, N_QUERY_VECTORS, RANDOM_SEED,
    )
    np.save(QUERIES_NPY, query_vectors)
    print(f"  Saved → {QUERIES_NPY}")
    print()

    # Normalise queries once; keep on GPU throughout all experiments
    q_tensor = torch.from_numpy(query_vectors).to(device)
    q_norm   = F.normalize(q_tensor, dim=1)              # (1000, 768)
    del q_tensor

    # ── Step 2: ground truth per experiment ───────────────────────────────────
    ground_truth: dict[str, dict[int, set[str]]] = {}

    for freq, project in EXPERIMENTS:
        print("=" * 64)
        print(f"  Experiment : {project}  ({freq})")

        db        = lancedb.connect(str(DB_URI / project))
        patch_tbl = db.open_table("patch_embeddings")

        # Load full table to CPU RAM
        db_embs, patch_ids = load_patch_table(patch_tbl)

        # Normalise on CPU (in-place to avoid a second full-size copy)
        print("  Normalising embeddings on CPU …")
        t0    = time.perf_counter()
        norms = np.linalg.norm(db_embs, axis=1, keepdims=True)
        np.clip(norms, a_min=1e-12, a_max=None, out=norms)
        db_embs /= norms
        del norms
        print(f"  Normalised in {time.perf_counter() - t0:.1f}s")

        # Exact top-k on GPU
        print(f"  Computing exact top-{TOP_K} on GPU "
              f"(chunk_size={CHUNK_SIZE:,}) …")
        t0 = time.perf_counter()
        top_row_idxs = compute_topk_cosine_gpu(
            db_embs, q_norm, TOP_K, CHUNK_SIZE, device,
        )                              # (1000, 10) global row indices
        elapsed = time.perf_counter() - t0
        print(f"  GPU search done in {elapsed:.1f}s  "
              f"({N_QUERY_VECTORS / elapsed:.0f} q/s)")

        # Map row indices → patch_id strings (globally unique identifier)
        patch_ids_arr = np.array(patch_ids)
        gt_ids = patch_ids_arr[top_row_idxs]             # (1000, 10)
        ground_truth[project] = {
            q_idx: set(gt_ids[q_idx].tolist())
            for q_idx in range(N_QUERY_VECTORS)
        }
        print(f"  Ground truth stored for {N_QUERY_VECTORS} queries  ✓")

        # Free CPU and GPU memory before next experiment
        del db_embs, patch_ids, patch_ids_arr, gt_ids, top_row_idxs
        torch.cuda.empty_cache()
        print()

    # ── Step 3: save ground truth ─────────────────────────────────────────────
    with open(GROUND_TRUTH_PKL, "wb") as f:
        pickle.dump(ground_truth, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Ground truth saved → {GROUND_TRUTH_PKL}")

    # Quick sanity check
    sample_exp   = EXPERIMENTS[0][1]
    sample_entry = ground_truth[sample_exp][0]
    print(f"  Sample ({sample_exp}, query 0): {len(sample_entry)} patch_ids")
    print(f"  e.g. {next(iter(sample_entry))!r}")


if __name__ == "__main__":
    main()
