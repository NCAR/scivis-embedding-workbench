#!/usr/bin/env python3
"""
benchmark_search_latency.py — Patch ANN search latency across temporal experiments.

Methodology
-----------
1. Identify the intersection query set: image_ids present in ALL 5 experiments
   are exactly the 24h-aligned midnight images (every other experiment is a
   superset). Fetch their patch embeddings from the 1h table and draw 1,000
   vectors (fixed seed=42). These exact vectors are reused for every experiment
   so comparisons are apples-to-apples.

2. For each experiment (24h → 12h → 6h → 3h → 1h, smallest to largest):
     - Pause for a sudo-purge cache clear
     - Read num_partitions from the live index metadata
     - nprobes = int(num_partitions * 0.05)   (5% of clusters)
     - 100 warm-up queries  (discarded)
     - 1,000 timed queries  (recorded) via
         table.search(v).nprobes(nprobes).limit(10).to_arrow()

3. Report Mean / p50 / p95 / p99 latency (ms) and save to
   results/search_latency.csv.

4. Save a publication-quality line + shaded-band figure to
   results/search_latency.pdf and .png.

Dataset (3 years hourly ERA5, 2016-01-01 → 2018-12-31, 896 patches/image):

    Experiment   Images    Patches    num_partitions   nprobes
    24h          ~1,096    ~0.98M          256            12
    12h          ~2,192    ~1.96M          512            25
    6h           ~4,384    ~3.93M        1,024            51
    3h           ~8,768    ~7.86M        2,048           102
    1h          ~26,304   ~23.60M        4,096           204

Usage
-----
    uv run python notebooks/04-benchmarking/benchmark_search_latency.py

Edit the CONFIGURATION section below to match your environment.
"""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import lancedb
import numpy as np
import pyarrow.compute as pc

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

# NCAR Casper
PROJECT_ROOT = Path("/glade/work/ncheruku/research/sample_data")
# Local Mac (uncomment to switch):
# PROJECT_ROOT = Path("/Users/ncheruku/Documents/Work/sample_data")

DB_URI = PROJECT_ROOT / "data" / "lancedb" / "experiments" / "era5"

# Experiments ordered smallest → largest (left-to-right on the scaling plot).
EXPERIMENTS = [
    ("24h", "dinov3_rect_24h"),
    ("12h", "dinov3_rect_12h"),
    ("6h",  "dinov3_rect_6h"),
    ("3h",  "dinov3_rect_3h"),
    ("1h",  "dinov3_rect_1h"),
]

N_QUERY_VECTORS = 1_000   # vectors drawn from the 24h intersection pool
N_WARMUP        = 100     # queries discarded before timing begins
N_TEST          = 1_000   # queries actually timed
TOP_K           = 10      # limit() for each ANN query
NPROBES_FRAC    = 0.05    # fraction of IVF clusters to probe
RANDOM_SEED     = 42

RESULTS_DIR = Path(__file__).parent / "results"
CSV_PATH    = RESULTS_DIR / "search_latency.csv"
PLOT_PDF    = RESULTS_DIR / "search_latency.pdf"
PLOT_PNG    = RESULTS_DIR / "search_latency.png"


# ── HELPERS ───────────────────────────────────────────────────────────────────

def get_num_partitions(tbl: lancedb.table.Table) -> int:
    """Read IVF partition count from the live index metadata.

    Falls back to estimating via 2^round(log2(N/4096)) if the metadata is
    unavailable (same formula used when building the index).
    """
    try:
        for idx_cfg in tbl.list_indices():
            stats = tbl.index_stats(idx_cfg.name)
            if stats is not None and stats.num_indices is not None:
                return int(stats.num_indices)
    except Exception:
        pass
    n = tbl.count_rows()
    return max(1, 2 ** round(math.log2(n / 4096)))


def sample_intersection_vectors(
    base_patch_tbl: lancedb.table.Table,
    smallest_img_tbl: lancedb.table.Table,
    n: int,
    seed: int,
) -> np.ndarray:
    """Sample n patch embeddings guaranteed to exist in every experiment.

    The 24h experiment contains only midnight images (00:00 UTC). Every other
    experiment is a strict superset, so 24h image_ids form the safe intersection.
    Vectors are fetched from the 1h base table (full precision) then subsampled.

    Parameters
    ----------
    base_patch_tbl    : patch_embeddings table of the 1h base experiment
    smallest_img_tbl  : image_embeddings table of the 24h experiment
    n                 : number of vectors to return
    seed              : RNG seed for reproducibility
    """
    # Step 1 — collect all image_ids that exist in the 24h experiment
    anchor_ids = (
        smallest_img_tbl.to_lance()
        .scanner(columns=["image_id"])
        .to_table()
        .column("image_id")
        .to_pylist()
    )
    print(f"  Intersection image_ids : {len(anchor_ids):,}  "
          f"(images present in all 5 experiments)")

    # Step 2 — scan patch embeddings for those image_ids from the 1h table
    anchor_patches = (
        base_patch_tbl.to_lance()
        .scanner(
            columns=["embedding"],
            filter=pc.field("image_id").isin(anchor_ids),
        )
        .to_table()
    )
    embs = np.stack(
        anchor_patches.column("embedding").to_pylist()
    ).astype(np.float32)
    print(f"  Patch pool             : {len(embs):,}")

    # Step 3 — draw n vectors without replacement
    rng = np.random.default_rng(seed)
    chosen = rng.choice(len(embs), size=min(n, len(embs)), replace=False)
    vecs = embs[chosen]
    print(f"  Query vectors sampled  : {len(vecs):,}  shape={vecs.shape}")
    return vecs


def run_benchmark(
    tbl: lancedb.table.Table,
    query_vectors: np.ndarray,
    nprobes: int,
    n_warmup: int = N_WARMUP,
    n_test: int = N_TEST,
) -> np.ndarray:
    """Return per-query latencies (ms) after warm-up."""
    rng = np.random.default_rng(0)

    # Warm-up: load index pages into RAM, prime internal caches
    warmup_idx = rng.choice(len(query_vectors), size=n_warmup, replace=False)
    for i in warmup_idx:
        tbl.search(query_vectors[i]).nprobes(nprobes).limit(TOP_K).to_arrow()

    # Timed queries
    test_idx = rng.choice(len(query_vectors), size=n_test, replace=False)
    latencies_ms = np.empty(n_test, dtype=np.float64)
    for k, i in enumerate(test_idx):
        t0 = time.perf_counter()
        tbl.search(query_vectors[i]).nprobes(nprobes).limit(TOP_K).to_arrow()
        latencies_ms[k] = (time.perf_counter() - t0) * 1_000.0

    return latencies_ms


def dir_size_gib(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024**3


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: build intersection query set ──────────────────────────────────
    print("=" * 62)
    print("Building intersection query set …")
    _base_db    = lancedb.connect(str(DB_URI / "dinov3_rect_1h"))
    _base_patch = _base_db.open_table("patch_embeddings")

    _24h_db  = lancedb.connect(str(DB_URI / "dinov3_rect_24h"))
    _24h_img = _24h_db.open_table("image_embeddings")

    query_vectors = sample_intersection_vectors(
        _base_patch, _24h_img, N_QUERY_VECTORS, RANDOM_SEED
    )
    print()

    # ── Step 2: benchmark each experiment ─────────────────────────────────────
    records: list[dict] = []

    for freq, project in EXPERIMENTS:
        exp_path = DB_URI / project
        size_gib = dir_size_gib(exp_path) if exp_path.exists() else float("nan")

        print("=" * 62)
        print(f"  Experiment : {project}  ({freq})")
        print(f"  DB size    : {size_gib:.1f} GiB")
        print()
        print(
            "  ⚠️  Please clear the OS page cache before proceeding.\n"
            "     Open a NEW terminal and run:\n\n"
            "         sudo purge\n\n"
            "     Then press Enter here to start the benchmark."
        )
        input("  [Press Enter when ready] ")
        print()

        db        = lancedb.connect(str(exp_path))
        patch_tbl = db.open_table("patch_embeddings")
        n_patches = patch_tbl.count_rows()
        n_images  = db.open_table("image_embeddings").count_rows()

        num_partitions = get_num_partitions(patch_tbl)
        nprobes        = max(1, int(num_partitions * NPROBES_FRAC))

        print(f"  n_images       : {n_images:,}")
        print(f"  n_patches      : {n_patches:,}")
        print(f"  num_partitions : {num_partitions}")
        print(f"  nprobes (5%)   : {nprobes}")
        print(f"  Running {N_WARMUP} warm-up + {N_TEST} timed queries …")

        latencies = run_benchmark(patch_tbl, query_vectors, nprobes)

        mean_ms = float(np.mean(latencies))
        p50_ms  = float(np.percentile(latencies, 50))
        p95_ms  = float(np.percentile(latencies, 95))
        p99_ms  = float(np.percentile(latencies, 99))
        max_ms  = float(np.max(latencies))

        print(f"\n  ── Latency results ──────────────────────────")
        print(f"  Mean : {mean_ms:8.2f} ms")
        print(f"  p50  : {p50_ms:8.2f} ms")
        print(f"  p95  : {p95_ms:8.2f} ms")
        print(f"  p99  : {p99_ms:8.2f} ms")
        print(f"  Max  : {max_ms:8.2f} ms")
        print()

        records.append({
            "experiment":     project,
            "resolution":     freq,
            "n_images":       n_images,
            "n_patches":      n_patches,
            "db_size_gib":    round(size_gib, 2),
            "num_partitions": num_partitions,
            "nprobes":        nprobes,
            "mean_ms":        round(mean_ms, 3),
            "p50_ms":         round(p50_ms, 3),
            "p95_ms":         round(p95_ms, 3),
            "p99_ms":         round(p99_ms, 3),
            "max_ms":         round(max_ms, 3),
        })

    # ── Step 3: save CSV ──────────────────────────────────────────────────────
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    print(f"Results saved → {CSV_PATH}")

    # ── Step 4: figure ────────────────────────────────────────────────────────
    _make_plot(records)
    print(f"Plot saved    → {PLOT_PDF}")
    print(f"              → {PLOT_PNG}")


# ── PLOT ──────────────────────────────────────────────────────────────────────

def _fmt_patches(n: int) -> str:
    """Format a patch count as a compact string, e.g. '0.98M', '23.60M'."""
    return f"{n / 1e6:.2f}M" if n >= 1_000_000 else f"{n / 1e3:.0f}K"


def _make_plot(records: list[dict]) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    plt.rcParams.update({
        "font.family":       "serif",
        "font.size":         10,
        "axes.labelsize":    11,
        "axes.titlesize":    11,
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   9,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "figure.dpi":        200,
    })

    freqs     = [r["resolution"] for r in records]
    x_vals    = np.array([r["n_patches"]  for r in records], dtype=float)
    means     = np.array([r["mean_ms"]    for r in records])
    p95s      = np.array([r["p95_ms"]     for r in records])
    p99s      = np.array([r["p99_ms"]     for r in records])
    nprobes   = [r["nprobes"]             for r in records]

    fig, ax = plt.subplots(figsize=(7, 4))

    # Shaded envelopes — draw back-to-front so blue sits on top
    ax.fill_between(x_vals, p95s,  p99s,  alpha=0.18, color="#b2182b",
                    label="p99 envelope")
    ax.fill_between(x_vals, means, p95s,  alpha=0.28, color="#2166ac",
                    label="p95 envelope")

    # Mean line
    ax.plot(x_vals, means, "o-", color="#2166ac", lw=2, ms=7,
            zorder=5, label="Mean latency")

    # Annotate: resolution label to the left of each marker
    for xv, mv, freq in zip(x_vals, means, freqs):
        ax.annotate(
            freq, xy=(xv, mv),
            textcoords="offset points", xytext=(-6, 7),
            ha="center", fontsize=8.5, color="#333",
            fontweight="bold",
        )

    # Annotate: nprobes below each marker
    for xv, mv, k in zip(x_vals, means, nprobes):
        ax.annotate(
            f"k={k}", xy=(xv, mv),
            textcoords="offset points", xytext=(0, -14),
            ha="center", fontsize=7.5, color="#555",
        )

    # x-axis: log scale, ticks at actual patch counts
    ax.set_xscale("log")
    ax.set_xticks(x_vals)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda v, _: _fmt_patches(int(v))
    ))
    ax.xaxis.set_minor_locator(ticker.NullLocator())

    ax.set_xlabel("Patches in Index")
    ax.set_ylabel("Query Latency (ms)")
    ax.set_title(
        "IVF-PQ ANN Search Latency vs. Index Size\n"
        r"(cosine metric, $k$=10, nprobes = 5 % of partitions, "
        "1 000 queries per experiment)",
        pad=10,
    )

    ax.set_ylim(bottom=0)
    ax.grid(axis="both", linestyle="--", linewidth=0.5, alpha=0.45, zorder=0)
    ax.legend(loc="upper left", framealpha=0.85)

    fig.tight_layout()
    fig.savefig(PLOT_PDF, bbox_inches="tight")
    fig.savefig(PLOT_PNG, bbox_inches="tight", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
