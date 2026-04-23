#!/usr/bin/env python3
"""
benchmark_search_latency.py — Patch ANN search latency across temporal experiments.

Methodology
-----------
1. Sample 1,000 query vectors from the 1 h base experiment (fixed seed → reused
   for every experiment so comparisons are apples-to-apples).
2. For each experiment (1h → 3h → 6h → 12h → 24h), pause for a sudo-purge cache
   clear, then run:
     - 100 warm-up queries  (discarded)
     - 1,000 timed queries  (recorded)
   using table.search(v).nprobes(nprobes).limit(10).to_arrow().
   nprobes = int(num_partitions * 0.05)  — always 5 % of clusters.
3. Report Mean / p95 / p99 latency (ms) to stdout and to results/search_latency.csv.
4. Save a publication-quality figure to results/search_latency.pdf + .png.

Usage
-----
    uv run python notebooks/04-benchmarking/benchmark_search_latency.py

Edit the CONFIGURATION section below to match your environment.
"""

from __future__ import annotations

import csv
import math
import sys
import time
from pathlib import Path

import lancedb
import numpy as np

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

# NCAR Casper
PROJECT_ROOT = Path("/glade/work/ncheruku/research/sample_data")
# Local Mac (uncomment to switch):
# PROJECT_ROOT = Path("/Users/ncheruku/Documents/Work/sample_data")

DB_URI = PROJECT_ROOT / "data" / "lancedb" / "experiments" / "era5"

# Experiments to benchmark, in the order they will be run.
# (Start with the biggest DB so cache-clear effort is front-loaded.)
EXPERIMENTS = [
    ("1h",  "dinov3_rect_1h"),
    ("3h",  "dinov3_rect_3h"),
    ("6h",  "dinov3_rect_6h"),
    ("12h", "dinov3_rect_12h"),
    ("24h", "dinov3_rect_24h"),
]

N_QUERY_VECTORS = 1_000   # vectors sampled from the 1 h base table
N_WARMUP        = 100     # queries discarded before timing begins
N_TEST          = 1_000   # queries actually timed
TOP_K           = 10      # limit() parameter
NPROBES_FRAC    = 0.05    # fraction of IVF clusters to probe
RANDOM_SEED     = 42

RESULTS_DIR = Path(__file__).parent / "results"
CSV_PATH    = RESULTS_DIR / "search_latency.csv"
PLOT_PDF    = RESULTS_DIR / "search_latency.pdf"
PLOT_PNG    = RESULTS_DIR / "search_latency.png"

# ── HELPERS ───────────────────────────────────────────────────────────────────

def get_num_partitions(tbl: lancedb.table.Table) -> int:
    """Read IVF partition count from the table's index metadata.

    Falls back to estimating via the same formula used in subsample_embeddings.py
    if the index metadata is unavailable.
    """
    try:
        for idx_cfg in tbl.list_indices():
            stats = tbl.index_stats(idx_cfg.name)
            if stats is not None and stats.num_indices is not None:
                return int(stats.num_indices)
    except Exception:
        pass
    # Fallback: nearest power-of-2 to N/4096
    n = tbl.count_rows()
    return max(1, 2 ** round(math.log2(n / 4096)))


def sample_query_vectors(tbl: lancedb.table.Table, n: int, seed: int) -> np.ndarray:
    """Randomly sample `n` embedding vectors from `tbl` (without loading blobs)."""
    n_total = tbl.count_rows()
    rng = np.random.default_rng(seed)
    # Draw 3× what we need so we have room to subsample after a take()
    pool_size = min(n_total, max(n * 3, 5_000))
    pool_indices = sorted(rng.choice(n_total, size=pool_size, replace=False).tolist())

    arrow = tbl.to_lance().take(pool_indices, columns=["embedding"])
    embs = np.stack(arrow.column("embedding").to_pylist()).astype(np.float32)

    chosen = rng.choice(len(embs), size=n, replace=False)
    return embs[chosen]   # shape (n, 768)


def run_benchmark(
    tbl: lancedb.table.Table,
    query_vectors: np.ndarray,
    nprobes: int,
    n_warmup: int = N_WARMUP,
    n_test:   int = N_TEST,
) -> np.ndarray:
    """Run warm-up then timed queries; return per-query latencies in ms."""
    rng = np.random.default_rng(0)

    # ── Warm-up (cache the index pages, JIT any internal paths) ──────────────
    warmup_idx = rng.choice(len(query_vectors), size=n_warmup, replace=False)
    for i in warmup_idx:
        tbl.search(query_vectors[i]).nprobes(nprobes).limit(TOP_K).to_arrow()

    # ── Timed queries ─────────────────────────────────────────────────────────
    test_idx = rng.choice(len(query_vectors), size=n_test, replace=False)
    latencies_ms = np.empty(n_test, dtype=np.float64)
    for k, i in enumerate(test_idx):
        t0 = time.perf_counter()
        tbl.search(query_vectors[i]).nprobes(nprobes).limit(TOP_K).to_arrow()
        latencies_ms[k] = (time.perf_counter() - t0) * 1_000.0

    return latencies_ms


def dir_size_gb(path: Path) -> float:
    """Return total on-disk size under `path` in GiB."""
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024**3


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Step 1: sample universal query vectors from the 1 h base ─────────────
    print("=" * 60)
    print("Sampling query vectors from 1 h base experiment …")
    _base_db    = lancedb.connect(str(DB_URI / "dinov3_rect_1h"))
    _base_patch = _base_db.open_table("patch_embeddings")
    query_vectors = sample_query_vectors(_base_patch, N_QUERY_VECTORS, RANDOM_SEED)
    print(f"  Sampled {len(query_vectors)} vectors  shape={query_vectors.shape}")
    print()

    # ── Step 2: benchmark each experiment ─────────────────────────────────────
    records: list[dict] = []

    for freq, project in EXPERIMENTS:
        exp_path = DB_URI / project
        size_gb  = dir_size_gb(exp_path) if exp_path.exists() else float("nan")

        print("=" * 60)
        print(f"  Experiment : {project}  ({freq})")
        print(f"  DB size    : {size_gb:.1f} GiB")
        print()
        print(
            f"  ⚠️  Please clear the OS page cache before proceeding.\n"
            f"     Open a NEW terminal and run:\n\n"
            f"         sudo purge\n\n"
            f"     Then press Enter here to start the benchmark."
        )
        input("  [Press Enter when ready] ")
        print()

        db         = lancedb.connect(str(exp_path))
        patch_tbl  = db.open_table("patch_embeddings")
        n_patches  = patch_tbl.count_rows()
        n_images   = db.open_table("image_embeddings").count_rows()

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

        print(f"\n  ── Latency results ──────────────────────")
        print(f"  Mean : {mean_ms:7.2f} ms")
        print(f"  p50  : {p50_ms:7.2f} ms")
        print(f"  p95  : {p95_ms:7.2f} ms")
        print(f"  p99  : {p99_ms:7.2f} ms")
        print(f"  Max  : {max_ms:7.2f} ms")
        print()

        records.append({
            "experiment":      project,
            "resolution":      freq,
            "n_images":        n_images,
            "n_patches":       n_patches,
            "db_size_gib":     round(size_gb, 2),
            "num_partitions":  num_partitions,
            "nprobes":         nprobes,
            "mean_ms":         round(mean_ms, 3),
            "p50_ms":          round(p50_ms, 3),
            "p95_ms":          round(p95_ms, 3),
            "p99_ms":          round(p99_ms, 3),
            "max_ms":          round(max_ms, 3),
        })

    # ── Step 3: save CSV ──────────────────────────────────────────────────────
    fieldnames = list(records[0].keys())
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(records)
    print(f"Results saved → {CSV_PATH}")

    # ── Step 4: publication-quality plot ──────────────────────────────────────
    _make_plot(records)
    print(f"Plot saved    → {PLOT_PDF}")
    print(f"              → {PLOT_PNG}")


# ── PLOT ──────────────────────────────────────────────────────────────────────

def _make_plot(records: list[dict]) -> None:
    import matplotlib.pyplot as plt
    import matplotlib.ticker as ticker

    plt.rcParams.update({
        "font.family":      "serif",
        "font.size":        10,
        "axes.labelsize":   11,
        "axes.titlesize":   11,
        "xtick.labelsize":  9,
        "ytick.labelsize":  9,
        "legend.fontsize":  9,
        "axes.spines.top":  False,
        "axes.spines.right": False,
        "figure.dpi":       200,
    })

    labels    = [r["resolution"] for r in records]
    means     = np.array([r["mean_ms"]  for r in records])
    p95s      = np.array([r["p95_ms"]   for r in records])
    p99s      = np.array([r["p99_ms"]   for r in records])
    nprobes   = [r["nprobes"]           for r in records]
    n_patches = [r["n_patches"]         for r in records]

    x     = np.arange(len(labels))
    width = 0.22

    fig, (ax_lat, ax_probe) = plt.subplots(
        1, 2,
        figsize=(9, 3.8),
        gridspec_kw={"width_ratios": [3, 1]},
    )

    # ── Left panel: latency bars ──────────────────────────────────────────────
    COLORS = {
        "mean": "#2166ac",   # blue
        "p95":  "#f4a582",   # salmon
        "p99":  "#b2182b",   # red
    }

    b_mean = ax_lat.bar(x - width, means, width, label="Mean",
                        color=COLORS["mean"], zorder=3)
    b_p95  = ax_lat.bar(x,         p95s,  width, label="p95",
                        color=COLORS["p95"],  zorder=3)
    b_p99  = ax_lat.bar(x + width, p99s,  width, label="p99",
                        color=COLORS["p99"],  zorder=3)

    # Annotate nprobes above each group
    for i, (xi, np_val) in enumerate(zip(x, nprobes)):
        ax_lat.text(
            xi, max(means[i], p95s[i], p99s[i]) + ax_lat.get_ylim()[1] * 0.02,
            f"k={np_val}",
            ha="center", va="bottom", fontsize=7.5, color="#444",
        )

    ax_lat.set_xticks(x)
    ax_lat.set_xticklabels(labels)
    ax_lat.set_xlabel("Temporal Resolution")
    ax_lat.set_ylabel("Query Latency (ms)")
    ax_lat.set_title("ANN Search Latency by Temporal Resolution")
    ax_lat.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax_lat.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
    ax_lat.legend(loc="upper left")
    ax_lat.set_ylim(bottom=0)

    # Add patch count as secondary x-axis annotation
    ax2 = ax_lat.secondary_xaxis("bottom")
    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [f"{p/1e6:.1f}M\npatches" for p in n_patches],
        fontsize=7,
    )
    ax2.tick_params(axis="x", pad=22, length=0)
    ax_lat.tick_params(axis="x", pad=4)

    # ── Right panel: nprobes bar ──────────────────────────────────────────────
    ax_probe.bar(x, nprobes, width=0.5, color="#4dac26", zorder=3)
    for xi, val in zip(x, nprobes):
        ax_probe.text(xi, val + 0.5, str(val), ha="center", va="bottom",
                      fontsize=8, color="#333")

    ax_probe.set_xticks(x)
    ax_probe.set_xticklabels(labels)
    ax_probe.set_xlabel("Temporal Resolution")
    ax_probe.set_ylabel("nprobes (5% of partitions)")
    ax_probe.set_title("Probed Clusters")
    ax_probe.grid(axis="y", linestyle="--", linewidth=0.5, alpha=0.6, zorder=0)
    ax_probe.set_ylim(bottom=0)

    fig.suptitle(
        "Patch Embedding ANN Search — Latency Benchmark\n"
        r"(IVF-PQ, cosine, $k$=10, nprobes=5% of partitions)",
        fontsize=10, y=1.02,
    )
    fig.tight_layout()

    fig.savefig(PLOT_PDF, bbox_inches="tight")
    fig.savefig(PLOT_PNG, bbox_inches="tight", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
