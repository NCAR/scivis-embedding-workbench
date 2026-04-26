#!/usr/bin/env python3
"""
02_benchmark_ann_latency.py — ANN latency, Recall@10, and QPS benchmark.

Run 01_generate_ground_truth.py first to produce the two input files:

    results/universal_queries.npy   — 1,000 query vectors (1000, 768) float32
    results/ground_truth.pkl        — exact top-10 patch_ids per query/experiment

Methodology
-----------
For each experiment (24h → 12h → 6h → 3h → 1h, smallest → largest index):

  1. Pause for sudo purge (Mac) to clear the OS page cache.
  2. Read num_partitions from the live IVF-PQ index; set nprobes = 5% of that.
  3. 100 warm-up queries (not timed).
  4. 1,000 timed queries:
       t0 = time.perf_counter()
       result = table.search(v).nprobes(nprobes).limit(10).to_arrow()
       latency = time.perf_counter() - t0          ← strictly wraps the ANN call

       After the timer:
       • Extract patch_id column from the Arrow result.
       • Recall@10 = |{ANN patch_ids} ∩ {GT patch_ids}| / 10

  5. Report Mean / p50 / p95 / p99 latency (ms), Mean Recall@10, QPS.

Outputs
-------
    results/search_latency.csv   — per-experiment metrics
    results/search_latency.pdf   — publication figure (latency + recall)
    results/search_latency.png   — raster version at 200 dpi

Dataset (3 years hourly ERA5, 2016-01-01 → 2018-12-31, 896 patches/image):

    Experiment   Images    Patches   num_partitions   nprobes
    24h          ~1,096    ~0.98M         256            12
    12h          ~2,192    ~1.96M         512            25
    6h           ~4,384    ~3.93M       1,024            51
    3h           ~8,768    ~7.86M       2,048           102
    1h          ~26,304   ~23.60M       4,096           204

Usage
-----
    uv run python notebooks/04-benchmarking/02_benchmark_ann_latency.py
"""

from __future__ import annotations

import csv
import math
import os
import pickle
import time
from pathlib import Path

import lancedb
import numpy as np
from tqdm import tqdm

# ── CONFIGURATION ─────────────────────────────────────────────────────────────

# NCAR Casper
PROJECT_ROOT = Path("/glade/work/ncheruku/research/sample_data")
# Local Mac (uncomment to switch):
# PROJECT_ROOT = Path("/Users/ncheruku/Documents/Work/sample_data")

if "NVME_DB_DIR" in os.environ:
    DB_URI = Path(os.environ["NVME_DB_DIR"])
    print("[INFO] NVME_DB_DIR environment variable detected.")
    print(f"[INFO] Routing database connections to local NVMe storage: {DB_URI}")
else:
    DB_URI = PROJECT_ROOT / "data" / "lancedb" / "experiments" / "era5"
    print("[INFO] NVME_DB_DIR not found in environment.")
    print(f"[INFO] Defaulting to network GLADE storage: {DB_URI}")

EXPERIMENTS = [
    ("24h", "dinov3_24h"),
    ("12h", "dinov3_12h"),
    ("6h",  "dinov3_6h"),
    ("3h",  "dinov3_3h"),
    ("1h",  "dinov3_1h"),
]

N_WARMUP     = 100
N_TEST       = 1_000
TOP_K        = 10
NPROBES_FRAC = 0.05

RESULTS_DIR      = Path(__file__).parent / "results"
QUERIES_NPY      = RESULTS_DIR / "universal_queries.npy"
GROUND_TRUTH_PKL = RESULTS_DIR / "ground_truth.pkl"
CSV_PATH         = RESULTS_DIR / "search_latency.csv"
PLOT_PDF         = RESULTS_DIR / "search_latency.pdf"
PLOT_PNG         = RESULTS_DIR / "search_latency.png"


# ── HELPERS ───────────────────────────────────────────────────────────────────

def get_num_partitions(tbl: lancedb.table.Table) -> int:
    """Read IVF partition count from the live index metadata.

    Falls back to 2^round(log2(N/4096)) if metadata is unavailable —
    the same formula used when the index was built.
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


def run_benchmark(
    tbl: lancedb.table.Table,
    query_vectors: np.ndarray,
    ground_truth: dict[int, set[str]],
    nprobes: int,
    n_warmup: int = N_WARMUP,
    n_test: int   = N_TEST,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Run warm-up then timed ANN queries; return latencies, recalls, and QPS.

    The perf_counter timer strictly wraps the LanceDB search call only.
    Recall computation happens outside the timer window.

    Parameters
    ----------
    tbl           : LanceDB patch_embeddings table (with IVF-PQ index)
    query_vectors : (Q, 768) float32 universal query vectors
    ground_truth  : mapping query_idx → set of exact top-10 patch_id strings
    nprobes       : number of IVF clusters to probe per query
    n_warmup      : warm-up queries to discard (loads index pages into RAM)
    n_test        : queries to time and record

    Returns
    -------
    latencies_ms : np.ndarray of shape (n_test,) — per-query latency in ms
    recalls      : np.ndarray of shape (n_test,) — per-query Recall@10 in [0,1]
    qps          : float — queries per second (wall-clock over full test loop)
    """
    rng = np.random.default_rng(0)

    # ── Warm-up: fill page cache, prime internal LanceDB caches ──────────────
    warmup_idx = rng.choice(len(query_vectors), size=n_warmup, replace=False)
    for i in tqdm(warmup_idx, desc="Warm-up queries", leave=False):
        tbl.search(query_vectors[i]).nprobes(nprobes).limit(TOP_K).to_arrow()

    # ── Timed queries ─────────────────────────────────────────────────────────
    test_idx     = rng.choice(len(query_vectors), size=n_test, replace=False)
    latencies_ms = np.empty(n_test, dtype=np.float64)
    recalls      = np.empty(n_test, dtype=np.float64)

    loop_start = time.perf_counter()

    for k, i in enumerate(tqdm(test_idx, desc="Timed queries", leave=False)):
        # ── Timer strictly wraps only the ANN search call ────────────────────
        t0     = time.perf_counter()
        result = (tbl.search(query_vectors[i])
                  .nprobes(nprobes)
                  .refine_factor(10)
                  .select(["patch_id"])
                  .limit(TOP_K)
                  .to_pandas())
        latencies_ms[k] = (time.perf_counter() - t0) * 1_000.0
        # ── Recall computed outside the timer ─────────────────────────────────
        # patch_id is the globally unique per-row identifier.
        # patch_index (0-895) is only unique within an image, not across the
        # table — using it for set intersection would produce false matches.
        ann_ids   = set(result["patch_id"].tolist())
        truth_ids = ground_truth[int(i)]
        recalls[k] = len(ann_ids & truth_ids) / TOP_K

    total_time_s = time.perf_counter() - loop_start
    qps = n_test / total_time_s

    return latencies_ms, recalls, qps


def dir_size_gib(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1024**3


def _fmt_patches(n: int) -> str:
    return f"{n / 1e6:.2f}M" if n >= 1_000_000 else f"{n / 1e3:.0f}K"


# ── MAIN ──────────────────────────────────────────────────────────────────────

def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load pre-computed queries and ground truth ─────────────────────────────
    if not QUERIES_NPY.exists():
        raise FileNotFoundError(
            f"{QUERIES_NPY} not found.\n"
            "Run 01_generate_ground_truth.py first."
        )
    if not GROUND_TRUTH_PKL.exists():
        raise FileNotFoundError(
            f"{GROUND_TRUTH_PKL} not found.\n"
            "Run 01_generate_ground_truth.py first."
        )

    query_vectors = np.load(QUERIES_NPY)
    with open(GROUND_TRUTH_PKL, "rb") as f:
        ground_truth_all: dict[str, dict[int, set[str]]] = pickle.load(f)

    print(f"Loaded {len(query_vectors):,} query vectors  shape={query_vectors.shape}")
    print(f"Ground truth experiments : {list(ground_truth_all.keys())}")
    print()

    # ── Benchmark loop ────────────────────────────────────────────────────────
    records: list[dict] = []

    for freq, project in EXPERIMENTS:
        exp_path = DB_URI / project
        size_gib = dir_size_gib(exp_path) if exp_path.exists() else float("nan")

        if project not in ground_truth_all:
            print(f"  [WARNING] No ground truth for {project} -- skipping.")
            continue

        print("=" * 64)
        print(f"  Experiment : {project}  ({freq})")
        print(f"  DB size    : {size_gib:.1f} GiB")
        print()
        print("  [INFO] Running in batch mode. Relying on strict RAM limits for out-of-core pressure.")
        print("  [INFO] Bypassing manual OS cache purge.")
        print()

        db        = lancedb.connect(str(exp_path))
        patch_tbl = db.open_table("patch_embeddings")
        n_patches = patch_tbl.count_rows()
        try:
            n_images = db.open_table("image_embeddings").count_rows()
        except Exception:
            n_images = None

        num_partitions = get_num_partitions(patch_tbl)
        nprobes        = max(1, int(num_partitions * NPROBES_FRAC))

        print(f"  n_images       : {f'{n_images:,}' if n_images is not None else 'N/A'}")
        print(f"  n_patches      : {n_patches:,}")
        print(f"  num_partitions : {num_partitions}")
        print(f"  nprobes (5%)   : {nprobes}")
        print(f"  Running {N_WARMUP} warm-up + {N_TEST} timed queries …")

        latencies, recalls, qps = run_benchmark(
            patch_tbl,
            query_vectors,
            ground_truth_all[project],
            nprobes,
        )

        mean_ms    = float(np.mean(latencies))
        p50_ms     = float(np.percentile(latencies, 50))
        p95_ms     = float(np.percentile(latencies, 95))
        p99_ms     = float(np.percentile(latencies, 99))
        max_ms     = float(np.max(latencies))
        mean_recall = float(np.mean(recalls))

        print(f"\n  ── Results ─────────────────────────────────────────")
        print(f"  Mean latency : {mean_ms:8.2f} ms")
        print(f"  p50          : {p50_ms:8.2f} ms")
        print(f"  p95          : {p95_ms:8.2f} ms")
        print(f"  p99          : {p99_ms:8.2f} ms")
        print(f"  Max          : {max_ms:8.2f} ms")
        print(f"  QPS          : {qps:8.1f} q/s")
        print(f"  Recall@10    : {mean_recall:8.4f}  ({mean_recall*100:.2f}%)")
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
            "qps":            round(qps, 1),
            "mean_recall":    round(mean_recall, 6),
        })

    if not records:
        print("No experiments completed — nothing to save.")
        return

    # ── Save CSV ──────────────────────────────────────────────────────────────
    with open(CSV_PATH, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        w.writeheader()
        w.writerows(records)
    print(f"Results saved → {CSV_PATH}")

    # ── Save figure ───────────────────────────────────────────────────────────
    _make_plot(records)
    print(f"Plot saved    → {PLOT_PDF}")
    print(f"              → {PLOT_PNG}")


# ── PLOT ──────────────────────────────────────────────────────────────────────

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
        "figure.dpi":        200,
    })

    freqs     = [r["resolution"] for r in records]
    x_vals    = np.array([r["n_patches"]   for r in records], dtype=float)
    means     = np.array([r["mean_ms"]     for r in records])
    p95s      = np.array([r["p95_ms"]      for r in records])
    p99s      = np.array([r["p99_ms"]      for r in records])
    recalls   = np.array([r["mean_recall"] for r in records])
    nprobes_l = [r["nprobes"]              for r in records]

    # ── Colours ───────────────────────────────────────────────────────────────
    LATENCY_COLOR = "#2166ac"   # blue
    P95_COLOR     = "#2166ac"
    P99_COLOR     = "#b2182b"   # red
    RECALL_COLOR  = "#1a9641"   # green

    fig, ax_lat = plt.subplots(figsize=(7.5, 4.2))

    # ── Latency: shaded envelopes + mean line ─────────────────────────────────
    ax_lat.fill_between(x_vals, p95s,  p99s,  alpha=0.18, color=P99_COLOR,
                        label="p99 envelope")
    ax_lat.fill_between(x_vals, means, p95s,  alpha=0.28, color=P95_COLOR,
                        label="p95 envelope")
    ax_lat.plot(x_vals, means, "o-", color=LATENCY_COLOR, lw=2, ms=7,
                zorder=5, label="Mean latency")

    # Annotate: resolution label above each latency marker
    for xv, mv, freq in zip(x_vals, means, freqs):
        ax_lat.annotate(
            freq, xy=(xv, mv),
            textcoords="offset points", xytext=(-6, 8),
            ha="center", fontsize=8.5, color="#333", fontweight="bold",
        )

    # Annotate: nprobes below each latency marker
    for xv, mv, k in zip(x_vals, means, nprobes_l):
        ax_lat.annotate(
            f"k={k}", xy=(xv, mv),
            textcoords="offset points", xytext=(0, -14),
            ha="center", fontsize=7.5, color="#555",
        )

    ax_lat.set_xscale("log")
    ax_lat.set_xticks(x_vals)
    ax_lat.xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: _fmt_patches(int(v)))
    )
    ax_lat.xaxis.set_minor_locator(ticker.NullLocator())
    ax_lat.set_xlabel("Patches in Index")
    ax_lat.set_ylabel("Query Latency (ms)", color=LATENCY_COLOR)
    ax_lat.tick_params(axis="y", colors=LATENCY_COLOR)
    ax_lat.yaxis.label.set_color(LATENCY_COLOR)
    ax_lat.spines["left"].set_color(LATENCY_COLOR)
    ax_lat.set_ylim(bottom=0)
    ax_lat.grid(axis="both", linestyle="--", linewidth=0.5, alpha=0.4, zorder=0)

    # ── Recall: secondary y-axis ──────────────────────────────────────────────
    ax_rec = ax_lat.twinx()
    ax_rec.plot(x_vals, recalls, "s--", color=RECALL_COLOR, lw=1.8, ms=7,
                zorder=6, label="Mean Recall@10")
    ax_rec.set_ylim(0.0, 1.05)
    ax_rec.set_ylabel("Mean Recall@10", color=RECALL_COLOR)
    ax_rec.tick_params(axis="y", colors=RECALL_COLOR)
    ax_rec.yaxis.label.set_color(RECALL_COLOR)
    ax_rec.spines["right"].set_color(RECALL_COLOR)
    ax_rec.spines["top"].set_visible(False)
    ax_rec.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda v, _: f"{v:.0%}")
    )

    # ── Combined legend ───────────────────────────────────────────────────────
    lines_lat, labs_lat = ax_lat.get_legend_handles_labels()
    lines_rec, labs_rec = ax_rec.get_legend_handles_labels()
    ax_lat.legend(
        lines_lat + lines_rec,
        labs_lat  + labs_rec,
        loc="upper left", framealpha=0.88,
    )

    ax_lat.set_title(
        "IVF-PQ ANN Search: Latency & Recall@10 vs. Index Size\n"
        r"(cosine, $k$=10, nprobes = 5 % of partitions, "
        "1 000 queries per experiment)",
        pad=10,
    )

    fig.tight_layout()
    fig.savefig(PLOT_PDF, bbox_inches="tight")
    fig.savefig(PLOT_PNG, bbox_inches="tight", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
