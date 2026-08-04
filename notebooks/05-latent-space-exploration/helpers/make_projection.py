#!/usr/bin/env python3
"""
make_projection.py — UMAP coordinates and HDBSCAN clusters for any embedding table.

Stage 1 of two. Deliberately domain-agnostic: it reads an id column and a vector
column out of a Lance table and writes coordinates. It knows nothing about
patches, images, ERA5 or storms — all of that lives in stage 2
(`write_projection_table.py`), which turns this script's output into a
projection table the explorer notebook can open.

Two UMAP runs come off **one shared kNN graph**: a low-dimensional one for
viewing (2-D) and a higher-dimensional one for clustering. Sharing the graph is
the point, not an optimisation — two independently-fitted embeddings are under
no obligation to agree, and clusters found in the clustering embedding would
land scattered across the view. Sharing it also roughly halves the runtime,
since the neighbour search dominates.

Nothing about dimensionality is hardcoded. The embedding width is read from the
Arrow type, and both output dimensionalities travel downstream as array shapes.

Inputs
------
    A Lance table with an id column and a fixed-size-list vector column.
    Defaults target the patch-embedding tables written by
    notebooks/02-generate-embeddings, but --table/--id-column/--vector-column
    point it at anything.

Outputs
-------
    <out>/projection.npz        — view/z coordinates, cluster labels, id fingerprint
    <out>/projection_meta.json  — every resolved parameter, for provenance

Usage
-----
    # Local iteration on a subsample, no GPU:
    uv run python notebooks/05-latent-space-exploration/helpers/make_projection.py \
        --experiment /path/to/lancedb/experiments/era5/dinov3_24h \
        --limit 50000 --allow-cpu

    # Full run on a CUDA node:
    python make_projection.py --experiment "$NVME_DB_DIR/dinov3_24h"

Backends
--------
    CUDA  — cuML (RAPIDS). The only supported way to run this at full scale.
    CPU   — umap-learn + scikit-learn. Correct but slow; hours at ~1M rows,
            so it is gated behind --allow-cpu.

    There is no MPS path: RAPIDS is CUDA-only and Numba has no Metal target.
    On Apple Silicon, use --limit to iterate on a subsample.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pyarrow as pa

# The embedding metric. The DINOv3 vectors are L2-normalized, so euclidean and
# cosine rank neighbours identically and euclidean is the cheaper of the two.
# Exposed as a flag for tables whose vectors are not normalized.
DEFAULT_METRIC = "euclidean"

# The viewer's contract is a column named `x` and a column named `y`
# (see `Projection` in helpers/experiment.py), so the view embedding cannot have
# fewer than two components. It may have more — see --n-components-view.
MIN_VIEW_COMPONENTS = 2


# ── timing / logging ─────────────────────────────────────────────────────────

_T0 = time.perf_counter()


def log(msg: str) -> None:
    """Timestamped line. Batch logs are read long after the fact, so every line
    carries elapsed wall-clock rather than relying on the reader to diff them."""
    print(f"[{time.perf_counter() - _T0:8.1f}s] {msg}", flush=True)


class Stage:
    """Context manager that logs a stage's start and duration."""

    def __init__(self, label: str):
        self.label = label

    def __enter__(self):
        log(f"{self.label} ...")
        self.t = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if exc[0] is None:
            log(f"{self.label} done in {time.perf_counter() - self.t:.1f}s")
        return False


# ── reading ──────────────────────────────────────────────────────────────────

def _as_array(column) -> pa.Array:
    """Collapse a ChunkedArray to a single Array. Returns Arrays unchanged."""
    if isinstance(column, pa.ChunkedArray):
        column = column.combine_chunks()
        if isinstance(column, pa.ChunkedArray):
            if column.num_chunks == 0:
                return pa.array([], type=column.type)
            if column.num_chunks == 1:
                return column.chunk(0)
            return pa.concat_arrays(column.chunks)
    return column


def vector_width(column: pa.Array) -> int:
    """Embedding width, read from the data rather than assumed.

    768 today, 1024 for a larger backbone, anything tomorrow. Taking it from the
    Arrow type is what lets the rest of this script stay width-agnostic.
    """
    if pa.types.is_fixed_size_list(column.type):
        return column.type.list_size
    if pa.types.is_list(column.type) or pa.types.is_large_list(column.type):
        if len(column) == 0:
            raise ValueError("cannot infer embedding width from an empty table")
        return len(column[0])
    raise TypeError(
        f"{column.type} is not a vector column; expected a fixed-size-list or list of floats"
    )


def to_matrix(column: pa.Array, dim: int) -> np.ndarray:
    """(N, dim) float32 view of a list column.

    One reshape over the flattened value buffer. The obvious
    `df[col].to_list()` route materializes a Python object per row, which at
    ~1M rows is the single slowest step in the whole pipeline -- minutes, and a
    large transient allocation, for what is really a reinterpretation of one
    contiguous buffer.
    """
    values = column.flatten().to_numpy(zero_copy_only=False)
    expected = len(column) * dim
    if values.size != expected:
        # Ragged rows or nulls: the reshape below would silently misalign every
        # row after the first offender, so refuse instead.
        raise ValueError(
            f"vector column has {values.size} values, expected {expected} "
            f"({len(column)} rows x {dim}); ragged or null rows are not supported"
        )
    values = np.asarray(values, dtype=np.float32)
    if not values.flags.writeable:
        # numpy hands back a read-only view onto Arrow's immutable buffer, and
        # pynndescent's numba kernels reject read-only arrays outright. One copy
        # here (N x dim x 4 bytes) is unavoidable for the CPU path; the CUDA
        # path would copy to the device regardless.
        values = values.copy()
    return values.reshape(len(column), dim)


def sample_offsets(n_rows: int, limit: int | None, seed: int) -> np.ndarray | None:
    """Sorted row offsets for a uniform subsample, or None to read everything.

    Uniform rather than stratified: this script does not know what the rows
    mean, so it cannot stratify by anything. For patch tables stored in
    image order a uniform draw still lands across every source image, because
    the sample interval is far smaller than one image's patch count.
    """
    if limit is None or limit >= n_rows:
        return None
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_rows, limit, replace=False))


def read_table(tbl, id_column: str, vector_column: str, limit: int | None, seed: int):
    """Read the id and vector columns. Returns (X, ids, dim, offsets).

    Only these two columns are touched; on a patch table this skips the
    `image_id` strings entirely, which stage 2 fetches later for the rows it
    actually needs.

    `offsets` is None for a full read, or the sampled row offsets. Stage 2 needs
    them to read back exactly the rows that were embedded -- recomputing the
    draw from the seed would work today but would break the moment the sampling
    logic changed, and the failure would be a silently mis-joined table.
    """
    dataset = tbl.to_lance()
    n_rows = dataset.count_rows()
    offsets = sample_offsets(n_rows, limit, seed)
    columns = [id_column, vector_column]

    if offsets is None:
        arrow = dataset.to_table(columns=columns)
    else:
        log(f"subsampling {len(offsets):,} of {n_rows:,} rows (seed {seed})")
        arrow = dataset.take(offsets, columns=columns)

    vectors = _as_array(arrow.column(vector_column))
    dim = vector_width(vectors)
    X = to_matrix(vectors, dim)
    ids = _as_array(arrow.column(id_column))
    return X, ids, dim, offsets


def fingerprint(ids: pa.Array) -> str:
    """Order-sensitive digest of the id column.

    Stage 2 re-reads identities from the source table and recomputes this. A
    mismatch means the table was rewritten or compacted between the two stages
    and the row offsets no longer line up -- which would otherwise join every
    coordinate to the wrong patch, silently and plausibly.

    A digest rather than the ids themselves: `patch_id` is a string, and a
    million of them would dominate an otherwise-small npz.

    Hashing the Arrow buffers directly does *not* work: an array read as part of
    one column set can carry different padding, or be a slice of a larger
    buffer, than the logically identical array read alongside other columns.
    Serializing through IPC first normalizes the layout, so the digest depends
    on the values and their order and nothing else.
    """
    batch = pa.record_batch([_as_array(ids)], names=["id"])
    sink = pa.BufferOutputStream()
    with pa.ipc.new_stream(sink, batch.schema) as writer:
        writer.write_batch(batch)
    h = hashlib.sha256()
    h.update(memoryview(sink.getvalue()))
    return h.hexdigest()


# ── backends ─────────────────────────────────────────────────────────────────

def detect_cuda() -> tuple[bool, str]:
    """(usable, reason). Never raises.

    Returns the *reason* as well as the verdict, because "cuML/CUDA not
    available" on its own is close to useless on a cluster: a missing package, a
    login node with no GPU, and a driver mismatch all look identical, and the
    fixes are completely different.

    Deliberately probes only what this script actually uses. An earlier version
    imported `cuml.common.device_selection`, which is far more specific than
    needed -- a working cuML that had merely moved that submodule would have
    been reported as no CUDA at all.
    """
    try:
        import cuml
    except Exception as exc:
        return False, f"cuML did not import -- {type(exc).__name__}: {exc}"

    version = getattr(cuml, "__version__", "?")
    try:
        import cupy

        count = cupy.cuda.runtime.getDeviceCount()
    except Exception as exc:
        return False, (
            f"cuML {version} imported but no usable CUDA device "
            f"-- {type(exc).__name__}: {exc}"
        )

    if count == 0:
        return False, (
            f"cuML {version} imported but 0 CUDA devices are visible. "
            "On a cluster this usually means a login node -- request a GPU node."
        )

    try:
        name = cupy.cuda.runtime.getDeviceProperties(0)["name"].decode()
    except Exception:
        name = "unknown device"
    return True, f"cuML {version}, {count} device(s), {name}"


class CpuBackend:
    """umap-learn + scikit-learn. Correct, and slow at scale."""

    name = "cpu"

    def __init__(self, metric: str, seed: int, deterministic: bool):
        self.metric = metric
        self.seed = seed
        self.deterministic = deterministic

    def knn_graph(self, X, k):
        from sklearn.utils import check_random_state
        from umap.umap_ import nearest_neighbors

        # Returns (indices, dists, search_index) -- exactly the tuple UMAP's
        # `precomputed_knn` expects, so it can be handed to both fits unchanged.
        return nearest_neighbors(
            X, k, self.metric, {}, False, check_random_state(self.seed),
            low_memory=False, verbose=True,
        )

    def embed(self, X, graph, n_components, min_dist, k):
        import umap

        return umap.UMAP(
            n_components=n_components,
            n_neighbors=k,
            min_dist=min_dist,
            metric=self.metric,
            precomputed_knn=graph,
            # Setting random_state disables umap-learn's parallel optimization,
            # which costs far more than it looks. Reproducibility is opt-in.
            random_state=self.seed if self.deterministic else None,
            verbose=True,
        ).fit_transform(X).astype(np.float32)

    def cluster(self, Z, min_cluster_size, min_samples, selection, max_cluster_size):
        from sklearn.cluster import HDBSCAN

        h = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=selection,
            max_cluster_size=max_cluster_size,
            metric="euclidean",
            copy=True,  # explicit: the default flips in scikit-learn 1.10
        ).fit(Z)
        return h.labels_, h.probabilities_

    def kmeans(self, Z, k):
        from sklearn.cluster import MiniBatchKMeans

        return MiniBatchKMeans(
            n_clusters=k, random_state=self.seed, n_init="auto"
        ).fit_predict(Z)


class CudaBackend:
    """cuML (RAPIDS). The path this pipeline is designed around.

    Untested on macOS by construction -- cuML cannot be installed there -- so
    this class is kept as thin as possible and every method mirrors the CPU one
    above, which *is* exercised locally.
    """

    name = "cuda"

    def __init__(self, metric: str, seed: int, deterministic: bool):
        self.metric = metric
        self.seed = seed
        self.deterministic = deterministic
        self.shared_graph = True

    def knn_graph(self, X, k):
        from cuml.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=k, metric=self.metric)
        nn.fit(X)
        return nn.kneighbors_graph(X, n_neighbors=k, mode="distance")

    def embed(self, X, graph, n_components, min_dist, k):
        from cuml.manifold import UMAP

        model = UMAP(
            n_components=n_components,
            n_neighbors=k,
            min_dist=min_dist,
            metric=self.metric,
            random_state=self.seed if self.deterministic else None,
            verbose=True,
        )
        if graph is not None and self.shared_graph:
            try:
                return np.asarray(model.fit_transform(X, knn_graph=graph), dtype=np.float32)
            except TypeError:
                # The knn_graph keyword has moved between cuML releases. Falling
                # back is correct but means the two embeddings are fitted
                # independently, so say so loudly -- clusters may not line up
                # with the view, and that would otherwise look like a bug in the
                # clustering rather than a missing keyword.
                log("WARNING: this cuML build does not accept knn_graph=; "
                    "falling back to independent kNN per embedding. Clusters may "
                    "not align with the view.")
                self.shared_graph = False
        return np.asarray(model.fit_transform(X), dtype=np.float32)

    def cluster(self, Z, min_cluster_size, min_samples, selection, max_cluster_size):
        from cuml.cluster import HDBSCAN

        h = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=selection,
            # cuML spells "no ceiling" as 0 rather than None.
            max_cluster_size=max_cluster_size or 0,
            metric="euclidean",
        ).fit(Z)
        return np.asarray(h.labels_), np.asarray(h.probabilities_)

    def kmeans(self, Z, k):
        from cuml.cluster import KMeans

        return np.asarray(KMeans(n_clusters=k, random_state=self.seed).fit_predict(Z))


# ── scale-derived defaults ───────────────────────────────────────────────────

def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


def default_min_cluster_size(n_rows: int) -> int:
    """HDBSCAN's floor on a cluster, scaled to the table.

    A constant tuned at 1M produces thousands of micro-clusters at 50k and one
    blob at 10M, so the default tracks N. Roughly 0.15% of the table, bounded so
    tiny and enormous inputs both stay sane.
    """
    return _clamp(n_rows // 700, 25, 5000)


def default_min_samples(min_cluster_size: int) -> int:
    """Conservativeness of the density estimate.

    Left unset, HDBSCAN ties this to min_cluster_size, which at these scales
    labels most of the table as noise.
    """
    return _clamp(min_cluster_size // 20, 5, 100)


def default_kmeans_k(n_rows: int) -> int:
    """A no-noise categorical to fall back on when HDBSCAN is still being tuned."""
    return _clamp(int(np.sqrt(n_rows) / 20), 8, 64)


# ── main ─────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="UMAP + HDBSCAN over a Lance embedding table.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--experiment",
        default=os.environ.get("EXPERIMENT_DIR"),
        help="Directory holding the Lance tables. Defaults to $EXPERIMENT_DIR.",
    )
    p.add_argument("--table", default="patch_embeddings")
    p.add_argument(
        "--id-column", default="patch_id",
        help="Unique row key. Note patch_index is NOT unique -- it repeats per image.",
    )
    p.add_argument("--vector-column", default="embedding")
    p.add_argument(
        "--out", default=None,
        help="Output directory. Defaults to <experiment>/_projection.",
    )

    p.add_argument("--n-neighbors", type=int, default=30,
                   help="Also the k of the shared kNN graph; UMAP requires k >= n_neighbors.")
    p.add_argument("--n-components-view", type=int, default=2,
                   help=f"View embedding width; must be >= {MIN_VIEW_COMPONENTS}.")
    p.add_argument("--n-components-cluster", type=int, default=10,
                   help="Clustering embedding width. 5, 10, 20, 30 all work.")
    p.add_argument("--min-dist", type=float, default=0.1,
                   help="View embedding: spread points out for legibility.")
    p.add_argument("--min-dist-cluster", type=float, default=0.0,
                   help="Clustering embedding: pack tightly for clean density estimates.")
    p.add_argument("--metric", default=DEFAULT_METRIC)

    p.add_argument("--min-cluster-size", type=int, default=None,
                   help="Default scales with row count.")
    p.add_argument("--min-samples", type=int, default=None,
                   help="Default scales with min-cluster-size.")
    p.add_argument("--cluster-selection-method", choices=("eom", "leaf"), default="eom",
                   help="eom favours few large clusters and collapses to one blob on "
                        "continuum-like data; leaf finds fine-grained structure at the "
                        "cost of far more noise. Try leaf if eom returns ~1 cluster.")
    p.add_argument("--max-cluster-size", type=int, default=None,
                   help="Ceiling on cluster size: a cluster larger than this is refused "
                        "and its sub-clusters emitted instead. The other fix for an eom "
                        "run that collapses into one blob. Keep it well above "
                        "--min-cluster-size (~10x) or almost everything becomes noise. "
                        "No effect with --cluster-selection-method leaf.")
    p.add_argument("--kmeans-k", type=int, default=None,
                   help="Default scales with row count. 0 disables k-means.")

    p.add_argument("--limit", type=int, default=None,
                   help="Uniformly subsample this many rows. For local iteration.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deterministic", action="store_true",
                   help="Reproducible but markedly slower: this serializes UMAP's optimization.")
    p.add_argument("--allow-cpu", action="store_true",
                   help="Permit the CPU backend. Hours at ~1M rows.")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if not args.experiment:
        print("error: --experiment is required (or set $EXPERIMENT_DIR)", file=sys.stderr)
        return 2
    experiment = Path(args.experiment).expanduser()
    if not experiment.is_dir():
        print(f"error: no such experiment directory: {experiment}", file=sys.stderr)
        return 2
    if args.n_components_view < MIN_VIEW_COMPONENTS:
        print(
            f"error: --n-components-view must be >= {MIN_VIEW_COMPONENTS}; the viewer "
            "reads columns named x and y.",
            file=sys.stderr,
        )
        return 2

    out_dir = Path(args.out).expanduser() if args.out else experiment / "_projection"

    import lancedb

    db = lancedb.connect(str(experiment))
    tbl = db.open_table(args.table)

    # ── read ────────────────────────────────────────────────────────────────
    with Stage(f"reading {args.table}.{args.vector_column}"):
        X, ids, dim, row_offsets = read_table(
            tbl, args.id_column, args.vector_column, args.limit, args.seed
        )
    n_rows = len(X)

    # Cross-check the width against the experiment config. A mismatch means one
    # of the two is stale; reshaping on the wrong width yields a plausible
    # matrix and a meaningless map, so fail rather than guess.
    config = {}
    try:
        cfg = db.open_table("config").to_pandas()
        config = dict(zip(cfg["key"], cfg["value"]))
    except Exception:
        pass
    declared = config.get("embedding_dim")
    if declared and int(declared) != dim:
        print(
            f"error: config says embedding_dim={declared} but the table holds "
            f"{dim}-d vectors.",
            file=sys.stderr,
        )
        return 2

    # ── resolve parameters ──────────────────────────────────────────────────
    k = args.n_neighbors
    min_cluster_size = args.min_cluster_size or default_min_cluster_size(n_rows)
    min_samples = args.min_samples or default_min_samples(min_cluster_size)
    kmeans_k = default_kmeans_k(n_rows) if args.kmeans_k is None else args.kmeans_k

    use_cuda, cuda_reason = detect_cuda()
    if not use_cuda and not args.allow_cpu:
        # The gate exists to stop a multi-hour CPU run starting by accident, so
        # the cost it quotes has to match the actual row count -- warning about
        # "hours" for a 3k smoke test would just train the reader to ignore it.
        if n_rows > 200_000:
            cost = f"A CPU run over {n_rows:,} rows takes hours."
        elif n_rows > 50_000:
            cost = f"A CPU run over {n_rows:,} rows takes many minutes."
        else:
            cost = f"{n_rows:,} rows is small enough to run on CPU."
        print(
            f"error: no CUDA backend.\n"
            f"       {cuda_reason}\n"
            f"       {cost} Re-run with --allow-cpu to accept that"
            f"{',' if n_rows > 50_000 else '.'}\n"
            + ("       or with --limit to work on a subsample.\n"
               if n_rows > 50_000 else ""),
            file=sys.stderr,
        )
        return 1

    Backend = CudaBackend if use_cuda else CpuBackend
    backend = Backend(args.metric, args.seed, args.deterministic)

    log("=" * 68)
    log(f"experiment      {experiment}")
    log(f"table           {args.table}  (id={args.id_column}, vec={args.vector_column})")
    log(f"rows x dim      {n_rows:,} x {dim}")
    log(f"backend         {backend.name.upper()}  ({cuda_reason})")
    if not use_cuda:
        log("                CPU backend: expect hours at full scale.")
        if sys.platform == "darwin":
            log("                (No MPS path exists: RAPIDS is CUDA-only, Numba has no Metal target.)")
    log(f"n_neighbors/k   {k}")
    log(f"view            {args.n_components_view}-d, min_dist={args.min_dist}")
    log(f"cluster         {args.n_components_cluster}-d, min_dist={args.min_dist_cluster}")
    log(f"hdbscan         min_cluster_size={min_cluster_size}, min_samples={min_samples}, "
        f"selection={args.cluster_selection_method}, "
        f"max_cluster_size={args.max_cluster_size or 'none'}"
        f"{'' if args.min_cluster_size else '  (sizes scaled from row count)'}")
    log(f"kmeans          k={kmeans_k}" if kmeans_k else "kmeans          disabled")
    log(f"metric          {args.metric}")
    log(f"out             {out_dir}")
    log("=" * 68)

    # ── shared kNN graph ────────────────────────────────────────────────────
    # Built once and reused. See the module docstring: this is what keeps the
    # clustering embedding and the view embedding talking about the same
    # neighbourhood structure.
    with Stage(f"kNN graph (k={k})"):
        graph = backend.knn_graph(X, k)

    # ── embeddings ──────────────────────────────────────────────────────────
    # UMAP's optimization is an opaque stretch: cuML logs only coarse phases and
    # umap-learn logs per-epoch. Neither reports a percentage, so the elapsed
    # timer is the honest progress signal here.
    with Stage(f"UMAP view -> {args.n_components_view}-d"):
        view = backend.embed(X, graph, args.n_components_view, args.min_dist, k)

    with Stage(f"UMAP cluster -> {args.n_components_cluster}-d"):
        z = backend.embed(X, graph, args.n_components_cluster, args.min_dist_cluster, k)

    # ── clustering ──────────────────────────────────────────────────────────
    with Stage(f"HDBSCAN on {z.shape[1]}-d ({args.cluster_selection_method})"):
        labels, probs = backend.cluster(
            z, min_cluster_size, min_samples, args.cluster_selection_method,
            args.max_cluster_size,
        )
    labels = np.asarray(labels, dtype=np.int32)
    probs = np.asarray(probs, dtype=np.float32)
    n_clusters = int(labels.max()) + 1 if labels.size and labels.max() >= 0 else 0
    noise = float((labels < 0).mean()) if labels.size else 0.0
    log(f"  {n_clusters} clusters, {noise:.1%} noise")

    # A single cluster holding almost everything is a useless colour-by, and it
    # is the expected outcome of excess-of-mass selection on data that forms a
    # continuum rather than discrete groups -- which patch embeddings often do.
    # Say so here: the alternative is a map that looks broken for a reason that
    # is not obvious from the output.
    dominance = float(np.bincount(labels[labels >= 0]).max() / labels.size) if n_clusters else 0.0
    if n_clusters and dominance > 0.9:
        log(f"  WARNING: one cluster holds {dominance:.1%} of all points.")
        if args.cluster_selection_method == "eom" and not args.max_cluster_size:
            log("           This is what 'eom' does on continuum-like data. Either:")
            log(f"             --max-cluster-size {int(n_rows * 0.10)}   (10% of N, keeps eom's semantics)")
            log("             --cluster-selection-method leaf   (finer, much more noise)")

    # A ceiling too close to the floor leaves almost nothing that can qualify as
    # a cluster, and the symptom is a plausible cluster count hiding near-total
    # noise rather than an error.
    if args.max_cluster_size and args.max_cluster_size < 5 * min_cluster_size:
        log(f"  WARNING: --max-cluster-size {args.max_cluster_size} is close to "
            f"--min-cluster-size {min_cluster_size}.")
        log(f"           Coverage is {1 - noise:.1%}; a cluster must fit between the two, "
            "so a narrow window starves them.")
    if n_clusters == 0:
        log("  WARNING: no clusters found -- try a smaller --min-cluster-size.")

    if kmeans_k:
        with Stage(f"k-means (k={kmeans_k})"):
            km = np.asarray(backend.kmeans(z, kmeans_k), dtype=np.int16)
    else:
        km = np.zeros(n_rows, dtype=np.int16)

    # ── write ───────────────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    npz_path = out_dir / "projection.npz"
    meta_path = out_dir / "projection_meta.json"

    arrays = {
        "view": view.astype(np.float32),
        "z": z.astype(np.float32),
        "cluster": labels,
        "cluster_prob": probs,
        "kmeans": km,
    }
    # Numeric ids ride along directly; anything else (patch_id is a string) is
    # represented only by the digest, which is all stage 2 needs.
    id_numpy = ids.to_numpy(zero_copy_only=False)
    if np.issubdtype(id_numpy.dtype, np.number):
        arrays["id"] = id_numpy
    if row_offsets is not None:
        arrays["row_offsets"] = row_offsets.astype(np.int64)
    np.savez(npz_path, **arrays)

    meta = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_experiment": str(experiment),
        "source_table": args.table,
        "id_column": args.id_column,
        "vector_column": args.vector_column,
        "id_fingerprint": fingerprint(ids),
        "n_rows": int(n_rows),
        "subsampled": row_offsets is not None,
        "embedding_dim": int(dim),
        "n_components_view": int(view.shape[1]),
        "n_components_cluster": int(z.shape[1]),
        "n_neighbors": int(k),
        "min_dist": float(args.min_dist),
        "min_dist_cluster": float(args.min_dist_cluster),
        "metric": args.metric,
        "min_cluster_size": int(min_cluster_size),
        "min_samples": int(min_samples),
        "cluster_selection_method": args.cluster_selection_method,
        "max_cluster_size": args.max_cluster_size,
        "cluster_dominance": round(dominance, 4),
        "coverage": round(1.0 - noise, 4),
        "kmeans_k": int(kmeans_k),
        "n_clusters": n_clusters,
        "noise_fraction": noise,
        "limit": args.limit,
        "seed": args.seed,
        "deterministic": bool(args.deterministic),
        "backend": backend.name,
        "shared_knn_graph": bool(getattr(backend, "shared_graph", True)),
        "elapsed_seconds": round(time.perf_counter() - _T0, 1),
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")

    size_mb = npz_path.stat().st_size / 1e6
    log(f"wrote {npz_path}  ({size_mb:.1f} MB)")
    log(f"wrote {meta_path}")
    log(f"next: write_projection_table.py --experiment {experiment} --npz {npz_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
