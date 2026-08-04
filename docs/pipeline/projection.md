# Projection

**Stage 1:** `notebooks/05-latent-space-exploration/helpers/make_projection.py`
**Stage 2:** `notebooks/05-latent-space-exploration/helpers/write_projection_table.py`
**Batch:** `notebooks/05-latent-space-exploration/helpers/run_projection.pbs`

Turns a table of patch embeddings into a 2-D map with clusters, written as a
`umap_*` table the latent-space explorer can open.

---

## Why two scripts

Stage 1 is expensive and needs a GPU. Stage 2 takes seconds. Splitting them
means re-running the metadata join, or re-clustering with a different
`min_cluster_size`, never costs another UMAP run.

```
patch_embeddings.lance
        │
        │  stage 1  (GPU, ~10–25 min at 1M rows)
        ▼
projection.npz + projection_meta.json      ← ~55 MB, portable
        │
        │  stage 2  (CPU, seconds)
        ▼
umap_patch_001.lance                       ← opens in latent_exploration.py
```

The `.npz` is small enough to `scp` off Casper, so stage 2 and the notebook can
run on a laptop against a copy of the source DBs.

---

## Stage 1 — coordinates and clusters

Domain-agnostic by design: it reads an id column and a vector column and writes
coordinates. It knows nothing about patches, images or storms.

Two UMAP runs come off **one shared kNN graph**:

| Embedding | Width | `min_dist` | Used for |
|---|---|---|---|
| view | 2 | 0.1 | the map you look at — spread out for legibility |
| cluster | 10 | 0.0 | HDBSCAN input — packed tight for clean density estimates |

Sharing the graph is the point, not an optimisation. Two independently-fitted
embeddings are under no obligation to agree, and clusters found in the
clustering embedding would land scattered across the map. It also roughly halves
the runtime, since the neighbour search dominates.

Clustering runs on the 10-D embedding rather than the 2-D one: the map is
optimised for viewing, and its distances are heavily distorted by the crowding
constraint.

### Nothing about dimensionality is hardcoded

The embedding width is read from the Arrow type — 768 for the DINOv3 ViT-B
experiments, 1024 for `dinov3_sat_rect`, anything for whatever comes next — and
cross-checked against `config["embedding_dim"]`. A mismatch is a hard error:
reshaping on the wrong width produces a plausible matrix and a meaningless map.

`--n-components-cluster` is free (5, 10, 20, 30 …) and travels downstream as an
array shape, becoming `z0 … z{C-1}` in the output table.

`--n-components-view` defaults to 2 and that is the supported case. A 3-D run
stores all three components, but **the notebook still draws the `x`/`y` plane** —
`helpers/scatter.py` rasterizes through datashader, which has no 3-D aggregation
path. The third component shows up as a `view2` colour-by, which is a useful way
to see what it captured, but it is not a 3-D scatter.

### Parameters worth touching

| Flag | Default | Note |
|---|---|---|
| `--n-neighbors` | 30 | Also the *k* of the shared graph. Above the UMAP default of 15 on purpose: neighbouring patches within one frame are near-duplicates, and a small neighbourhood mostly rediscovers "this is one weather frame". |
| `--min-cluster-size` | scales with N | ~0.15% of the table. A constant tuned at 1M gives thousands of micro-clusters at 50k. |
| `--min-samples` | scales | Left to HDBSCAN's own default, most of the table comes back as noise. |
| `--cluster-selection-method` | `eom` | See below. The first thing to change if the map comes back as one colour. |
| `--max-cluster-size` | none | Ceiling on cluster size; the other fix for the same problem. See below. |
| `--limit` | none | Uniform subsample. The way to iterate locally. |
| `--metric` | `euclidean` | The DINOv3 vectors are L2-normalized, so euclidean and cosine rank neighbours identically and euclidean is cheaper. |

### `eom` vs `leaf` — read this before tuning anything else

Measured on `dinov3_24h`, 50k patches, `n_neighbors=30`:

| Method | `min_cluster_size` | Clusters | Noise | Largest cluster |
|---|---|---|---|---|
| `eom` | 71 | 2 | 1.6% | **97.7%** |
| `eom` | 250 | 2 | 1.6% | **97.7%** |
| `leaf` | 71 | 84 | 72.2% | 0.8% |
| `leaf` | 250 | 31 | 66.3% | 3.1% |

HDBSCAN's default selection method, excess-of-mass (`eom`), prefers a few large
stable clusters. On data that forms a *continuum* rather than discrete groups it
collapses into a single blob holding nearly everything — which is a useless
colour-by, and looks like a broken pipeline rather than a property of the data.
Patch embeddings of a continuous physical field are exactly that kind of data:
there is no reason to expect discrete "kinds" of atmospheric patch.

`leaf` selects the finest-grained clusters in the hierarchy instead, and finds
real structure — at the cost of a lot of noise, since everything between the
dense cores goes unlabelled.

Neither is *wrong*; they answer different questions. `eom` asks "what are the
major modes", `leaf` asks "what are the tight groups". Stage 1 keeps `eom` as
the default to match HDBSCAN's own convention, and **prints a warning whenever
one cluster holds more than 90% of the points**, naming `leaf` as the fix. The
warning and the observed dominance are both recorded in
`projection_meta.json`.

Note these numbers come from a 50k subsample. Density structure can look
different at the full million, where the estimates are better resolved — so
re-check rather than assuming the subsample's answer carries over.

### `--max-cluster-size`

A **ceiling**: HDBSCAN refuses to emit any cluster holding more than this many
points and descends into its sub-clusters instead. It is the other fix for the
`eom` collapse, and it keeps excess-of-mass semantics rather than swapping them
out — you still get "the major modes", just not the degenerate one-mode answer.

**It only affects `eom`.** With `leaf` it is a no-op, because leaf clusters are
already below any sensible ceiling (measured: 31 clusters and 3.1% largest, with
and without a 5000 ceiling).

Sweeping it on the same 50k embedding, `min_cluster_size=250`:

| `max_cluster_size` | % of N | Clusters | Noise | Largest | **Covered** |
|---|---|---|---|---|---|
| none | – | 2 | 1.6% | 97.7% | 98.4% |
| 25000 / 20000 / 15000 | 50–30% | 10 | 39.8% | 27.1% | 60.2% |
| 10000 | 20% | 21 | 57.8% | 13.8% | 42.2% |
| 5000 / 2500 | 10–5% | 28 | 63.5% | 4.8% | 36.5% |
| 1000 | 2% | 29 | 71.7% | 1.9% | 28.3% |
| 500 | 1% | 18 | 87.2% | 1.0% | 12.8% |
| 300 | 0.6% | 6 | 96.7% | 0.6% | 3.3% |

Three things to read off it:

**It behaves as a step function, not a dial.** Several values give *identical*
results. Nothing changes until the ceiling crosses some cluster's natural size.
You are choosing a shelf, not tuning a number, so the exact value barely matters
— anywhere in 5–10% of N gives the same answer.

**Keep it well above `min_cluster_size` — roughly 10×.** A cluster has to fit
between the two bounds, so a narrow window starves them: at 2× the floor,
coverage collapses to 12.8%, and at 1.2× to 3.3%. That failure is quiet — you
get a plausible cluster count while almost everything is unlabelled. Stage 1
warns when the ratio drops below 5×.

**Judge by coverage, not cluster count.** 28 clusters at 36% coverage and 10 at
60% are different answers to different questions, and cluster count alone hides
that. Express your choice as a fraction of N so it transfers from a subsample to
the full run.

Stage 1 prints a concrete suggestion when it detects the collapse:

```
WARNING: one cluster holds 97.8% of all points.
           --max-cluster-size 5000   (10% of N, keeps eom's semantics)
           --cluster-selection-method leaf   (finer, much more noise)
```

### Backends

| Detected | Stack | 1M × 768 |
|---|---|---|
| CUDA | cuML (RAPIDS) | ~10–25 min |
| anything else | umap-learn + scikit-learn | hours — gated behind `--allow-cpu` |

There is no MPS path. RAPIDS is CUDA-only and Numba has no Metal target, so on
Apple Silicon this runs on CPU cores regardless of the GPU. Use `--limit` to
iterate locally and run the full table on Casper.

Progress reporting is honest about its limits: every stage is timed and logged,
but UMAP's optimization itself is an opaque stretch — cuML emits coarse phase
logging and umap-learn logs per-epoch, and neither reports a percentage.

---

## Stage 2 — the projection table

Where the patch-specific knowledge lives. The join is mechanical; there is no
ERA5- or hurricane-specific code.

1. **Join every scalar column** from the source image table on `id == image_id`.
   Blobs, vectors and **strings** are skipped. `hurricane_present`,
   `max_wind_kts` and the rest arrive because they are scalar columns, not
   because anything recognises them. A different dataset's columns arrive the
   same way.

    Strings are dropped because `infer_color_roles` classifies only bools and
    numerics — a string column can never reach the colour-by dropdown whatever
    its cardinality, so joining it is pure cost. On this dataset `filename`
    plus three storm-detail strings were 23% of the finished table, none of
    them reachable from the UI. `--keep-strings` opts back in.
2. **Expand timestamps** into `_year/_month/_day/_hour/_dayofyear`, detected by
   dtype — a column called `valid_time` works without a code change.
3. **Derive `patch_row`/`patch_col`** from `patch_index` and the grid from
   `patch_grid(config)`, which handles rectangular grids (ERA5 is 16×56).
4. **Derive `lat`/`lon`** only when the source table records a `spatial_extent`.
   Absent it, the columns are simply not written.
5. **Classify colour-by columns** with `infer_color_roles()` and
   `usable_color_columns()` — the viewer's *own* inference — recorded as
   `color_by` schema metadata. Writer and viewer therefore cannot disagree, and
   constant columns drop out automatically (an all-zero `hour` in a
   24h-subsampled experiment would otherwise paint everything one colour).

Every rule fails soft. With no source image DB you still get a working table of
coordinates, clusters and patch geometry.

### `cluster_top`

`cluster` is written raw, and `cluster_top` alongside it.

Datashader allocates one aggregate plane per category, so `helpers/data.py` caps
categoricals at `MAX_CATEGORIES` (64) and prunes anything above it. A run that
finds 200 clusters would have `cluster` dropped from the dropdown entirely.
`cluster_top` keeps the largest clusters and folds the tail into `OTHER` (`-2`),
which is kept distinct from HDBSCAN's noise label (`-1`) — "too small to give
its own colour" is a different statement from "belongs to no cluster".

This keeps a usable categorical colour-by whatever `min_cluster_size` produces,
without distorting the clustering to satisfy a rendering limit.

### Integrity

Stage 1 leaves `image_id` out of the `.npz` — a million strings would dominate
the file — so stage 2 re-reads it. An order-sensitive fingerprint of the id
column guards that: if the table is compacted or rewritten between the two
stages, stage 2 aborts rather than joining every coordinate to the wrong patch.

Note the id column defaults to `patch_id`, not `patch_index`. `patch_index` runs
0…895 *within each image* and is not unique across the table.

---

## Running on Casper

```bash
qsub -A <YOUR_PROJECT_CODE> \
     -v PROJECT_DIR=/glade/work/$USER/scivis-embedding-workbench,\
GLADE_EXPERIMENTS=/glade/work/$USER/research/sample_data/data/lancedb/experiments/era5,\
EXPERIMENT_NAME=dinov3_24h \
     notebooks/05-latent-space-exploration/helpers/run_projection.pbs
```

Nothing personal is baked into the script — every path comes from the
environment.

### Choosing a node

```
#PBS -l select=1:ncpus=8:mem=64GB:ngpus=1:gpu_type=a100
```

**`gpu_type=a100` is deliberate.** Casper has both V100s and A100s and will hand
you a V100 if you do not ask. (`scripts/casper/casper-marimo.env` defaults to
`v100` for the interactive notebook launcher, where it is the right call.) A
V100 has 16 or 32 GB against the A100's 40/80 and materially lower fp32
throughput; a 1M × 768 run fits but takes longer, and 2M × 768 is tight on a
16 GB card.

| Resource | Guidance |
|---|---|
| `ncpus` | 8 is plenty. Stage 1 is GPU-bound; the CPUs read Lance and write the npz. |
| `mem` | Must exceed the host-side matrix — `N × D × 4` bytes, so ~3 GB at 1M × 768 and ~6 GB at 2M × 768 — plus one copy and Arrow overhead. 64 GB is generous and costs nothing in the queue. |
| `walltime` | ~1 h covers 1M end to end; the GPU work is 10–25 min of that. Raise for 2M+. |

### Why there is no NVMe staging

`notebooks/04-benchmarking/run_benchmark.pbs` copies its tables to
`/local_scratch` first, and it is right to: that job is an ANN latency benchmark
doing thousands of random-access lookups with the page cache purged between
runs, so storage latency is the thing being measured.

This job is the opposite. It reads the embedding column **once**, sequentially,
then works in GPU memory. Staging would mean reading the whole table off GLADE,
writing it to NVMe, and reading it back — strictly more I/O than reading it once.

It also does not scale: `patch_embeddings` is ~3 GB for `dinov3_24h` and ~6 GB
for `dinov3_12h`, but **~72 GB for `dinov3_1h`**, which may not fit in
`/local_scratch` at all.

Staging would only pay off if one job read the table repeatedly — a parameter
sweep. A `--recluster-from` flag reusing the saved 10-D embedding is the cheaper
answer there, since re-clustering needs no second UMAP run.

### RAPIDS

cuML is **not** in `pyproject.toml` and cannot be: it is CUDA-only, and the
project has to resolve on macOS for local development. Load it separately:

```bash
CUML_ENV="module load conda && conda activate rapids"
```

If cuML will not import, stage 1 refuses to run without `--allow-cpu` — a broken
RAPIDS environment fails fast instead of quietly taking six hours.

### First thing to check in the log

```
"backend": "cuda"
"shared_knn_graph": true
```

`backend: cpu` means RAPIDS did not load. `shared_knn_graph: false` means this
cuML build did not accept the `knn_graph=` keyword (it has moved between
releases) and the two embeddings were fitted independently — in which case
clusters may not line up with the map, and that is the cause rather than
anything wrong with the clustering.

---

## Local iteration

```bash
# ~1 minute, no GPU
uv run python notebooks/05-latent-space-exploration/helpers/make_projection.py \
    --experiment <experiments>/dinov3_24h --limit 50000 --allow-cpu

uv run python notebooks/05-latent-space-exploration/helpers/write_projection_table.py \
    --experiment <experiments>/dinov3_24h --table-name umap_patch_smoke
```

Then open `latent_exploration.py` and pick `umap_patch_smoke` from the projection
dropdown. Tables must be named `umap_*` to be listed.

Tuning `n_neighbors` and `min_cluster_size` on a subsample first is strongly
recommended — you do not want each iteration to be a queued GPU job.
