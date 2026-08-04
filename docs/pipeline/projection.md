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
        │  stage 1  (GPU, 4.5 min at 982k rows)
        ▼
projection.npz + projection_meta.json      ← 57 MB, portable
        │
        │  stage 2  (CPU, seconds)
        ▼
umap_patch_001.lance                       ← opens in latent_exploration.py
```

The `.npz` is small enough to `scp` off Casper, so stage 2 and the notebook can
run on a laptop against a copy of the source DBs.

### Where the time goes

Measured on `dinov3_24h`, 982,016 × 768, one A100-80GB, cuML 26.06:

| Stage | Time | Share |
|---|---|---|
| read + reshape | 3.9s | 1% |
| **kNN graph** | **190.3s** | **71%** |
| UMAP → 2-D | 5.2s | 2% |
| UMAP → 10-D | 9.0s | 3% |
| HDBSCAN | 49.0s | 18% |
| k-means | 0.1s | — |

Two things worth reading off it. The neighbour search dominates — clustering is
under a fifth of the run, which is the opposite of what most people expect. And
the 10-D UMAP costs a fraction of the 2-D one despite more output dimensions,
because it reuses the shared graph and pays only for optimisation. That is the
clearest evidence the sharing is working.

**The kNN search is exact brute force, so it scales quadratically:**

| Experiment | Patches | kNN |
|---|---|---|
| `dinov3_24h` | 982k | 190s (measured) |
| `dinov3_12h` | 2.0M | ~13 min (projected) |
| `dinov3_1h` | 23.6M | **~30 hours** (projected) |

The first two are comfortable; the third is not reachable this way and would
need an approximate neighbour search.

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

**At full scale it behaves differently.** The rows above are a 50k CPU
subsample. On the full 982k with the same *proportional* parameters
(`min_cluster_size` 1402 ≈ 0.14% of N in both cases) plus a ceiling:

| Run | Clusters | Noise | Largest |
|---|---|---|---|
| 50k, `eom`, no ceiling | 2 | 1.6% | 97.7% |
| 982k, `eom`, `--max-cluster-size 98000` | **235** | 33.7% | — |

The GPU run reproduced the 50k collapse almost exactly before the ceiling was
applied (2 clusters, 97.0% dominance) — a different backend and a different UMAP
implementation reaching the same answer, which is good evidence the collapse is
a property of the data rather than of either library.

Whether the 235 clusters come from the ceiling or from density resolving better
at 20× the points has not been separated. A control run without the ceiling
would settle it, and matters if you need to describe the method: if the ceiling
is load-bearing, the cluster count is partly an imposed constraint rather than a
discovered structure.

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

**On the real run it is lossy, not cosmetic.** `cluster_top` was designed
assuming a few large clusters plus a tail of dust, where folding costs almost
nothing. The 982k run does not look like that:

| | Points | Share |
|---|---|---|
| noise (`-1`) | ~330,900 | 33.7% |
| `OTHER` (`-2`) | 353,267 | 36.0% |
| the 62 named clusters | ~297,800 | 30.3% |

Roughly 70% of the map renders as two flat colours. The size distribution is
*flat* rather than long-tailed — the 173 folded clusters average ~2,040 points
against ~4,800 for the kept 62 — so the 64-category cap cuts an arbitrary line
through 235 comparably-sized groups.

Three ways to live with it:

- **Colour by `kmeans` instead.** Already in the table, no noise class and no
  folding, so every point gets one of k distinguishable colours. Usually the
  more legible categorical for a first look at a million points.
- **Raise `--min-cluster-size`** until the run lands near 60 clusters, making
  `cluster_top` lossless. Roughly 4× the current value; one 5-minute run.
- **Accept it** as "the 62 largest modes, everything else greyed", stated
  explicitly.

Worth being clear that no colour encoding shows 235 categories. People
distinguish perhaps 8–12 colours reliably in a scatter, and the best categorical
palettes stretch to 20–30. If you need to inspect individual clusters at that
count, highlighting one at a time is the interaction that works, not a bigger
palette.

`cluster` itself remains available as a *continuous* colour-by, since 235 values
exceed the categorical cap. Two caveats when reading it: the ramp implies an
order that HDBSCAN's labels do not have, and `ds.mean` averages the ids — with
33.7% noise sitting at `-1`, a pixel holding ten points from cluster 200 and
five noise points renders as 133.

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

### RAPIDS — cuML needs its own environment

cuML cannot go in `pyproject.toml`, and not only because it is CUDA-only. cuDF,
which cuML imports at package load, **requires pandas < 3**; this project
requires `pandas>=3.0.1`. That is a real incompatibility, not a packaging
inconvenience, and no install method resolves it — a conda RAPIDS environment
simply *is* an environment where pandas is already 2.x.

So give cuML a separate venv. This is the recipe that works on Casper:

```bash
uv venv /glade/work/$USER/.venvs/rapids --python 3.13

RAPIDS_PY=/glade/work/$USER/.venvs/rapids/bin/python
uv pip install --python "$RAPIDS_PY" cuml-cu12 lancedb pylance numpy pandas

"$RAPIDS_PY" -c "import cuml, lancedb; print('cuml', cuml.__version__)"
```

Then run **both** stages with that interpreter. Stage 2 only needs long-stable
pandas APIs, so one environment covers the whole pipeline:

```bash
"$RAPIDS_PY" notebooks/05-latent-space-exploration/helpers/make_projection.py ...
"$RAPIDS_PY" notebooks/05-latent-space-exploration/helpers/write_projection_table.py ...
```

Three traps, all of which have bitten:

- **`uv run` re-syncs the environment from `uv.lock` on every invocation**, which
  silently undoes any `uv pip install`. Installing cuML and then running with
  `uv run` restores pandas 3 and cuDF fails on `pandas.api.types.is_interval`.
  Use the interpreter path directly.
- **`uv pip install` targets the project's `.venv`** when you are inside the
  project directory, even with another venv activated. Pass
  `--python "$RAPIDS_PY"` so there is no ambiguity.
- **Bare `python` on Casper is the NCAR base install (3.9)**, which fails on
  `from datetime import UTC`. Always the explicit path.

If cuML will not import, stage 1 refuses to run without `--allow-cpu` — a broken
RAPIDS environment fails fast instead of quietly taking hours.

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

Confirmed working on cuML 26.06.00 / A100-80GB: `knn_graph=` is accepted, and
the cuML debug log says `Calling UMAP::fit() with precomputed KNN`.

### Did it work? — the correspondence check

The one check worth running after a full run. Clusters are found in the 10-D
embedding but drawn on the 2-D map; if the two disagree, clusters come out
smeared across the map and the whole thing is untrustworthy.

```bash
"$RAPIDS_PY" -c "
import sys; sys.path.insert(0,'notebooks/05-latent-space-exploration')
import numpy as np
from helpers import PatchExperiment
exp = PatchExperiment.open('<experiments>', 'dinov3_24h')
p = exp.load_projection('umap_patch_001'); df = p.df
overall = np.hypot(df.x.std(), df.y.std())
s = np.array([np.hypot(g.x.std(), g.y.std())/overall
              for _, g in df[df.cluster>=0].groupby('cluster') if len(g)>50])
print(f'{len(s)} clusters, median spread {np.median(s):.2f}x overall, '
      f'{(s<0.5).mean():.0%} tighter than half')
"
```

Measured on the 982k run: **235 clusters, median spread 0.11× overall, 96%
tighter than half.** Spreads near 1.0× would mean the embeddings diverged.

Worth noting the exact and approximate neighbour searches agree here: the CPU
path uses pynndescent (approximate) and the CUDA path exact brute force, and
both produced compact clusters — 0.15× at 50k, 0.11× at 982k.

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
