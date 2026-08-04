#!/usr/bin/env python3
"""
write_projection_table.py — turn make_projection.py's output into a projection table.

Stage 2 of two, and the only half that knows what the rows mean. Stage 1 emits
coordinates for anonymous vectors; this script attaches identity, geometry and
whatever else the experiment recorded, then writes a Lance table the explorer
notebook can open.

The join is mechanical -- there is no ERA5-specific or hurricane-specific code
here. Five rules cover it:

  1. Join every *scalar* column from the source image table on id == image_id.
     `hurricane_present`, `max_wind_kts` and friends arrive because they are
     scalar columns, not because anything here knows what they mean. A
     different dataset's columns arrive the same way.
  2. Expand any timestamp column into year/month/day/hour/dayofyear, detected
     from the dtype rather than from a column named `dt`.
  3. Derive patch_row/patch_col from patch_index and the grid in config.
  4. Derive lat/lon only when the source table records a spatial extent.
  5. Classify the colour-by columns with the *viewer's own* inference, so the
     writer and the viewer cannot disagree about what is categorical.

Each rule fails soft. With no source image DB you still get a working table of
coordinates, clusters and patch geometry.

Runs in seconds, so re-running after a re-cluster is free.

Inputs
------
    --experiment  the experiment directory (config + patch_embeddings live here)
    --npz         projection.npz from make_projection.py
                  (projection_meta.json is read from the same directory)

Outputs
-------
    <experiment>/<table-name>.lance   default table name: umap_patch_001

Usage
-----
    uv run python notebooks/05-latent-space-exploration/helpers/write_projection_table.py \
        --experiment /path/to/experiments/era5/dinov3_24h \
        --npz /path/to/experiments/era5/dinov3_24h/_projection/projection.npz \
        --table-name umap_patch_001
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa

# Sibling imports rather than `from helpers import ...` on purpose. Two notebook
# directories in this repo ship a package called `helpers`, and Python caches
# one of them per process -- whichever is imported first wins, and the other's
# submodules then become unreachable. Importing the siblings directly sidesteps
# that entirely, and works the same whether this file is run as a script or
# loaded by a test. The modules used here (data, geometry) have no intra-package
# imports of their own, so nothing is lost.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import data as _data  # noqa: E402
from geometry import patch_grid  # noqa: E402
from make_projection import _as_array, fingerprint  # noqa: E402

# Source tables that must never be clobbered by a careless --table-name.
PROTECTED_TABLES = frozenset({"config", "image_embeddings", "patch_embeddings"})

# Sentinel for clusters folded out of `cluster_top`. Distinct from HDBSCAN's
# -1 noise label: "too small to give its own colour" is not the same statement
# as "density-wise, this point belongs to nothing".
OTHER_CLUSTER = -2

# Columns that carry identity or coordinates. Colouring by them says nothing
# about the data, so they are kept out of the colour-by dropdown. Mirrors
# helpers/data.py's own exclusion list, extended with this script's outputs.
NOT_COLOURABLE = ("x", "y", "image_id", "patch_index", "patch_id")

# The clustering embedding's axes are stored so a re-cluster needs no second
# UMAP run, not so they can be coloured by. There may be thirty of them, and
# they would crowd out the columns that actually describe the data.
#
# `view2`, `view3`, ... are deliberately *not* excluded: a 3-D run has only one
# or two of them and colouring the map by its own third dimension is a genuinely
# useful way to see what that component captured.
CLUSTER_AXIS_PREFIX = "z"


def log(msg: str) -> None:
    print(msg, flush=True)


# ── the five join rules ──────────────────────────────────────────────────────

def scalar_columns(schema: pa.Schema, skip=(), keep_strings: bool = False) -> list[str]:
    """Rule 1: names of the columns worth joining.

    Blobs, vectors and nested types are excluded -- `image_blob` would be
    ruinous to join across a million patch rows, and a list column cannot be a
    colour-by anyway.

    Strings are excluded too, which is less obvious. The viewer's role inference
    (`data.infer_color_roles`) only ever classifies bools and numerics, so a
    string column can *never* become a colour-by no matter its cardinality --
    joining it buys nothing and costs real space. Measured on this dataset,
    `filename` plus three storm-detail strings were 23% of the finished table,
    none of them reachable from the UI.

    `keep_strings=True` (via --keep-strings) restores them for anyone who wants
    the traceability and will pay for it.
    """
    keep = []
    for field in schema:
        if field.name in skip:
            continue
        t = field.type
        if pa.types.is_nested(t) or pa.types.is_binary(t) or pa.types.is_large_binary(t):
            continue
        if not keep_strings and (pa.types.is_string(t) or pa.types.is_large_string(t)):
            # The join key itself still has to come through.
            if field.name != "id":
                continue
        keep.append(field.name)
    return keep


def join_image_columns(df: pd.DataFrame, src_tbl, keep_strings: bool = False) -> pd.DataFrame:
    """Rule 1: attach every scalar column of the source image table.

    A left join on image_id. The image table is small (one row per frame), so
    this is a broadcast of a few thousand rows across a million.
    """
    if src_tbl is None:
        log("  rule 1: no source image table -- skipping the metadata join")
        return df

    cols = scalar_columns(src_tbl.schema, skip=("image_blob",), keep_strings=keep_strings)
    if "id" not in cols:
        log("  rule 1: source table has no `id` column -- skipping the join")
        return df

    meta = src_tbl.to_lance().to_table(columns=cols).to_pandas()
    meta = meta.rename(columns={"id": "image_id"})
    # Never let the join clobber what stage 1 produced.
    overlap = (set(meta.columns) & set(df.columns)) - {"image_id"}
    meta = meta.drop(columns=list(overlap))

    before = len(df)
    df = df.merge(meta, on="image_id", how="left", validate="many_to_one")
    assert len(df) == before, "the metadata join changed the row count"
    log(f"  rule 1: joined {len(meta):,} images, "
        f"{len(meta.columns) - 1} columns{f' (skipped {sorted(overlap)})' if overlap else ''}")
    return df


def expand_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """Rule 2: split every datetime column into its parts.

    Detected by dtype, so a dataset whose time column is called `valid_time`
    works without a code change. The parts are what make a temporal colour-by
    useful -- a raw timestamp is continuous and near-unique, which datashader
    renders as an unreadable gradient.
    """
    made = []
    for col in list(df.columns):
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            continue
        dt = pd.DatetimeIndex(df[col])
        parts = {
            f"{col}_year": dt.year.to_numpy(np.int16),
            f"{col}_month": dt.month.to_numpy(np.int8),
            f"{col}_day": dt.day.to_numpy(np.int8),
            f"{col}_hour": dt.hour.to_numpy(np.int8),
            f"{col}_dayofyear": dt.dayofyear.to_numpy(np.int16),
        }
        for name, values in parts.items():
            if name not in df.columns:
                df[name] = values
                made.append(name)
    log(f"  rule 2: expanded {len(made)} time columns" if made
        else "  rule 2: no timestamp columns found")
    return df


def add_patch_geometry(df: pd.DataFrame, config: dict) -> tuple[pd.DataFrame, tuple | None]:
    """Rule 3: patch_index -> (row, col) on the experiment's own grid.

    The grid comes from `patch_grid`, which prefers the attention grid the
    embedding run recorded and derives it from image size and patch size
    otherwise. Nothing here assumes a square grid or a particular size.
    """
    try:
        grid = patch_grid(config)
    except KeyError as exc:
        log(f"  rule 3: {exc} -- skipping patch geometry")
        return df, None

    spatial_h, spatial_w = grid
    idx = df["patch_index"].to_numpy(np.int64)
    df["patch_row"] = (idx // spatial_w).astype(np.int16)
    df["patch_col"] = (idx % spatial_w).astype(np.int16)

    out_of_range = int((idx >= spatial_h * spatial_w).sum())
    if out_of_range:
        log(f"  rule 3: WARNING {out_of_range:,} patch_index values exceed the "
            f"{spatial_h}x{spatial_w} grid -- is the config's grid correct?")
    log(f"  rule 3: patch grid {spatial_h}x{spatial_w}")
    return df, grid


def add_latlon(df: pd.DataFrame, grid, extent: dict | None) -> pd.DataFrame:
    """Rule 4: grid cell -> centre lat/lon, only when an extent is recorded.

    `get_spatial_extent` returns None rather than guessing when the source
    table has no `dataset_info` metadata, and this follows suit: no extent
    means no geographic columns, not an error.
    """
    if grid is None or extent is None:
        log("  rule 4: no spatial extent recorded -- skipping lat/lon")
        return df

    spatial_h, spatial_w = grid
    row = df["patch_row"].to_numpy(np.float64)
    col = df["patch_col"].to_numpy(np.float64)
    lat_span = extent["lat_max"] - extent["lat_min"]
    lon_span = extent["lon_max"] - extent["lon_min"]
    df["lat"] = (extent["lat_max"] - (row + 0.5) * lat_span / spatial_h).astype(np.float32)
    lon = extent["lon_min"] + (col + 0.5) * lon_span / spatial_w
    # Match helpers.geometry.format_latlon, which normalizes to -180..180.
    df["lon"] = (((lon + 180.0) % 360.0) - 180.0).astype(np.float32)

    # The orientation is an assumption, not a recorded fact -- see the docstring
    # of helpers.geometry.patch_latlon, which verified it against IBTrACS storm
    # positions for *this* dataset only.
    log(f"  rule 4: lat/lon from extent {extent['lat_min']}..{extent['lat_max']}N, "
        f"{extent['lon_min']}..{extent['lon_max']}E")
    log("          (assumes row 0 = north, col 0 = west; verified for ERA5 only)")
    return df


def colour_roles(df: pd.DataFrame, extra_categorical=()) -> dict:
    """Rule 5: classify colour-by columns using the viewer's own inference.

    `infer_color_roles` then `usable_color_columns` are exactly what
    helpers/data.py falls back to when a table carries no roles, so recording
    their output here means the writer and the viewer can never disagree.
    Constant columns are dropped for free -- an all-zero `hour` in a
    24h-subsampled experiment paints everything one colour, which reads as a
    bug.
    """
    exclude = tuple(NOT_COLOURABLE) + tuple(
        c for c in df.columns
        if c.startswith(CLUSTER_AXIS_PREFIX) and c[len(CLUSTER_AXIS_PREFIX):].isdigit()
    )
    roles = _data.infer_color_roles(df, exclude=exclude)
    for col in extra_categorical:
        if col in df.columns and col not in roles["categorical"]:
            roles["categorical"].append(col)
            if col in roles["continuous"]:
                roles["continuous"].remove(col)
    return _data.usable_color_columns(df, roles)


# ── cluster labels ───────────────────────────────────────────────────────────

def fold_rare_clusters(labels: np.ndarray, cap: int = _data.MAX_CATEGORIES) -> np.ndarray:
    """Largest clusters kept, the rest folded into OTHER.

    `cluster` needs to be a categorical colour-by, but datashader allocates one
    aggregate plane per category, so helpers/data.py caps categoricals at
    MAX_CATEGORIES and prunes anything above it. Left alone, a run that finds
    200 clusters would have `cluster` dropped from the dropdown entirely.

    Folding keeps a usable colour-by whatever min_cluster_size produces, without
    distorting the clustering to satisfy a rendering limit. The raw `cluster`
    column is still written for analysis.

    Budget: noise and OTHER each occupy one of the cap's slots.
    """
    labels = np.asarray(labels)
    real = labels[labels >= 0]
    if real.size == 0:
        return labels.copy()

    has_noise = bool((labels < 0).any())
    budget = cap - (1 if has_noise else 0)
    uniq, counts = np.unique(real, return_counts=True)
    if len(uniq) <= budget:
        return labels.copy()

    keep = uniq[np.argsort(-counts)[: budget - 1]]
    out = labels.copy()
    out[(labels >= 0) & ~np.isin(labels, keep)] = OTHER_CLUSTER
    return out


# ── assembly ─────────────────────────────────────────────────────────────────

def unpack_embeddings(df: pd.DataFrame, view: np.ndarray, z: np.ndarray) -> pd.DataFrame:
    """Spread the two embeddings into columns, widths taken from array shape.

    `x`/`y` are the viewer's contract. Any further view components become
    `view2`, `view3`, ... and are picked up as continuous colour-bys by rule 5,
    so a 3-D run yields a 2-D map colourable by its own third dimension.

    The clustering embedding becomes `z0..z{C-1}`. Its width is read from
    `z.shape[1]`, never assumed, so re-running with a different
    --n-components-cluster needs no change here.
    """
    df["x"] = view[:, 0].astype(np.float32)
    df["y"] = view[:, 1].astype(np.float32)
    for i in range(2, view.shape[1]):
        df[f"view{i}"] = view[:, i].astype(np.float32)
    for i in range(z.shape[1]):
        df[f"z{i}"] = z[:, i].astype(np.float32)
    return df


def read_identities(experiment: Path, meta: dict, row_offsets):
    """Re-read the identity columns for exactly the rows stage 1 embedded.

    Stage 1 leaves `image_id` out of the npz -- a million large strings would
    dominate an otherwise-small file -- so it is fetched here. The fingerprint
    check below is what makes that safe.
    """
    import lancedb

    db = lancedb.connect(str(experiment))
    tbl = db.open_table(meta["source_table"])
    dataset = tbl.to_lance()

    id_column = meta["id_column"]
    wanted = [c for c in dict.fromkeys([id_column, "image_id", "patch_index"])
              if c in tbl.schema.names]
    if row_offsets is None:
        arrow = dataset.to_table(columns=wanted)
    else:
        arrow = dataset.take(row_offsets, columns=wanted)

    # Order-sensitive: catches a table compacted or rewritten between stages,
    # which would otherwise join every coordinate to the wrong patch.
    got = fingerprint(_as_array(arrow.column(id_column)))
    if got != meta["id_fingerprint"]:
        raise SystemExit(
            f"error: {meta['source_table']!r} no longer matches the projection.\n"
            f"       id fingerprint {got[:16]}... != {meta['id_fingerprint'][:16]}...\n"
            "       The table changed since make_projection.py ran. Re-run stage 1."
        )
    return arrow.to_pandas()


def build_frame(experiment: Path, npz, meta: dict,
                keep_strings: bool = False) -> tuple[pd.DataFrame, dict, dict]:
    """Everything from coordinates to a finished DataFrame."""
    row_offsets = npz["row_offsets"] if "row_offsets" in npz.files else None
    df = read_identities(experiment, meta, row_offsets)

    view, z = npz["view"], npz["z"]
    if len(df) != len(view):
        raise SystemExit(
            f"error: read {len(df):,} identity rows but the projection has {len(view):,}."
        )

    df = unpack_embeddings(df, view, z)
    df["cluster"] = np.asarray(npz["cluster"], dtype=np.int32)
    df["cluster_prob"] = np.asarray(npz["cluster_prob"], dtype=np.float32)
    df["cluster_top"] = fold_rare_clusters(df["cluster"].to_numpy()).astype(np.int32)
    df["kmeans"] = np.asarray(npz["kmeans"], dtype=np.int16)

    n_folded = int((df["cluster_top"] == OTHER_CLUSTER).sum())
    if n_folded:
        log(f"  cluster_top: folded {n_folded:,} points from rare clusters into "
            f"OTHER ({OTHER_CLUSTER})")

    import lancedb

    config = _data.load_config_dict(lancedb.connect(str(experiment)))
    src_tbl = _data.open_source_table(str(experiment), config)

    log("applying join rules:")
    df = join_image_columns(df, src_tbl, keep_strings=keep_strings)
    df = expand_timestamps(df)
    df, grid = add_patch_geometry(df, config)
    df = add_latlon(df, grid, _data.get_spatial_extent(src_tbl))

    # The id column was needed to verify the read lined up with stage 1; it is
    # not table content. `image_id` + `patch_index` is an equivalent key, and
    # `patch_id` is a large_string that would cost ~18% of the finished table.
    id_column = meta["id_column"]
    if id_column not in ("image_id", "patch_index") and id_column in df.columns:
        df = df.drop(columns=[id_column])

    return df, config, {"grid": grid}


def apply_extra_columns(df, module_path: str, config: dict, experiment: Path):
    """Escape hatch: a user module exposing enrich(df, config, experiment) -> df.

    Plain Python rather than a join-spec config format. The five rules cover the
    general case; anything genuinely bespoke is easier to write as six lines of
    pandas than to express in a DSL.
    """
    import importlib.util

    path = Path(module_path).expanduser().resolve()
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "enrich"):
        raise SystemExit(f"error: {path} defines no enrich(df, config, experiment)")
    out = module.enrich(df, config, experiment)
    log(f"  extra: {path.name} added {len(out.columns) - len(df.columns)} columns")
    return out


# ── main ─────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Write a projection table from make_projection.py output.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--experiment", required=True,
                   help="Experiment directory holding config and the embedding tables.")
    p.add_argument("--npz", default=None,
                   help="projection.npz. Defaults to <experiment>/_projection/projection.npz.")
    p.add_argument("--table-name", default="umap_patch_001",
                   help="Must start with umap_ to appear in the notebook's dropdown.")
    p.add_argument("--extra-columns", default=None,
                   help="Path to a module exposing enrich(df, config, experiment).")
    p.add_argument("--keep-strings", action="store_true",
                   help="Join string columns too. They can never be colour-bys, so "
                        "this is for traceability only and costs real space.")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    experiment = Path(args.experiment).expanduser()
    if not experiment.is_dir():
        print(f"error: no such experiment directory: {experiment}", file=sys.stderr)
        return 2

    npz_path = Path(args.npz).expanduser() if args.npz else experiment / "_projection" / "projection.npz"
    if not npz_path.exists():
        print(f"error: no projection at {npz_path}; run make_projection.py first",
              file=sys.stderr)
        return 2
    meta_path = npz_path.parent / "projection_meta.json"
    if not meta_path.exists():
        print(f"error: {meta_path} is missing; it is written beside the npz",
              file=sys.stderr)
        return 2

    table = args.table_name
    if table in PROTECTED_TABLES:
        print(f"error: refusing to overwrite source table {table!r}", file=sys.stderr)
        return 2
    if not table.startswith("umap_"):
        log(f"note: {table!r} does not start with `umap_`, so the notebook's "
            "projection dropdown will not list it.")

    meta = json.loads(meta_path.read_text())
    npz = np.load(npz_path, allow_pickle=False)

    log("=" * 68)
    log(f"experiment   {experiment}")
    log(f"projection   {npz_path}")
    log(f"source       {meta['source_table']} ({meta['n_rows']:,} rows, "
        f"{meta['embedding_dim']}-d, backend={meta['backend']})")
    log(f"view / z     {npz['view'].shape[1]}-d / {npz['z'].shape[1]}-d")
    log(f"clusters     {meta['n_clusters']} + noise ({meta['noise_fraction']:.1%})")
    if not meta.get("shared_knn_graph", True):
        log("WARNING      stage 1 fell back to independent kNN graphs; clusters "
            "may not align with the view")
    log(f"table        {table}")
    log("=" * 68)

    df, config, _ = build_frame(experiment, npz, meta, keep_strings=args.keep_strings)

    if args.extra_columns:
        df = apply_extra_columns(df, args.extra_columns, config, experiment)

    roles = colour_roles(df, extra_categorical=("cluster_top", "kmeans"))
    log(f"  rule 5: {len(roles['categorical'])} categorical, "
        f"{len(roles['continuous'])} continuous colour-by columns")

    # Identity and coordinates first so the table reads sensibly in a dump.
    lead = [c for c in ("image_id", "patch_index", "x", "y", "cluster",
                        "cluster_prob", "cluster_top", "kmeans") if c in df.columns]
    df = df[lead + [c for c in df.columns if c not in lead]]

    table_meta = {k: str(v) for k, v in meta.items()}
    table_meta["color_by"] = json.dumps(roles)
    table_meta["written_by"] = Path(__file__).name

    arrow = pa.Table.from_pandas(df, preserve_index=False)
    arrow = arrow.replace_schema_metadata(table_meta)

    import lancedb

    # Rebuilt from scratch: this is a derived artifact, not a store.
    shutil.rmtree(experiment / f"{table}.lance", ignore_errors=True)
    lancedb.connect(str(experiment)).create_table(table, arrow)

    size_mb = sum(
        f.stat().st_size
        for f in (experiment / f"{table}.lance").rglob("*")
        if f.is_file()
    ) / 1e6
    log(f"\nwrote {experiment / (table + '.lance')}")
    log(f"  {len(df):,} rows x {len(df.columns)} cols, {size_mb:.1f} MB")
    log(f"  open {experiment.name} -> {table} in latent_exploration.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
