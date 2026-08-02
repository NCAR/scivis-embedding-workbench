"""Database access for the patch-embedding exploration notebook.

Everything that talks to LanceDB lives here: experiments, the patch-embedding
matrix, and the raw source images those patches were cut from.

Pure data in / data out — no marimo, plotting, or UI imports.
"""


def list_experiments(db_path: str) -> list:
    """Scan a LanceDB directory for experiment names (subdirs with config.lance)."""
    from pathlib import Path
    p = Path(db_path)
    if not p.exists() or not p.is_dir():
        return []
    return sorted(
        d.name for d in p.iterdir()
        if d.is_dir() and (d / "config.lance").is_dir()
    )


def load_config_dict(db, config_table_name: str = "config") -> dict:
    """Load a key/value config table into a Python dict."""
    df = db.open_table(config_table_name).to_pandas()
    return dict(zip(df["key"], df["value"]))


def open_experiment(db_path: str, exp_name: str):
    """Open one experiment. Returns (config_dict, patch_emb_tbl)."""
    import lancedb
    from pathlib import Path
    db = lancedb.connect(str(Path(db_path) / exp_name))
    return load_config_dict(db), db.open_table("patch_embeddings")


def load_patch_matrix(patch_emb_tbl, limit: int = None, random_sample: bool = True,
                      seed: int = 42):
    """Materialize patch embeddings as a numpy matrix.

    Returns (X, image_ids, patch_indices) where:
      X              : (N, D) float32 array of L2-normalized DINO patch vectors
      image_ids      : (N,)   object array of source image ids
      patch_indices  : (N,)   int32  array of patch positions within each image

    limit         : max rows to load (None loads the whole table).
    random_sample : draw a uniform sample across the table. Rows are stored in
                    ingest order, so taking the first `limit` rows biases the
                    sample toward whatever was ingested first (usually the
                    earliest timestamps and only a handful of source images).
    """
    import numpy as np
    if limit is None:
        df = patch_emb_tbl.to_pandas()
    elif random_sample:
        # Lance take() fetches just the requested row offsets, so an unbiased
        # sample costs no more than a head() of the same size.
        ds = patch_emb_tbl.to_lance()
        n_rows = ds.count_rows()
        if limit >= n_rows:
            df = patch_emb_tbl.to_pandas()
        else:
            rng = np.random.default_rng(seed)
            offsets = np.sort(rng.choice(n_rows, limit, replace=False))
            df = ds.take(
                offsets, columns=["image_id", "patch_index", "embedding"]
            ).to_pandas()
    else:
        # head() pushes the limit down into LanceDB. to_pandas().head(limit)
        # would read every row first and then throw almost all of them away.
        df = patch_emb_tbl.head(limit).to_pandas()
    X = np.asarray(df["embedding"].to_list(), dtype=np.float32)
    return X, df["image_id"].to_numpy(), df["patch_index"].to_numpy(dtype=np.int32)


def list_projection_tables(db_path: str, exp_name: str, prefix: str = "umap_") -> list:
    """Names of projection tables inside one experiment, e.g. ["umap_patch_001"].

    A filesystem scan rather than `db.list_tables()`, mirroring
    `list_experiments`: this runs every time the experiment selection changes,
    and a scan costs nothing and needs no open connection.
    """
    from pathlib import Path
    p = Path(db_path) / exp_name
    if not p.is_dir():
        return []
    return sorted(
        d.name[: -len(".lance")]
        for d in p.iterdir()
        if d.is_dir() and d.name.startswith(prefix) and d.name.endswith(".lance")
    )


def open_projection_table(db_path: str, exp_name: str, table_name: str):
    """Open one projection table from an experiment."""
    import lancedb
    from pathlib import Path
    db = lancedb.connect(str(Path(db_path) / exp_name))
    return db.open_table(table_name)


def get_color_roles(tbl):
    """Read the `color_by` roles from a projection table's schema metadata.

    The writer records which columns are meaningfully categorical and which are
    continuous, so the viewer's colour-by dropdown is built from the data rather
    than a hardcoded list. Returns None when absent -- same contract as
    `get_spatial_extent`, so callers can fall back to inferring roles.
    """
    import json

    if tbl is None:
        return None
    raw = (tbl.schema.metadata or {}).get(b"color_by")
    if not raw:
        return None
    roles = json.loads(raw) or {}
    names = set(tbl.schema.names)
    return {
        "categorical": [c for c in roles.get("categorical", []) if c in names],
        "continuous": [c for c in roles.get("continuous", []) if c in names],
    }


def get_table_metadata(tbl) -> dict:
    """Schema metadata as a plain str->str dict. Empty when there is none."""
    return {
        k.decode(): v.decode() for k, v in (tbl.schema.metadata or {}).items()
    }


# Never useful to colour by, and expensive to carry: `image_id` is a
# large_string repeated across ~1M rows.
_PROJECTION_SKIP_COLUMNS = ("image_id", "embedding")


def load_projection_frame(tbl, columns=None):
    """Materialize a whole projection table as a DataFrame.

    Deliberately has no `limit`: projections are a couple of dozen narrow
    columns, so the full table is tens of MB and reads in well under a second --
    and subsampling would defeat the point of datashading every point. The
    column subset is the same trick `load_patch_matrix` uses for rows, applied
    to width instead.
    """
    cols = columns or [
        n for n in tbl.schema.names if n not in _PROJECTION_SKIP_COLUMNS
    ]
    return tbl.to_lance().to_table(columns=cols).to_pandas()


# One aggregate plane is allocated per category by datashader's count_cat, so a
# high-cardinality column would try to build thousands of them.
MAX_CATEGORIES = 64

# Identity and coordinates: colouring by these says nothing about the data.
_NOT_COLOURABLE = ("x", "y", "image_id", "patch_index", "dt")


def infer_color_roles(df, exclude=_NOT_COLOURABLE, max_categories: int = MAX_CATEGORIES):
    """Guess categorical/continuous roles from dtypes, when metadata is absent.

    Bools and low-cardinality integers are categorical; anything else numeric is
    continuous; everything else is dropped.
    """
    import pandas as pd

    categorical, continuous = [], []
    for col in df.columns:
        if col in exclude:
            continue
        s = df[col]
        if pd.api.types.is_bool_dtype(s):
            categorical.append(col)
        elif pd.api.types.is_integer_dtype(s):
            (categorical if s.nunique() <= max_categories else continuous).append(col)
        elif pd.api.types.is_numeric_dtype(s):
            continuous.append(col)
    return {"categorical": categorical, "continuous": continuous}


def usable_color_columns(df, roles, max_categories: int = MAX_CATEGORIES) -> dict:
    """Drop colour-by columns that would render as a flat, meaningless picture.

    Two cases, both data-dependent rather than hardcoded so this self-corrects
    on a different experiment:

      * constant columns -- `hour` is all zeros in a 24h-subsampled experiment,
        and a single-valued column paints everything one colour, which reads as
        a bug rather than as information.
      * categoricals above `max_categories` -- see MAX_CATEGORIES. Applied to
        metadata-supplied roles too, not just inferred ones, so a mistake in the
        writer cannot blow up the renderer.
    """
    def keep(col, cap=None):
        if col not in df.columns:
            return False
        n = df[col].nunique(dropna=True)
        return n > 1 and (cap is None or n <= cap)

    return {
        "categorical": [c for c in roles.get("categorical", []) if keep(c, max_categories)],
        "continuous": [c for c in roles.get("continuous", []) if keep(c)],
    }


def representative_offsets(df, n: int = 24, x: str = "x", y: str = "y",
                           percentile: float = 1.0, min_fraction: float = 0.01):
    """Row offsets of ~`n` points spread evenly across the populated area.

    Bins the plane into a grid and takes, from each qualifying cell, the point
    closest to that cell's centre. Coverage rather than density: a random sample
    of the same size clumps wherever the data is dense and leaves the sparse
    arms of a projection unrepresented.

    Note the grid decides *which* points are picked, not where they are drawn --
    each stays at its own coordinates, so the result is only as regular as the
    data underneath it.

    Three details, each fixing a way the layout otherwise comes out gappy:

    `percentile` clips the grid to the p..100-p range instead of min..max. A
    projection's outliers stretch its bounding box far past the bulk -- measured
    at one point here, the middle 98% of the data covered 39% of the box width
    and 30% of its height -- so a grid over the full extent spends most of its
    cells on empty space and squeezes the interesting core into a handful.

    `min_fraction` drops cells holding less than that fraction of the median
    occupied cell's count. Without it a cell containing three stray noise points
    is as eligible as one containing a hundred thousand, and tiles land out in
    the void.

    The surplus is then thinned *evenly* in raster order rather than at random.
    The grid is deliberately finer than `n`, so there are usually more
    qualifying cells than tiles wanted; dropping them at random punched holes
    in otherwise-even coverage, which was the single biggest source of gaps.
    """
    import numpy as np

    n = int(n)
    if n <= 0 or len(df) == 0:
        return np.empty(0, dtype=int)

    xs_all = df[x].to_numpy(dtype="float64")
    ys_all = df[y].to_numpy(dtype="float64")

    if percentile and percentile > 0:
        x0, x1 = np.nanpercentile(xs_all, [percentile, 100 - percentile])
        y0, y1 = np.nanpercentile(ys_all, [percentile, 100 - percentile])
    else:
        x0, x1 = np.nanmin(xs_all), np.nanmax(xs_all)
        y0, y1 = np.nanmin(ys_all), np.nanmax(ys_all)

    inside = (xs_all >= x0) & (xs_all <= x1) & (ys_all >= y0) & (ys_all <= y1)
    if not inside.any():
        return np.empty(0, dtype=int)

    # Offsets into the original frame, so callers can index straight into it.
    where = np.flatnonzero(inside)
    xs, ys = xs_all[where], ys_all[where]

    g = max(1, int(np.ceil(np.sqrt(n * 2.0))))
    xw = (x1 - x0) or 1.0
    yw = (y1 - y0) or 1.0

    col = np.clip(((xs - x0) / xw * g).astype(int), 0, g - 1)
    row = np.clip(((ys - y0) / yw * g).astype(int), 0, g - 1)
    cell = row * g + col

    counts = np.bincount(cell, minlength=g * g)
    occupied = counts[counts > 0]
    floor = max(1.0, float(np.median(occupied)) * min_fraction)
    qualifies = counts >= floor
    keep = qualifies[cell]
    if not keep.any():
        keep = np.ones(len(cell), dtype=bool)

    # Distance to the cell's centre, so the chosen point sits where the tile
    # will be drawn rather than at the cell's edge.
    cx = x0 + (col + 0.5) * xw / g
    cy = y0 + (row + 0.5) * yw / g
    d2 = np.where(keep, (xs - cx) ** 2 + (ys - cy) ** 2, np.inf)

    order = np.lexsort((d2, cell))
    ordered_cells = cell[order]
    first = np.ones(len(order), dtype=bool)
    first[1:] = ordered_cells[1:] != ordered_cells[:-1]
    first &= np.isfinite(d2[order])
    picked = order[first]

    if len(picked) > n:
        # Evenly spaced through the raster-ordered cells: keeps the survivors
        # spread instead of leaving random holes.
        picked = picked[np.linspace(0, len(picked) - 1, n).round().astype(int)]
    return where[picked]


def resolve_source_path(experiments_db_path: str, source_path_from_config: str):
    """Resolve the config's source_path to an absolute path, or None.

    Walks up from the experiments DB until the relative source path resolves,
    so the same config works from different mount points.
    """
    from pathlib import Path
    p = Path(source_path_from_config)
    if p.is_absolute():
        return str(p) if p.exists() else None
    candidate = Path(experiments_db_path)
    for _ in range(10):
        candidate = candidate.parent
        resolved = candidate / source_path_from_config
        if resolved.exists():
            return str(resolved)
    return None


def open_source_table(experiments_db_path: str, config: dict):
    """Open the raw image table this experiment was built from. None if absent."""
    import lancedb
    src_path = resolve_source_path(experiments_db_path, config.get("source_path", ""))
    if src_path is None:
        return None
    db = lancedb.connect(src_path)
    return db.open_table(config.get("raw_table", "images"))


def fetch_image_blobs(src_img_tbl, image_ids, extra_cols=None):
    """Batch-fetch {image_id: row_dict} for the given ids.

    Uses the Lance scanner with a pyarrow isin filter rather than
    .search().where(): .search() is the ANN vector API and can silently drop
    rows. One scan serves every patch that shares a parent image.
    """
    import pyarrow.compute as pc

    if src_img_tbl is None or not len(image_ids):
        return {}
    wanted = list(dict.fromkeys(image_ids))  # de-dupe, keep order
    cols = ["id", "image_blob"] + list(extra_cols or [])
    cols = [c for c in dict.fromkeys(cols) if c in src_img_tbl.schema.names]
    table = (
        src_img_tbl.to_lance()
        .scanner(columns=cols, filter=pc.field("id").isin(wanted))
        .to_table()
        .to_pandas()
    )
    return {row["id"]: row for _, row in table.iterrows()}


def get_spatial_extent(src_img_tbl):
    """Read lat/lon bounds from the source table's schema metadata, or None.

    The experiment config does not carry the extent -- only the raw image
    table's `dataset_info` blob does. Returns None rather than guessing when
    the metadata is absent, so callers can fall back to grid coordinates.
    """
    import json

    if src_img_tbl is None:
        return None
    raw = (src_img_tbl.schema.metadata or {}).get(b"dataset_info")
    if not raw:
        return None
    extent = (json.loads(raw) or {}).get("spatial_extent") or {}
    needed = ("lat_min", "lat_max", "lon_min", "lon_max")
    return extent if all(k in extent for k in needed) else None
