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
