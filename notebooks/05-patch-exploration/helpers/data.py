"""Minimal data helpers for the patch-embedding exploration notebook.

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
