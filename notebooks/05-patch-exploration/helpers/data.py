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


def load_patch_matrix(patch_emb_tbl, limit: int = None):
    """Materialize patch embeddings as a numpy matrix.

    Returns (X, image_ids, patch_indices) where:
      X              : (N, D) float32 array of L2-normalized DINO patch vectors
      image_ids      : (N,)   object array of source image ids
      patch_indices  : (N,)   int32  array of patch positions within each image
    """
    import numpy as np
    df = patch_emb_tbl.to_pandas() if limit is None else patch_emb_tbl.to_pandas().head(limit)
    X = np.asarray(df["embedding"].to_list(), dtype=np.float32)
    return X, df["image_id"].to_numpy(), df["patch_index"].to_numpy(dtype=np.int32)
