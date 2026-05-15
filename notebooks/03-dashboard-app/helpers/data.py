"""Data-side helpers for the dashboard: DB lookups, loading, transforms, ML.

Extracted from app.py to keep the marimo notebook focused on reactive UI cells.
No marimo, matplotlib, plotly, or PIL imports here — pure data in / data out.
"""


def list_experiments(db_path: str) -> list:
    """Scan a LanceDB directory for experiment names by finding subdirs with config.lance."""
    from pathlib import Path
    p = Path(db_path)
    if not p.exists() or not p.is_dir():
        return []
    return sorted(
        d.name for d in p.iterdir()
        if d.is_dir() and (d / "config.lance").is_dir()
    )


def load_config_dict(db, config_table_name: str) -> dict:
    """Load a config table into a Python dict keyed by the 'key' column."""
    tbl = db.open_table(config_table_name)
    df = tbl.to_pandas()
    return dict(zip(df["key"], df["value"]))


def resolve_source_path(experiments_db_path: str, source_path_from_config: str) -> str:
    """Resolve source_path from config to an absolute path."""
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


def get_spatial_extent(src_img_tbl, config):
    """Read lat/lon bounds and patch grid size from table schema metadata + config."""
    import json
    lat_min, lat_max = 0.0, 10.0
    lon_min, lon_max = 0.0, 10.0
    n_rows = 14
    n_cols = 14
    try:
        raw_meta = src_img_tbl.schema.metadata or {}
        ds_info = json.loads(raw_meta.get(b"dataset_info", "{}")) if raw_meta else {}
        ext = ds_info.get("spatial_extent", {})
        if ext:
            lat_min = float(ext["lat_min"])
            lat_max = float(ext["lat_max"])
            lon_min = float(ext["lon_min"])
            lon_max = float(ext["lon_max"])
        if config.get("attention_spatial_h"):
            n_rows = int(config["attention_spatial_h"])
        if config.get("attention_spatial_w"):
            n_cols = int(config["attention_spatial_w"])
    except Exception:
        pass
    return lat_min, lat_max, lon_min, lon_max, n_rows, n_cols


def load_embedding_matrix(img_emb_tbl, n_vectors: int):
    """Load image embeddings from LanceDB, subsampled to n_vectors rows."""
    import numpy as np
    df = img_emb_tbl.to_pandas()
    X = np.asarray(df["embedding"].to_list(), dtype=np.float32)
    image_ids = df["image_id"].to_numpy()
    if n_vectors < len(X):
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), n_vectors, replace=False)
        X = X[idx]
        image_ids = image_ids[idx]
    return X, image_ids, len(df)


def fetch_metadata_for_ids(src_img_tbl, image_ids):
    """Batch-fetch non-blob columns from src_img_tbl for given image ids."""
    import pandas as pd
    import pyarrow.compute as pc

    if not len(image_ids) or src_img_tbl is None:
        return pd.DataFrame()
    # Normalize to plain Python str to avoid numpy.str_ vs str mismatch
    image_ids = [str(i) for i in image_ids]
    blob_cols = {"image_blob", "thumb_blob"}
    keep_cols = [f.name for f in src_img_tbl.schema if f.name not in blob_cols]
    # Use Lance scanner with pyarrow filter instead of .search().where()
    # (.search() is the ANN vector API and may silently drop rows)
    scanner = src_img_tbl.to_lance().scanner(
        columns=keep_cols,
        filter=pc.field("id").isin(image_ids),
    )
    df = scanner.to_table().to_pandas()
    if len(df) < len(image_ids):
        _missing = set(image_ids) - set(df["id"])
        print(f"[fetch_metadata] WARNING: {len(_missing)} of {len(image_ids)} "
              f"image IDs not found in source table")
    df = df.set_index("id").reindex(image_ids).reset_index()
    return df


def fetch_thumbnails_batch(src_img_tbl, image_ids: list, max_images: int = 20):
    """Batch-fetch image blobs, filenames, and timestamps by image id."""
    if not image_ids or src_img_tbl is None:
        return []
    image_ids = image_ids[:max_images]
    escaped = ", ".join(f"'{i}'" for i in image_ids)
    df = (
        src_img_tbl.search()
        .where(f"id IN ({escaped})")
        .select(["id", "filename", "image_blob", "dt"])
        .limit(max_images)
        .to_pandas()
    )
    return list(zip(df["filename"], df["image_blob"], df["dt"]))


def fetch_attention_maps(img_emb_tbl, image_ids: list) -> dict:
    """Batch-fetch attention maps from the image embeddings table.
    Returns {image_id: flat_float_list} for each found ID."""
    import pyarrow.compute as pc
    if not image_ids or img_emb_tbl is None:
        return {}
    df = (
        img_emb_tbl.to_lance()
        .scanner(
            columns=["image_id", "attention_map"],
            filter=pc.field("image_id").isin(image_ids),
        )
        .to_table()
        .to_pandas()
    )
    return dict(zip(df["image_id"], df["attention_map"]))


def get_thumb_dimensions(src_img_tbl, base_size=192):
    """Extract spatial extent from table metadata and compute thumb dimensions."""
    import json
    raw_meta = src_img_tbl.schema.metadata or {}
    ds_info = json.loads(raw_meta.get(b"dataset_info", "{}")) if raw_meta else {}
    ext = ds_info.get("spatial_extent", {})
    if ext:
        return compute_thumb_dimensions(ext, base_size)
    return base_size, base_size


def compute_thumb_dimensions(spatial_extent, base_size=192):
    """Compute (width_px, height_px) preserving geographic aspect ratio."""
    import math
    lat_range = abs(spatial_extent["lat_max"] - spatial_extent["lat_min"])
    lon_range = abs(spatial_extent["lon_max"] - spatial_extent["lon_min"])
    mean_lat = (spatial_extent["lat_min"] + spatial_extent["lat_max"]) / 2
    # Cosine correction: 1° longitude shrinks toward poles
    effective_lon = lon_range * math.cos(math.radians(mean_lat))
    if lat_range == 0 or effective_lon == 0:
        return base_size, base_size
    aspect = effective_lon / lat_range
    if aspect >= 1:
        return base_size, round(base_size / aspect)
    else:
        return round(base_size * aspect), base_size


def run_pca_best(X, n_components: int):
    """Run PCA with best available backend. Returns (evr, scores, backend_label)."""
    import numpy as np
    n_components = min(n_components, X.shape[0], X.shape[1])
    try:
        from cuml.decomposition import PCA
        pca = PCA(n_components=n_components, output_type="numpy")
        scores = pca.fit_transform(X)
        return np.array(pca.explained_variance_ratio_), scores, "cuML (CUDA)"
    except ImportError:
        pass
    try:
        import torch
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS not available")
        Xt = torch.tensor(X, device="mps")
        Xc = Xt - Xt.mean(dim=0)
        U, S, _ = torch.pca_lowrank(Xc, q=n_components)
        total_var = Xc.var(dim=0, unbiased=True).sum().item()
        evr = (S.cpu().numpy() ** 2 / (X.shape[0] - 1)) / total_var
        scores = (U * S).cpu().numpy()
        return evr, scores, "PyTorch (MPS / Apple Silicon)"
    except (ImportError, RuntimeError):
        pass
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components, svd_solver="randomized", random_state=0)
    scores = pca.fit_transform(X)
    return pca.explained_variance_ratio_, scores, "sklearn (CPU)"


def run_umap_best(X, n_neighbors=15, min_dist=0.1):
    """Run UMAP with best available backend. Returns (embedding_2d, backend_label)."""
    try:
        from cuml.manifold import UMAP
        reducer = UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=float(min_dist),
                       random_state=42, output_type="numpy")
        return reducer.fit_transform(X), "cuML (CUDA)"
    except ImportError:
        pass
    try:
        import umap as _umap_lib
    except ImportError:
        raise ImportError(
            "No UMAP backend found. Install umap-learn:  uv add umap-learn"
        )
    reducer = _umap_lib.UMAP(n_components=2, n_neighbors=int(n_neighbors),
                              min_dist=float(min_dist), random_state=42)
    return reducer.fit_transform(X), "umap-learn (CPU)"


def apply_brush_filter(data_cols, brush_extents):
    """Return row indices passing all brush_extents filters (AND logic)."""
    import numpy as np
    if not brush_extents:
        return None  # no brush active → show all
    n_rows = len(next(iter(data_cols.values())))
    mask = np.ones(n_rows, dtype=bool)
    for axis_name, info in brush_extents.items():
        if axis_name not in data_cols:
            continue
        _raw = data_cols[axis_name]
        _first = _raw.flat[0] if hasattr(_raw, "flat") else _raw[0]
        _is_str = isinstance(_first, str)
        vals = np.asarray(_raw, dtype=object if _is_str else float)
        if "range" in info and not _is_str:
            # Numeric / timestamp: filter by [lo, hi] range
            lo = min(info["range"])
            hi = max(info["range"])
            axis_mask = (vals >= lo) & (vals <= hi)
            if info.get("include_infnans", False):
                axis_mask |= ~np.isfinite(vals)
        elif "values" in info:
            # Categorical: filter by set membership
            if _is_str:
                _allowed = set(str(v) for v in info["values"])
                axis_mask = np.array([v in _allowed for v in vals], dtype=bool)
            else:
                allowed = [float(v) for v in info["values"]]
                axis_mask = np.isin(vals, allowed)
        else:
            continue
        mask &= axis_mask
    return np.where(mask)[0].tolist()
