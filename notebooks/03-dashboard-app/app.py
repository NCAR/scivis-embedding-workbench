import marimo

__generated_with = "0.23.4"
app = marimo.App(layout_file="layouts/app.grid.json")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import IPython  # must be imported before lancedb to avoid circular import via tqdm→ipywidgets
    import lancedb
    import pandas as pd
    import polars as pl
    import matplotlib.pyplot as plt
    from wigglystuff import ParallelCoordinates

    return ParallelCoordinates, lancedb, mo, np, pd, pl, plt


@app.function
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


@app.function
def get_theme_colors(theme: str) -> dict:
    """Centralized color palette for light/dark theme. Used by all plot functions.
    Values are identical to those previously hardcoded in each plot function.
    Edit here to change colors app-wide.
    """
    is_dark = (theme == "dark")
    return {
        # Shared text + border
        "text":            "#e0e0e0" if is_dark else "#222222",
        "border":          "#444444" if is_dark else "#cccccc",
        # Map (make_extent_map)
        "bg":              "#1a1a1a" if is_dark else "#ffffff",
        "ocean":           "#1e3a5f" if is_dark else "#a8c8e8",
        "land":            "#3a3a3a" if is_dark else "#d4d4d4",
        "coast":           "#aaaaaa" if is_dark else "#555555",
        "grid":            "#666666" if is_dark else "#888888",
        # Thumbnail gallery (render_thumbnail_gallery)
        "gallery_bg":      "rgba(30,30,30,0.85)" if is_dark else "#ffffff",
        "gallery_bg_rgb":  (30, 30, 30) if is_dark else (255, 255, 255),
        # Scree plot (make_scree_plot)
        "bar_color":       "#4FC3F7" if is_dark else "#1565C0",
        "line_color":      "#FF7043" if is_dark else "#C62828",
        "plotly_template": "plotly_dark" if is_dark else "plotly_white",
    }


@app.function
def load_config_dict(db, config_table_name: str) -> dict:
    """Load a config table into a Python dict keyed by the 'key' column."""
    tbl = db.open_table(config_table_name)
    df = tbl.to_pandas()
    return dict(zip(df["key"], df["value"]))


@app.function
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


@app.function
def make_extent_map(lat_min, lat_max, lon_min, lon_max, spatial_h, spatial_w, patch_size=16, theme="light", experiment=""):
    """Cartopy map cropped to spatial extent with patch grid line overlay."""
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    colors = get_theme_colors(theme)
    proj = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(8, 5), subplot_kw={"projection": proj})
    fig.patch.set_facecolor(colors["bg"])
    ax.set_facecolor(colors["bg"])
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
    ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor=colors["ocean"], zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("110m"),  facecolor=colors["land"],  zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor=colors["coast"], linewidth=0.8, zorder=2)
    gl = ax.gridlines(crs=proj, draw_labels=True, linewidth=0)
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"color": colors["text"], "fontsize": 8}
    gl.ylabel_style = {"color": colors["text"], "fontsize": 8}
    n_rows, n_cols = spatial_h, spatial_w
    img_h = n_rows * patch_size
    img_w = n_cols * patch_size
    lat_step = (lat_max - lat_min) / n_rows
    lon_step = (lon_max - lon_min) / n_cols
    for i in range(1, n_rows):
        ax.plot([lon_min, lon_max], [lat_min + i * lat_step] * 2,
                transform=proj, color=colors["grid"], linewidth=0.4, zorder=3)
    for j in range(1, n_cols):
        ax.plot([lon_min + j * lon_step] * 2, [lat_min, lat_max],
                transform=proj, color=colors["grid"], linewidth=0.4, zorder=3)
    _title = f"{img_w}×{img_h}px  |  {n_rows}×{n_cols} patch grid ({n_rows * n_cols} patches)"
    if experiment:
        _title = f"{experiment}  —  {_title}"
    ax.set_title(_title, color=colors["text"], fontsize=10)
    fig.tight_layout()
    return fig


@app.function
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


@app.function
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


@app.function
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


@app.function
def make_scree_plot(evr, n_total: int, emb_dim: int, n_used: int, backend: str, theme: str = "light"):
    """Render an interactive Plotly scree plot with per-component and cumulative variance."""
    import numpy as np
    import plotly.graph_objects as go

    _c = get_theme_colors(theme)
    plotly_template = _c["plotly_template"]
    bar_color       = _c["bar_color"]
    line_color      = _c["line_color"]

    cum = np.cumsum(evr) * 100
    per_comp = evr * 100
    components = np.arange(1, len(evr) + 1)
    _sample_label = f"{n_used:,} / {n_total:,}" if n_used < n_total else f"{n_total:,}"

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=components,
        y=per_comp,
        name="Per component",
        marker=dict(color=bar_color, line=dict(color=bar_color, width=0.5)),
        hovertemplate="PC %{x}<br>Variance: %{y:.2f}%<extra></extra>",
    ))

    fig.add_trace(go.Scatter(
        x=components,
        y=cum,
        name="Cumulative",
        mode="lines+markers",
        line=dict(color=line_color, width=2.5),
        marker=dict(size=5),
        yaxis="y2",
        hovertemplate="PC %{x}<br>Cumulative: %{y:.1f}%<extra></extra>",
    ))

    fig.update_layout(
        template=plotly_template,
        title=dict(
            text=f"PCA Scree — {_sample_label} images × dim {emb_dim}  [{backend}]",
            font=dict(size=14),
        ),
        xaxis=dict(title="Principal component", showgrid=False),
        yaxis=dict(title="Explained variance (%)", showgrid=True, gridwidth=0.5),
        yaxis2=dict(
            title="Cumulative variance (%)",
            overlaying="y",
            side="right",
            range=[0, 105],
            showgrid=False,
        ),
        height=320,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=40, l=60, r=60),
        hovermode="x unified",
    )
    return fig


@app.cell
def _(mo):
    embedding_db_path = mo.ui.text(
        value="",
        placeholder="e.g. /data/lancedb/experiments/era5",
        label="Experiments DB path",
        full_width=True,
    )
    map_theme = mo.ui.switch(label="Dark mode")
    return embedding_db_path, map_theme


@app.cell
def _(embedding_db_path, mo):
    _experiments = list_experiments(embedding_db_path.value)
    if _experiments:
        experiment_selector = mo.ui.dropdown(
            options=_experiments,
            value=_experiments[0],
            label="Experiment",
        )
    else:
        experiment_selector = mo.ui.dropdown(options=[], label="Experiment")
    return (experiment_selector,)


@app.cell
def _(embedding_db_path, experiment_selector, lancedb):
    if not embedding_db_path.value or experiment_selector.value is None:
        db = config = img_emb_tbl = patch_emb_tbl = src_img_tbl = None
    else:
        from pathlib import Path as _Path
        _exp_db_path = str(_Path(embedding_db_path.value) / experiment_selector.value)
        db = lancedb.connect(_exp_db_path)
        config = load_config_dict(db, "config")
        _source_path = resolve_source_path(embedding_db_path.value, config.get("source_path", ""))
        if _source_path:
            source_db = lancedb.connect(_source_path)
            src_img_tbl = source_db.open_table(config["source"])
        else:
            src_img_tbl = None
        img_emb_tbl = db.open_table("image_embeddings")
        patch_emb_tbl = db.open_table("patch_embeddings")
    return config, img_emb_tbl, patch_emb_tbl, src_img_tbl


@app.cell
def _(mo, pd):
    overview_tab = mo.vstack([
        mo.md("# SciVis Embedding Workbench"),
        mo.callout(
            mo.md(
                "A research toolkit for generating, exploring, and analyzing "
                "patch-level embeddings from scientific imagery (ERA5 weather composites)."
            ),
            kind="info",
        ),
        mo.md("## Notebooks"),
        mo.ui.table(
            pd.DataFrame([
                {"Section": "Data",       "Notebook": "📦 Prepare Data",    "Description": "Ingest raw JPEG images into a LanceDB table with metadata"},
                {"Section": "Embeddings", "Notebook": "⚙️ Generate",         "Description": "Configure and run DINOv3 / OpenCLIP embedding experiments"},
                {"Section": "Embeddings", "Notebook": "🔍 Explore",          "Description": "Interactive experiment explorer with patch similarity search"},
                {"Section": "Embeddings", "Notebook": "🗺️ Spatial Analysis", "Description": "Spatial extent map with patch grid and experiment metadata"},
                {"Section": "Embeddings", "Notebook": "📊 UMAP / PCA",       "Description": "GPU-accelerated dimensionality reduction and 2D visualization"},
            ]),
            selection=None,
        ),
    ])
    return


@app.cell
def _(
    config,
    embedding_db_path,
    experiment_selector,
    img_emb_tbl,
    map_theme,
    mo,
    patch_emb_tbl,
    pd,
    plt,
    src_img_tbl,
):
    import json

    if config is None:
        explore_tab = mo.vstack([
            embedding_db_path,
            experiment_selector,
            mo.callout(mo.md("Enter a DB path and select an experiment to explore."), kind="info"),
        ])
    else:
        def schema_df(tbl):
            return pd.DataFrame([{"Column": f.name, "Type": str(f.type)} for f in tbl.schema])

        def index_info(tbl):
            indices = list(tbl.list_indices())
            if not indices:
                return mo.md("*(no indexes)*")
            rows = []
            for idx in indices:
                stats = tbl.index_stats(idx.name)
                row = {
                    "Name": idx.name,
                    "Type": getattr(idx, "index_type", getattr(idx, "type", "?")),
                    "Variables": ", ".join(idx.columns),
                }
                if stats is not None:
                    row["Indexed rows"] = getattr(stats, "num_indexed_rows", "—")
                    row["Unindexed rows"] = getattr(stats, "num_unindexed_rows", "—")
                rows.append(row)
            return mo.ui.table(pd.DataFrame(rows), selection=None)

        def flatten(d, prefix=""):
            rows = []
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else k
                if isinstance(v, dict):
                    rows.extend(flatten(v, key))
                else:
                    rows.append({"Key": key, "Value": str(v)})
            return rows

        # Source sub-tab
        _raw_meta = src_img_tbl.schema.metadata or {} if src_img_tbl else {}
        _dataset_info = json.loads(_raw_meta.get(b"dataset_info", "{}")) if _raw_meta else {}
        _src_subtab = mo.ui.tabs({
            "Metadata": mo.ui.table(pd.DataFrame(flatten(_dataset_info)), selection=None)
            if _dataset_info else mo.md("*(no schema metadata)*"),
            "Variables": mo.ui.table(schema_df(src_img_tbl), selection=None) if src_img_tbl else mo.md("*(unavailable)*"),
        })

        # Experiment sub-tab
        _img_n = img_emb_tbl.count_rows()
        _patch_n = patch_emb_tbl.count_rows()
        _cfg_df = pd.DataFrame(list(config.items()), columns=["Key", "Value"])
        _exp_subtab = mo.ui.tabs({
            "Config": mo.ui.table(_cfg_df, selection=None),
            f"Image Embeddings ({_img_n:,})": mo.vstack([
                mo.md("### Schema"), mo.ui.table(schema_df(img_emb_tbl), selection=None),
                mo.md("### Indexes"), index_info(img_emb_tbl),
            ]),
            f"Patch Embeddings ({_patch_n:,})": mo.vstack([
                mo.md("### Schema"), mo.ui.table(schema_df(patch_emb_tbl), selection=None),
                mo.md("### Indexes"), index_info(patch_emb_tbl),
            ]),
        })

        # Map sub-tab
        _ext = _dataset_info.get("spatial_extent", {})
        if _ext and "attention_spatial_h" in config and "attention_spatial_w" in config:
            _map_fig = make_extent_map(
                lat_min=float(_ext["lat_min"]), lat_max=float(_ext["lat_max"]),
                lon_min=float(_ext["lon_min"]), lon_max=float(_ext["lon_max"]),
                spatial_h=int(config["attention_spatial_h"]),
                spatial_w=int(config["attention_spatial_w"]),
                patch_size=int(config.get("patch_size", 16)),
                theme="dark" if map_theme.value else "light",
                experiment=experiment_selector.value,
            )
            _map_subtab = mo.as_html(_map_fig)
            plt.close(_map_fig)
        else:
            _map_subtab = mo.callout(mo.md("Spatial extent or `attention_spatial_h`/`attention_spatial_w` not available."), kind="warn")

        _data_tabs = mo.ui.tabs({"Source": _src_subtab, "Experiment": _exp_subtab})
        _t = f'<div style="flex:3 3 0;min-width:0;overflow:auto;">{_data_tabs.text}</div>'
        _m = f'<div style="flex:2 2 0;min-width:0;overflow:auto;">{_map_subtab.text}</div>'
        explore_tab = mo.vstack([
            mo.hstack([embedding_db_path, experiment_selector, map_theme], justify="start"),
            mo.Html(f'<div style="display:flex;align-items:flex-start;gap:8px;">{_t}{_m}</div>'),
        ])
    return (explore_tab,)


@app.cell
def _(mo):
    get_pca, set_pca = mo.state(None)
    return get_pca, set_pca


@app.cell
def _(mo):
    n_vectors = mo.ui.number(value=5000, start=2, label="Max vectors")
    run_pca = mo.ui.run_button(
        label="▶ Run PCA",
        kind="success",
    )
    pca_controls_ui = mo.hstack([n_vectors, run_pca], justify="start", align="end")
    return n_vectors, pca_controls_ui, run_pca


@app.cell
def _(
    experiment_selector,
    get_pca,
    img_emb_tbl,
    mo,
    n_vectors,
    run_pca,
    set_pca,
    src_img_tbl,
):
    _exp = experiment_selector.value if experiment_selector.value else None
    _current = get_pca()
    if _current is not None and _current.get("experiment") != _exp:
        set_pca(None)
    if not run_pca.value:
        pca_status = mo.callout(mo.md("Configure options above and click Run PCA."), kind="info")
    elif img_emb_tbl is None:
        pca_status = mo.callout(mo.md("No experiment loaded — enter a DB path above."), kind="warn")
    else:
        _X, _image_ids, _n_total = load_embedding_matrix(img_emb_tbl, n_vectors.value)
        _evr, _scores, _backend = run_pca_best(_X, _X.shape[1])
        _meta_df = fetch_metadata_for_ids(src_img_tbl, list(_image_ids))
        set_pca({
            "evr": _evr, "scores": _scores, "image_ids": _image_ids,
            "metadata": _meta_df,
            "backend": _backend,
            "n_total": _n_total, "emb_dim": _X.shape[1], "n_used": len(_X),
            "experiment": _exp,
        })
        pca_status = None
    return (pca_status,)


@app.cell
def _(get_pca, map_theme, mo):
    _r = get_pca()
    if _r is None:
        scree_html = None
    else:
        _fig = make_scree_plot(
            _r["evr"],
            _r["n_total"], _r["emb_dim"], _r["n_used"], _r["backend"],
            "dark" if map_theme.value else "light",
        )
        scree_html = mo.as_html(_fig)
    return (scree_html,)


@app.function
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


@app.function
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


@app.function
def get_thumb_dimensions(src_img_tbl, base_size=192):
    """Extract spatial extent from table metadata and compute thumb dimensions."""
    import json
    raw_meta = src_img_tbl.schema.metadata or {}
    ds_info = json.loads(raw_meta.get(b"dataset_info", "{}")) if raw_meta else {}
    ext = ds_info.get("spatial_extent", {})
    if ext:
        return compute_thumb_dimensions(ext, base_size)
    return base_size, base_size


@app.function
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


@app.function
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


@app.function
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


@app.function
def composite_attention_overlay(
    image_blob: bytes,
    attention_flat,
    spatial_h: int,
    spatial_w: int,
    alpha_min: float = 0.05,
    bg_color: tuple = (255, 255, 255),
) -> bytes:
    """Modulate image visibility by attention: high-attention areas show the full
    image, low-attention areas fade toward bg_color.  No colormap is used.
    Returns JPEG bytes of the composited image."""
    import io
    import numpy as np
    from PIL import Image

    img = Image.open(io.BytesIO(image_blob)).convert("RGBA")
    img_w, img_h = img.size

    # Reshape flat attention → 2-D, normalise to [0, 1]
    attn = np.array(attention_flat, dtype=np.float32).reshape(spatial_h, spatial_w)
    attn = (attn - attn.min()) / (attn.max() - attn.min() + 1e-8)

    # Scale to [alpha_min, 1] so even the lowest-attention pixels remain slightly visible
    attn_alpha = alpha_min + (1.0 - alpha_min) * attn

    # Upsample attention mask to image size
    attn_img = Image.fromarray((attn_alpha * 255).astype(np.uint8), "L").resize(
        (img_w, img_h), Image.NEAREST
    )

    # Replace image alpha channel with the attention mask
    img_arr = np.array(img)
    img_arr[..., 3] = np.array(attn_img)
    masked = Image.fromarray(img_arr, "RGBA")

    # Composite over solid background so JPEG can be saved
    bg = Image.new("RGBA", (img_w, img_h), bg_color + (255,))
    composite = Image.alpha_composite(bg, masked).convert("RGB")

    buf = io.BytesIO()
    composite.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


@app.function
def render_thumbnail_gallery(thumbs, n_filtered, max_display, theme="light",
                             thumb_w=192, thumb_h=192):
    """Build HTML for a theme-aware thumbnail gallery with datetime labels."""
    import base64

    _c = get_theme_colors(theme)
    bg, text, border = _c["gallery_bg"], _c["text"], _c["border"]

    imgs = []
    for fname, blob, dt in thumbs:
        b64 = base64.b64encode(blob).decode()
        dt_str = dt.strftime("%Y-%m-%d %H:%M") if hasattr(dt, "strftime") else str(dt)
        imgs.append(
            f'<div style="display:inline-block;margin:3px;text-align:center">'
            f'<img src="data:image/jpeg;base64,{b64}" '
            f'style="width:{thumb_w}px;height:{thumb_h}px;object-fit:fill;border:1px solid {border};'
            f'border-radius:4px" title="{fname}"/>'
            f'<div style="font-size:11px;color:{text};max-width:{thumb_w}px;'
            f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap">'
            f'{dt_str}</div></div>'
        )

    count_msg = f"Showing {len(thumbs)} of {n_filtered} selected"
    if n_filtered > max_display:
        count_msg += f" (capped at {max_display})"

    gallery_html = (
        f'<div style="display:flex;flex-wrap:wrap;gap:4px;align-content:flex-start;'
        f'height:600px;overflow-y:auto;background:{bg};'
        f'border-radius:8px;padding:8px;border:1px solid {border}">'
        + "".join(imgs)
        + "</div>"
    )
    return count_msg, gallery_html


@app.cell
def _(get_pca, mo):
    _r = get_pca()
    if _r is None:
        pc_axes_input = mo.ui.text(value="", label="PCs to display (comma-separated)")
    else:
        _default = ",".join(str(i + 1) for i in range(min(6, len(_r["evr"]))))
        pc_axes_input = mo.ui.text(
            value=_default,
            label="PCs to display (comma-separated)",
        )
    return (pc_axes_input,)


@app.cell(hide_code=True)
def _(get_pca, mo, np):
    _r = get_pca()

    if _r is None:
        extra_axes_select = mo.ui.multiselect(options=[], value=[], label="Extra axes")
        normalize_toggle = mo.ui.switch(value=False, label="Equal PC axis range")
        attention_toggle = mo.ui.switch(value=False, label="Attention overlay")
    else:
        # Build options: datetime components + any numeric source columns
        _dt_options = ["dt:year", "dt:month", "dt:day", "dt:hour", "dt:minute", "dt:second"]
        _extra_options = list(_dt_options)
        _meta = _r.get("metadata")
        if _meta is not None and len(_meta) > 0:
            _skip = {"id", "filename", "dt"}
            for _col in _meta.columns:
                try:
                    _is_numeric = np.issubdtype(_meta[_col].dtype, np.number)
                except TypeError:
                    _is_numeric = False
                if _col not in _skip and _is_numeric:
                    _extra_options.append(_col)

        extra_axes_select = mo.ui.multiselect(
            options=_extra_options,
            value=["dt:month"],
            label="Extra axes",
        )
        normalize_toggle = mo.ui.switch(value=False, label="Equal PC axis range")
        attention_toggle = mo.ui.switch(value=False, label="Attention overlay")

    parcoord_options_ui = mo.hstack(
        [extra_axes_select, normalize_toggle, attention_toggle], justify="start"
    )
    return (
        attention_toggle,
        extra_axes_select,
        normalize_toggle,
        parcoord_options_ui,
    )


@app.cell(hide_code=True)
def _(
    ParallelCoordinates,
    extra_axes_select,
    get_pca,
    mo,
    np,
    pc_axes_input,
    pl,
):
    _r = get_pca()

    if _r is None:
        parcoord_data_cols = {}
        parcoord_widget = None
    else:
        # Parse PC indices
        _max_pc = _r["scores"].shape[1]
        _indices = []
        for _tok in pc_axes_input.value.split(","):
            _tok = _tok.strip()
            if _tok.isdigit() and 1 <= int(_tok) <= _max_pc:
                _indices.append(int(_tok) - 1)

        if len(_indices) == 0:
            parcoord_data_cols = {}
            parcoord_widget = mo.callout(mo.md("Enter valid PC numbers."), kind="warn")
        else:
            # Build extra columns from metadata
            _meta = _r.get("metadata")
            _extra_cols = {}
            if _meta is not None and len(_meta) > 0:
                _MONTH_LABELS = {
                    1: "01 Jan", 2: "02 Feb", 3: "03 Mar", 4: "04 Apr",
                    5: "05 May", 6: "06 Jun", 7: "07 Jul", 8: "08 Aug",
                    9: "09 Sep", 10: "10 Oct", 11: "11 Nov", 12: "12 Dec",
                }
                for _sel in extra_axes_select.value:
                    if _sel.startswith("dt:") and "dt" in _meta.columns:
                        _part = _sel.split(":")[1]
                        if hasattr(_meta["dt"].dt, _part):
                            if _sel == "dt:month":
                                # String labels sort alphabetically = chronologically
                                _month_ints = getattr(_meta["dt"].dt, "month").values
                                _extra_cols[_sel] = np.array(
                                    [_MONTH_LABELS.get(int(v), str(v)) for v in _month_ints]
                                )
                            else:
                                _vals = getattr(_meta["dt"].dt, _part).astype(float)
                                _n_nan = int(_vals.isna().sum())
                                if _n_nan > 0:
                                    print(f"[parcoord] WARNING: {_sel} has {_n_nan} NaN values "
                                          f"out of {len(_vals)} rows (missing metadata)")
                                _extra_cols[_sel] = _vals.values
                    elif _sel in _meta.columns:
                        _extra_cols[_sel] = _meta[_sel].astype(float).values

            # Build PC columns (reversed so HiPlot's internal .reverse() yields PC1→PCn left-to-right)
            _scores = _r["scores"]
            _pc_values = {f"PC{i + 1}": _scores[:, i] for i in reversed(_indices)}

            # Combine in reverse: PCs (reversed) then extras (reversed)
            # HiPlot's componentDidMount reverses Object.keys order, so after
            # .reverse() the plot renders: extra1, extra2, ..., PC1, PC2, ... (left→right)
            _reversed_extras = dict(reversed(list(_extra_cols.items())))
            parcoord_data_cols = {**_pc_values, **_reversed_extras}

            # Always include 2 sentinel rows (per-column min/max) so data.length
            # stays constant — prevents HiPlot React key change & axis-order flip.
            _keys = list(parcoord_data_cols.keys())

            def _col_extreme(arr, hi=False):
                first = arr.flat[0] if hasattr(arr, "flat") else arr[0]
                if isinstance(first, str):
                    sv = sorted(set(arr))
                    return sv[-1] if hi else sv[0]
                return float(np.nanmax(arr) if hi else np.nanmin(arr))

            _sentinel_lo = {k: _col_extreme(parcoord_data_cols[k], hi=False) for k in _keys}
            _sentinel_hi = {k: _col_extreme(parcoord_data_cols[k], hi=True) for k in _keys}
            _base_df = pl.DataFrame(parcoord_data_cols)
            _sent_df = pl.DataFrame([_sentinel_lo, _sentinel_hi]).cast(dict(zip(_base_df.columns, _base_df.dtypes)))
            _df = pl.concat([_base_df, _sent_df]).select(_keys)
            _color_axis = _keys[-1]
            parcoord_widget = mo.ui.anywidget(ParallelCoordinates(_df, height=350, color_by=_color_axis))
    return parcoord_data_cols, parcoord_widget


@app.cell
def _(normalize_toggle, np, parcoord_data_cols, parcoord_widget):
    if parcoord_widget is not None and parcoord_data_cols:
        _pc_keys = [k for k in parcoord_data_cols if k.startswith("PC")]
        _keys = list(parcoord_data_cols.keys())

        # Build base rows with explicit key order, NaN→None for JSON
        _col_vals = []
        for _k in _keys:
            _raw = parcoord_data_cols[_k]
            _lst = _raw.tolist() if hasattr(_raw, "tolist") else list(_raw)
            _col_vals.append([None if (isinstance(_v, float) and _v != _v) else _v for _v in _lst])
        _base_rows = [dict(zip(_keys, _row)) for _row in zip(*_col_vals)]

        def _col_extreme(arr, hi=False):
            first = arr.flat[0] if hasattr(arr, "flat") else arr[0]
            if isinstance(first, str):
                sv = sorted(set(arr))
                return sv[-1] if hi else sv[0]
            return float(np.nanmax(arr) if hi else np.nanmin(arr))

        if normalize_toggle.value and _pc_keys:
            _all_pc = np.column_stack([parcoord_data_cols[_k] for _k in _pc_keys])
            _gmin, _gmax = float(_all_pc.min()), float(_all_pc.max())
            _sentinel_lo = {k: (_gmin if k in _pc_keys else _col_extreme(parcoord_data_cols[k], hi=False)) for k in _keys}
            _sentinel_hi = {k: (_gmax if k in _pc_keys else _col_extreme(parcoord_data_cols[k], hi=True)) for k in _keys}
        else:
            # Per-column natural extremes — no visual effect, keeps data.length stable
            _sentinel_lo = {k: _col_extreme(parcoord_data_cols[k], hi=False) for k in _keys}
            _sentinel_hi = {k: _col_extreme(parcoord_data_cols[k], hi=True) for k in _keys}

        # Always N+2 rows → React key stays constant → no remount → no order flip
        parcoord_widget.widget.data = _base_rows + [_sentinel_lo, _sentinel_hi]
    return


@app.cell
def _(
    attention_toggle,
    config,
    get_pca,
    img_emb_tbl,
    map_theme,
    mo,
    parcoord_data_cols,
    parcoord_widget,
    src_img_tbl,
):
    _r = get_pca()

    if _r is None or src_img_tbl is None or parcoord_widget is None or not parcoord_data_cols:
        gallery_ui = None
    else:
        # Use brush_extents (synced) to filter — filtered_uids is bugged in wigglystuff
        _val = parcoord_widget.value
        _brush = _val.get("brush_extents", {}) if isinstance(_val, dict) else {}
        _ids = _r["image_ids"]
        _n_real = len(_ids)

        _filtered = apply_brush_filter(parcoord_data_cols, _brush)
        if _filtered is None:
            _filtered = list(range(_n_real))
        else:
            # Exclude sentinel row indices
            _filtered = [i for i in _filtered if i < _n_real]

        _selected_ids = [str(_ids[i]) for i in _filtered]
        _max_display = 20

        # Compute thumbnail dimensions from spatial extent metadata
        _tw, _th = get_thumb_dimensions(src_img_tbl)

        _thumbs = fetch_thumbnails_batch(src_img_tbl, _selected_ids, _max_display)

        # Attention overlay — composite each thumbnail when toggle is on
        if attention_toggle.value and img_emb_tbl is not None:
            _attn_cols = [f.name for f in img_emb_tbl.schema]
            if "attention_map" in _attn_cols:
                _sh = int(config.get("attention_spatial_h", 14))
                _sw = int(config.get("attention_spatial_w", 14))
                _display_ids = _selected_ids[:len(_thumbs)]
                _attn_maps = fetch_attention_maps(img_emb_tbl, _display_ids)
                _bg_color  = get_theme_colors(
                    "dark" if map_theme.value else "light"
                )["gallery_bg_rgb"]
                _thumbs = [
                    (fname,
                     composite_attention_overlay(blob, _attn_maps[iid], _sh, _sw,
                                                 bg_color=_bg_color)
                     if iid in _attn_maps else blob,
                     dt)
                    for (fname, blob, dt), iid in zip(_thumbs, _display_ids)
                ]

        _count_msg, _gallery_html = render_thumbnail_gallery(
            _thumbs, len(_filtered), _max_display,
            theme="dark" if map_theme.value else "light",
            thumb_w=_tw, thumb_h=_th,
        )
        gallery_ui = mo.vstack([mo.md(f"**{_count_msg}**"), mo.Html(_gallery_html)])
    return (gallery_ui,)


@app.cell(hide_code=True)
def _(
    gallery_ui,
    mo,
    parcoord_options_ui,
    parcoord_widget,
    pc_axes_input,
    pca_controls_ui,
    pca_status,
    scree_html,
):
    _items = [pca_controls_ui]
    if pca_status is not None:
        _items.append(pca_status)
    if scree_html is not None:
        _items += [scree_html, pc_axes_input, parcoord_options_ui, parcoord_widget]
    if gallery_ui is not None:
        _items.append(gallery_ui)
    pca_tab = mo.vstack([i for i in _items if i is not None])
    return (pca_tab,)


@app.cell
def _(mo):
    get_umap_result, set_umap_result = mo.state(None)
    return get_umap_result, set_umap_result


@app.cell
def _(mo):
    umap_n_vectors   = mo.ui.number(value=5000, start=2, label="Max vectors")
    umap_n_neighbors = mo.ui.number(value=15, start=2, stop=200, step=1, label="n_neighbors")
    umap_min_dist    = mo.ui.number(value=0.1, start=0.0, stop=1.0, step=0.05, label="min_dist")
    run_umap = mo.ui.run_button(
        label="▶ Run UMAP", kind="success",
    )
    umap_controls_ui = mo.hstack(
        [umap_n_vectors, umap_n_neighbors, umap_min_dist, run_umap],
        justify="start", align="end",
    )
    return (
        run_umap,
        umap_controls_ui,
        umap_min_dist,
        umap_n_neighbors,
        umap_n_vectors,
    )


@app.cell
def _(
    experiment_selector,
    get_umap_result,
    img_emb_tbl,
    mo,
    run_umap,
    set_umap_result,
    src_img_tbl,
    umap_min_dist,
    umap_n_neighbors,
    umap_n_vectors,
):
    _exp = experiment_selector.value if experiment_selector.value else None
    if get_umap_result() is not None and get_umap_result().get("experiment") != _exp:
        set_umap_result(None)
    if not run_umap.value:
        umap_status = mo.callout(
            mo.md("Configure options above and click **▶ Run UMAP**."), kind="info")
    elif img_emb_tbl is None:
        umap_status = mo.callout(mo.md("No experiment loaded."), kind="warn")
    else:
        try:
            _X, _ids, _n_total = load_embedding_matrix(img_emb_tbl, umap_n_vectors.value)
            _emb, _backend = run_umap_best(_X, umap_n_neighbors.value, umap_min_dist.value)
            _meta = fetch_metadata_for_ids(src_img_tbl, list(_ids))
            set_umap_result({
                "embedding": _emb, "image_ids": _ids, "metadata": _meta,
                "backend": _backend,
                "n_total": _n_total, "emb_dim": _X.shape[1], "n_used": len(_X),
                "experiment": _exp,
            })
            umap_status = None
        except ImportError as _e:
            umap_status = mo.callout(
                mo.md(f"**Missing dependency:** {_e}"), kind="danger")
        except Exception as _e:
            umap_status = mo.callout(
                mo.md(f"**UMAP failed:** {_e}"), kind="danger")
    return (umap_status,)


@app.cell
def _(get_umap_result, mo, np):
    _r = get_umap_result()
    if _r is None:
        umap_color_select = mo.ui.dropdown(options=[], value=None, label="Color by")
    else:
        _dt_opts = ["dt:year", "dt:month", "dt:day", "dt:hour"]
        _opts = list(_dt_opts)
        _meta = _r.get("metadata")
        if _meta is not None and len(_meta) > 0:
            _skip = {"id", "filename", "dt"}
            for _col in _meta.columns:
                try:
                    _ok = np.issubdtype(_meta[_col].dtype, np.number)
                except TypeError:
                    _ok = False
                if _col not in _skip and _ok:
                    _opts.append(_col)
        umap_color_select = mo.ui.dropdown(
            options=_opts, value="dt:month", label="Color by")
    return (umap_color_select,)


@app.cell
def _(get_umap_result, map_theme, mo, np, umap_color_select):
    import plotly.graph_objects as _go_umap
    _r = get_umap_result()
    if _r is None:
        umap_scatter = mo.callout(mo.md("Run UMAP first."), kind="neutral")
    else:
        _emb   = _r["embedding"]
        _meta  = _r.get("metadata")
        _theme = "dark" if map_theme.value else "light"
        _c     = get_theme_colors(_theme)
        _bg    = "#1a1a1a" if _theme == "dark" else "white"

        _ck = umap_color_select.value
        _cvals = None
        if _ck and _meta is not None and len(_meta):
            if _ck.startswith("dt:") and "dt" in _meta.columns:
                _part = _ck.split(":")[1]
                if hasattr(_meta["dt"].dt, _part):
                    _cvals = getattr(_meta["dt"].dt, _part).astype(float).values
            elif _ck in _meta.columns:
                _cvals = _meta[_ck].astype(float).values

        # Categorical vs continuous detection
        _is_cat = False
        if _cvals is not None:
            _uniq = np.unique(_cvals[np.isfinite(_cvals)])
            _is_cat = _ck.startswith("dt:") or len(_uniq) <= 20

        if _is_cat:
            import plotly.colors as _pc
            _n = len(_uniq)
            _pal = _pc.qualitative.Dark24 + _pc.qualitative.Light24
            _cat_map = {v: i for i, v in enumerate(_uniq)}
            _ccodes = np.array([_cat_map.get(float(v), 0) for v in _cvals], dtype=float)
            _eps = 1e-6
            _cscale = []
            for _i in range(_n):
                _col = _pal[_i % len(_pal)]
                _lo = _i / _n + (_eps if _i > 0 else 0)
                _hi = (_i + 1) / _n - (_eps if _i < _n - 1 else 0)
                _cscale.append([_lo, _col])
                _cscale.append([_hi, _col])
            if _ck == "dt:month":
                _MNAMES = ["Jan","Feb","Mar","Apr","May","Jun",
                           "Jul","Aug","Sep","Oct","Nov","Dec"]
                _ticktext = [_MNAMES[int(v)-1] if 1 <= int(v) <= 12 else str(v)
                             for v in _uniq]
            elif _ck in ("dt:year", "dt:day", "dt:hour"):
                _ticktext = [str(int(v)) for v in _uniq]
            else:
                _ticktext = [str(v) for v in _uniq]
            _marker_kw = dict(
                size=5, opacity=0.75,
                color=_ccodes, colorscale=_cscale, cmin=0, cmax=_n,
                showscale=True,
                colorbar=dict(
                    title=_ck, thickness=12,
                    tickmode="array",
                    tickvals=[_i + 0.5 for _i in range(_n)],
                    ticktext=_ticktext,
                ),
            )
        else:
            _marker_kw = dict(
                size=5, opacity=0.75,
                color=_cvals if _cvals is not None else _c["bar_color"],
                colorscale="Viridis" if _cvals is not None else None,
                showscale=_cvals is not None,
                colorbar=dict(title=_ck or "", thickness=12) if _cvals is not None else None,
            )

        # Compact tooltip data: [datetime_str, value_label] per point
        _has_dt = _meta is not None and "dt" in _meta.columns
        _dt_strs = (
            _meta["dt"].dt.strftime("%Y-%m-%d %H:%M").fillna("").values
            if _has_dt else np.full(len(_emb), "")
        )
        if _cvals is not None:
            _val_labels = (
                np.array([_ticktext[_cat_map.get(float(v), 0)] for v in _cvals])
                if _is_cat
                else np.array([f"{v:.3g}" for v in _cvals])
            )
            _hover_val_line = f"<br><b>{_ck}:</b> %{{customdata[1]}}"
        else:
            _val_labels = np.full(len(_emb), "")
            _hover_val_line = ""
        _customdata = np.column_stack([_dt_strs, _val_labels])

        _fig = _go_umap.Figure(_go_umap.Scattergl(
            x=_emb[:, 0], y=_emb[:, 1],
            mode="markers",
            marker=_marker_kw,
            customdata=_customdata,
            hovertemplate="%{customdata[0]}" + _hover_val_line + "<extra></extra>",
            selected=dict(marker=dict(color="orange", size=8, opacity=1.0)),
            unselected=dict(marker=dict(opacity=0.15)),
        ))

        _xr = float(np.ptp(_emb[:, 0]))
        _yr = float(np.ptp(_emb[:, 1]))
        _l2, _r2, _t2, _b2 = 50, 20, 35, 40
        _plot_w = max(600 - _l2 - _r2, 1)
        _fig_h  = int(_plot_w * (_yr / _xr) + _t2 + _b2) if _xr else 480
        _fig_h  = max(300, min(_fig_h, 700))

        _fig.update_layout(
            template=_c["plotly_template"],
            dragmode="select",
            clickmode="event+select",
            xaxis=dict(title="UMAP 1", showgrid=True,
                       gridcolor="rgba(255,255,255,0.12)" if _theme == "dark" else "rgba(0,0,0,0.10)",
                       zeroline=False),
            yaxis=dict(title="UMAP 2", showgrid=True,
                       gridcolor="rgba(255,255,255,0.12)" if _theme == "dark" else "rgba(0,0,0,0.10)",
                       zeroline=False, scaleanchor="x", scaleratio=1),
            height=_fig_h,
            margin=dict(l=_l2, r=_r2, t=_t2, b=_b2),
            paper_bgcolor=_bg, plot_bgcolor=_bg,
            title=dict(
                text=(f"UMAP — {_r['n_used']:,} / {_r['n_total']:,} images"
                      f" × dim {_r['emb_dim']}  [{_r['backend']}]"),
                font=dict(size=13),
            ),
        )
        umap_scatter = mo.ui.plotly(_fig)
    return (umap_scatter,)


@app.cell
def _(get_umap_result, map_theme, mo, np, src_img_tbl, umap_scatter):
    _r = get_umap_result()
    if _r is None or src_img_tbl is None or not hasattr(umap_scatter, "value"):
        umap_gallery_ui = None
    else:
        _MAX = 20
        # umap_scatter.value is a processed list of point dicts (marimo output).
        # umap_scatter.ranges holds the raw Plotly range {"x":[x0,x1],"y":[y0,y1]}.
        # marimo's bridge filters by X only, so we apply the full 2D box ourselves.
        _sel_ids = []
        _rng = umap_scatter.ranges  # {} when nothing selected
        if _rng and "x" in _rng and "y" in _rng:
            _emb = _r["embedding"]
            _x0, _x1 = min(_rng["x"]), max(_rng["x"])
            _y0, _y1 = min(_rng["y"]), max(_rng["y"])
            _mask = (
                (_emb[:, 0] >= _x0) & (_emb[:, 0] <= _x1) &
                (_emb[:, 1] >= _y0) & (_emb[:, 1] <= _y1)
            )
            _sel_ids = [str(_r["image_ids"][i]) for i in np.where(_mask)[0]]
        if not _sel_ids:
            umap_gallery_ui = mo.callout(
                mo.md("*Box-select points on the scatter to see thumbnails.*"), kind="neutral")
        else:
            _tw, _th = get_thumb_dimensions(src_img_tbl)
            _thumbs  = fetch_thumbnails_batch(src_img_tbl, _sel_ids, _MAX)
            _msg, _html = render_thumbnail_gallery(
                _thumbs, len(_sel_ids), _MAX,
                theme="dark" if map_theme.value else "light",
                thumb_w=_tw, thumb_h=_th,
            )
            umap_gallery_ui = mo.vstack([mo.md(f"**{_msg}**"), mo.Html(_html)])
    return (umap_gallery_ui,)


@app.cell
def _(
    mo,
    umap_color_select,
    umap_controls_ui,
    umap_gallery_ui,
    umap_scatter,
    umap_status,
):
    _items_u = [umap_controls_ui]
    if umap_status is not None:
        _items_u.append(umap_status)
    else:
        _items_u.append(mo.hstack([umap_color_select], justify="start"))
        _gallery_u = (umap_gallery_ui if umap_gallery_ui is not None
                      else mo.callout(
                          mo.md("*Select points to see thumbnails.*"), kind="neutral"))
        _items_u.append(mo.Html(
            '<div style="display:flex;gap:1rem;align-items:flex-start;width:100%">'
            f'<div style="flex:0 0 60%;min-width:0">{umap_scatter.text}</div>'
            f'<div style="flex:0 0 38%;min-width:0">{_gallery_u.text}</div>'
            '</div>'
        ))
    umap_tab = mo.vstack(_items_u)
    return (umap_tab,)


@app.cell
def _(mo, pca_tab, umap_tab):
    dim_reduction_tab = mo.ui.tabs({"PCA": pca_tab, "UMAP": umap_tab})
    return (dim_reduction_tab,)


@app.function
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


@app.function
def build_coastline_traces(lat_min, lat_max, lon_min, lon_max, n_rows, n_cols):
    """Return a list of go.Scatter coastline traces clipped to the given extent."""
    import numpy as np
    import plotly.graph_objects as go

    lat_step = (lat_max - lat_min) / n_rows
    lon_step = (lon_max - lon_min) / n_cols
    buffer = max(lat_step, lon_step)

    lon_offset = 360 if lon_min > 180 else 0
    lon_min_180 = lon_min - lon_offset
    lon_max_180 = lon_max - lon_offset

    try:
        import cartopy.feature as cfeature
        from shapely.geometry import box as shapely_box
    except ImportError as e:
        raise ImportError(
            "cartopy is required for coastline rendering. "
            "Install with: pip install cartopy"
        ) from e

    bbox = shapely_box(
        lon_min_180 - buffer, lat_min - buffer,
        lon_max_180 + buffer, lat_max + buffer,
    )
    try:
        coast_geoms = list(cfeature.COASTLINE.with_scale("110m").geometries())
    except Exception:
        coast_geoms = []

    traces = []
    for geom in coast_geoms:
        try:
            if not geom.intersects(bbox):
                continue
            lines = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
            for line in lines:
                try:
                    xy = np.array(line.coords)
                    if xy.ndim != 2 or xy.shape[0] < 2:
                        continue
                    traces.append(go.Scatter(
                        x=[xi + lon_offset for xi in xy[:, 0].tolist()],
                        y=xy[:, 1].tolist(),
                        mode="lines",
                        line=dict(color="white", width=1.5),
                        opacity=0.8,
                        showlegend=False,
                        hoverinfo="skip",
                        name="coastline",
                    ))
                except Exception:
                    continue
        except Exception:
            continue
    return traces


@app.function
def make_patch_heatmap(lat_min, lat_max, lon_min, lon_max, n_rows, n_cols):
    """Invisible N×M heatmap whose z values are flat patch indices (click target)."""
    import numpy as np
    import plotly.graph_objects as go

    lat_step = (lat_max - lat_min) / n_rows
    lon_step = (lon_max - lon_min) / n_cols
    z = np.arange(n_rows * n_cols).reshape(n_rows, n_cols)
    hm_x = [lon_min + (c + 0.5) * lon_step for c in range(n_cols)]
    hm_y = [lat_max - (r + 0.5) * lat_step for r in range(n_rows)]
    return go.Heatmap(
        z=z, x=hm_x, y=hm_y,
        opacity=0.01,
        showscale=False,
        colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
        hovertemplate="Patch %{z}  (%{y:.2f}°, %{x:.2f}°)<extra></extra>",
    )


@app.function
def make_selection_shape(patch_idx, lat_min, lat_max, lon_min, lon_max, n_rows, n_cols):
    """Return a red rectangle shape dict for the selected patch, or None."""
    if patch_idx is None:
        return None
    lat_step = (lat_max - lat_min) / n_rows
    lon_step = (lon_max - lon_min) / n_cols
    p_row = patch_idx // n_cols
    p_col = patch_idx % n_cols
    x0 = lon_min + p_col * lon_step
    y1 = lat_max - p_row * lat_step
    y0 = y1 - lat_step
    return dict(
        type="rect",
        x0=x0, x1=x0 + lon_step,
        y0=y0, y1=y1,
        line=dict(color="red", width=3),
        fillcolor="rgba(0,0,0,0)",
        layer="above",
    )


@app.function
def build_geo_patch_figure(
    img_arr, lon_min, lon_max, lat_min, lat_max,
    coast_traces, heatmap_trace, selection_shape, theme="light", target_w=600,
):
    """Assemble the three-layer geo patch figure from pre-built components."""
    import plotly.graph_objects as go

    _is_dark = (theme == "dark")
    _bg   = "#1a1a1a" if _is_dark else "white"
    _text = "#e0e0e0" if _is_dark else "#222222"

    H, W = img_arr.shape[:2]
    fig = go.Figure()

    fig.add_trace(go.Image(
        z=img_arr,
        x0=lon_min, dx=(lon_max - lon_min) / W,
        y0=lat_max, dy=-(lat_max - lat_min) / H,
        hoverinfo="skip",
    ))

    for trace in coast_traces:
        fig.add_trace(trace)

    fig.add_trace(heatmap_trace)

    shapes = [selection_shape] if selection_shape is not None else []
    _l, _r, _t, _b = 65, 10, 10, 40
    _plot_w = max(target_w - _l - _r, 1)
    _lat_range = lat_max - lat_min
    _lon_range = lon_max - lon_min
    _fig_h = int(_plot_w * (_lat_range / _lon_range) + _t + _b) if _lon_range else 400
    fig.update_layout(
        xaxis=dict(range=[lon_min, lon_max], title="Longitude",
                   tickformat=".2f", ticksuffix="°", showgrid=False,
                   tickfont=dict(color=_text), title_font=dict(color=_text)),
        yaxis=dict(range=[lat_min, lat_max], title="Latitude",
                   tickformat=".2f", ticksuffix="°",
                   scaleanchor="x", scaleratio=1, showgrid=False,
                   tickfont=dict(color=_text), title_font=dict(color=_text)),
        shapes=shapes,
        uirevision="geo_patch_map",
        clickmode="event+select",
        dragmode="pan",
        height=_fig_h,
        margin=dict(l=_l, r=_r, t=_t, b=_b),
        plot_bgcolor=_bg,
        paper_bgcolor=_bg,
    )
    return fig


@app.function
def apply_similarity_overlay(image_blob, matched_patch_distances, n_rows, n_cols, alpha_min=0.08, bg_color=(0, 0, 0)):
    """Fade non-matched patches toward bg_color; matched patches stay opaque."""
    import io
    import numpy as np
    from PIL import Image

    img = Image.open(io.BytesIO(image_blob)).convert("RGBA")
    iw, ih = img.size

    alpha_grid = np.full((n_rows, n_cols), alpha_min, dtype=np.float32)
    if matched_patch_distances:
        dists = np.array(list(matched_patch_distances.values()), dtype=np.float32)
        d_min, d_max = dists.min(), dists.max()
        for pidx, dist in matched_patch_distances.items():
            row, col = int(pidx) // n_cols, int(pidx) % n_cols
            norm = (dist - d_min) / (d_max - d_min + 1e-8)
            alpha_grid[row, col] = 1.0 - norm * 0.5

    ph, pw = ih // n_rows, iw // n_cols
    alpha_up = np.repeat(np.repeat(alpha_grid, ph, axis=0), pw, axis=1)
    pad_h, pad_w = ih - alpha_up.shape[0], iw - alpha_up.shape[1]
    if pad_h > 0 or pad_w > 0:
        alpha_up = np.pad(alpha_up, ((0, pad_h), (0, pad_w)), mode="edge")

    img_arr = np.array(img)
    img_arr[..., 3] = (alpha_up * 255).astype(np.uint8)
    masked = Image.fromarray(img_arr, "RGBA")
    bg = Image.new("RGBA", (iw, ih), bg_color + (255,))
    composite = Image.alpha_composite(bg, masked).convert("RGB")

    buf = io.BytesIO()
    composite.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


@app.function
def render_basemap(lat_min, lat_max, lon_min, lon_max, target_w=512, theme="light"):
    """Render a cartopy land/ocean map of the extent as a numpy RGB array."""
    import io
    import numpy as np
    from PIL import Image as _PILImage
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
    except ImportError as e:
        raise ImportError(
            "cartopy is required for the basemap. Install with: pip install cartopy"
        ) from e

    _is_dark = (theme == "dark")
    _ocean = "#1e3a5f" if _is_dark else "#a8c8e8"
    _land  = "#3a3a3a" if _is_dark else "#d4d4d4"
    _coast = "#aaaaaa" if _is_dark else "#555555"
    _bg    = "#1a1a1a" if _is_dark else "#ffffff"

    lon_offset = 360 if lon_min > 180 else 0
    lon_min_180 = lon_min - lon_offset
    lon_max_180 = lon_max - lon_offset
    aspect = (lon_max - lon_min) / max(lat_max - lat_min, 1e-6)
    target_h = max(1, int(target_w / aspect))

    fig, ax = plt.subplots(
        figsize=(target_w / 100, target_h / 100), dpi=100,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    fig.patch.set_facecolor(_bg)
    ax.set_facecolor(_bg)
    ax.set_extent([lon_min_180, lon_max_180, lat_min, lat_max], crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN.with_scale("110m"),    color=_ocean, zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("110m"),     color=_land,  zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor=_coast, linewidth=0.8, zorder=2)
    ax.add_feature(cfeature.BORDERS.with_scale("110m"),   edgecolor=_coast, linewidth=0.5, zorder=2)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0, facecolor=_bg)
    plt.close(fig)
    buf.seek(0)
    return np.array(_PILImage.open(buf).convert("RGB"))


@app.function
def build_spatial_filter_shapes(selected_indices, lat_min, lat_max, lon_min, lon_max, n_rows, n_cols):
    """Return a list of Plotly rect shapes highlighting each selected patch."""
    if not selected_indices:
        return []
    lat_step = (lat_max - lat_min) / n_rows
    lon_step = (lon_max - lon_min) / n_cols
    shapes = []
    for idx in selected_indices:
        row, col = idx // n_cols, idx % n_cols
        x0 = lon_min + col * lon_step
        x1 = x0 + lon_step
        y1 = lat_max - row * lat_step
        y0 = y1 - lat_step
        shapes.append(dict(
            type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
            line=dict(color="rgba(255,80,0,0.9)", width=1.5),
            fillcolor="rgba(255,80,0,0.25)",
        ))
    return shapes


@app.cell
def _(mo):
    ss_load_button = mo.ui.run_button(label="▶ Load Spatial Search", kind="success")
    return (ss_load_button,)


@app.cell
def _(mo):
    get_ss_init, set_ss_init = mo.state(None)
    get_ss_patch, set_ss_patch = mo.state(None)
    get_ss_spatial_filter, set_ss_spatial_filter = mo.state(None)
    return (
        get_ss_patch,
        get_ss_spatial_filter,
        set_ss_init,
        set_ss_patch,
        set_ss_spatial_filter,
    )


@app.cell
def _(config, map_theme, mo, set_ss_init, src_img_tbl, ss_load_button):
    if not ss_load_button.value:
        ss_init = None
        ss_init_status = mo.callout(
            mo.md("Click **▶ Load Spatial Search** to initialize."), kind="info"
        )
    elif src_img_tbl is None or config is None:
        ss_init = None
        ss_init_status = mo.callout(
            mo.md("Load an experiment first."), kind="warn"
        )
    else:
        _theme = "dark" if map_theme.value else "light"
        _lat_min, _lat_max, _lon_min, _lon_max, _n_rows, _n_cols = get_spatial_extent(src_img_tbl, config)
        _coast = build_coastline_traces(_lat_min, _lat_max, _lon_min, _lon_max, _n_rows, _n_cols)
        _heatmap = make_patch_heatmap(_lat_min, _lat_max, _lon_min, _lon_max, _n_rows, _n_cols)
        _bmap = render_basemap(_lat_min, _lat_max, _lon_min, _lon_max, theme=_theme)
        ss_init = dict(
            lat_min=_lat_min, lat_max=_lat_max,
            lon_min=_lon_min, lon_max=_lon_max,
            n_rows=_n_rows, n_cols=_n_cols,
            coast_traces=_coast,
            heatmap_trace=_heatmap,
            basemap=_bmap,
            theme=_theme,
        )
        set_ss_init(ss_init)
        ss_init_status = None
    return ss_init, ss_init_status


@app.cell
def _(mo):
    ss_n_similar_images = mo.ui.number(start=1, stop=50, step=1, value=50, label="Similar images")
    ss_n_similar_patches = mo.ui.number(start=10, stop=500, step=200, value=100, label="Max patches")
    ss_max_gallery = mo.ui.number(start=4, stop=100, step=4, value=25, label="Gallery cap")
    ss_refine_factor = mo.ui.number(start=1, stop=50, step=1, value=20, label="Refine factor")
    ss_similarity_toggle = mo.ui.switch(label="Similarity overlay")
    ss_search_mode = mo.ui.dropdown(
        options=["Image first", "Patch first"],
        value="Image first",
        label="Search mode",
    )
    return (
        ss_max_gallery,
        ss_n_similar_images,
        ss_n_similar_patches,
        ss_refine_factor,
        ss_search_mode,
        ss_similarity_toggle,
    )


@app.cell
def _(mo, src_img_tbl, ss_init):
    if src_img_tbl is None or ss_init is None:
        _col_opts = ["Best cosine dist"]
    else:
        _exclude = {"image_blob", "thumb_blob", "id", "filename", "dt"}
        _meta = [f.name for f in src_img_tbl.schema if f.name not in _exclude]
        _col_opts = ["Best cosine dist"] + _meta
    ss_timeline_color_by = mo.ui.dropdown(
        options=_col_opts,
        value="Best cosine dist",
        label="Color by",
    )
    return


@app.cell
def _(src_img_tbl, ss_init):
    if ss_init is None:
        ss_available_ids = []
        ss_available_dates = []
    else:
        _df = (
            src_img_tbl.search()
            .select(["id", "dt"])
            .to_pandas()
            .sort_values("dt")
            .reset_index(drop=True)
        )
        ss_available_ids = _df["id"].tolist()
        ss_available_dates = [str(d)[:10] for d in _df["dt"].tolist()]
    return ss_available_dates, ss_available_ids


@app.cell
def _(mo, src_img_tbl, ss_init):
    if ss_init is None or src_img_tbl is None:
        import pandas as _pd_mf
        ss_metadata_filter = mo.ui.dataframe(_pd_mf.DataFrame())
    else:
        _blob_cols = {"image_blob", "thumb_blob"}
        _meta_cols = [f.name for f in src_img_tbl.schema if f.name not in _blob_cols]
        _meta_df = src_img_tbl.search().select(_meta_cols).to_pandas()
        ss_metadata_filter = mo.ui.dataframe(_meta_df)
    return (ss_metadata_filter,)


@app.cell
def _(mo, ss_available_dates):
    _n = len(ss_available_dates)
    ss_date_picker = mo.ui.datetime(
        value=ss_available_dates[0] if _n else None,
        start=ss_available_dates[0] if _n else None,
        stop=ss_available_dates[-1] if _n else None,
    )
    return (ss_date_picker,)


@app.cell
def _(ss_available_dates, ss_available_ids, ss_date_picker):
    _s = str(ss_date_picker.value)[:10] if ss_date_picker.value else ""
    ss_selected_img_id = (
        ss_available_ids[ss_available_dates.index(_s)]
        if _s in ss_available_dates
        else (ss_available_ids[0] if ss_available_ids else "")
    )
    return (ss_selected_img_id,)


@app.cell
def _(get_ss_spatial_filter, mo, ss_init):
    if ss_init is None:
        ss_spatial_filter_map = mo.callout(mo.md("Load spatial search first."), kind="neutral")
    else:
        _d = ss_init
        _theme = _d["theme"]
        _fig_sf = build_geo_patch_figure(
            _d["basemap"], _d["lon_min"], _d["lon_max"], _d["lat_min"], _d["lat_max"],
            [], _d["heatmap_trace"], None, theme=_theme,
        )
        _active = get_ss_spatial_filter() or []
        _fig_sf.update_layout(
            shapes=build_spatial_filter_shapes(
                _active, _d["lat_min"], _d["lat_max"], _d["lon_min"], _d["lon_max"], _d["n_rows"], _d["n_cols"]
            ),
            uirevision="spatial_filter_map",
            dragmode="select",
        )
        ss_spatial_filter_map = mo.ui.plotly(_fig_sf)
    return (ss_spatial_filter_map,)


@app.cell
def _(
    get_ss_spatial_filter,
    set_ss_spatial_filter,
    ss_init,
    ss_spatial_filter_map,
):
    if ss_init is not None:
        _d = ss_init
        _rng = ss_spatial_filter_map.ranges          # {} when no box drawn
        if _rng and "x" in _rng and "y" in _rng:
            # Box select: convert lat/lon bounds → patch indices (replaces selection)
            _lon0, _lon1 = min(_rng["x"]), max(_rng["x"])
            _lat0, _lat1 = min(_rng["y"]), max(_rng["y"])
            _n_rows = _d["n_rows"]
            _n_cols = _d["n_cols"]
            _lnst = (_d["lon_max"] - _d["lon_min"]) / _n_cols
            _ltst = (_d["lat_max"] - _d["lat_min"]) / _n_rows
            _c0 = max(0,        int((_lon0 - _d["lon_min"]) / _lnst))
            _c1 = min(_n_cols-1, int((_lon1 - _d["lon_min"]) / _lnst))
            _r0 = max(0,        int((_d["lat_max"] - _lat1) / _ltst))
            _r1 = min(_n_rows-1, int((_d["lat_max"] - _lat0) / _ltst))
            _new = [r * _n_cols + c for r in range(_r0, _r1+1) for c in range(_c0, _c1+1)]
            set_ss_spatial_filter(_new if _new else None)
        else:
            # Single click: toggle the clicked patch in/out of selection
            _val = ss_spatial_filter_map.value
            if isinstance(_val, list) and _val and "z" in _val[0]:
                _idx = int(_val[0]["z"])
                _cur = list(get_ss_spatial_filter() or [])
                if _idx in _cur:
                    _cur.remove(_idx)
                else:
                    _cur.append(_idx)
                set_ss_spatial_filter(_cur if _cur else None)
    return


@app.cell
def _(get_ss_patch, mo, src_img_tbl, ss_init, ss_selected_img_id):
    import io as _io_ss
    import numpy as _np_ss
    from PIL import Image as _Image_ss

    if ss_init is None:
        ss_geo_patch_map = mo.callout(mo.md("Load spatial search first."), kind="neutral")
    else:
        _d = ss_init
        _theme = _d["theme"]
        _row = (
            src_img_tbl.search()
            .where(f"id = '{ss_selected_img_id}'")
            .select(["image_blob"])
            .limit(1)
            .to_pandas()
            .iloc[0]
        )
        _img_arr = _np_ss.array(_Image_ss.open(_io_ss.BytesIO(_row["image_blob"])).convert("RGB"))
        _sel = get_ss_patch()
        _shape = make_selection_shape(_sel, _d["lat_min"], _d["lat_max"], _d["lon_min"], _d["lon_max"], _d["n_rows"], _d["n_cols"])
        _fig = build_geo_patch_figure(
            _img_arr, _d["lon_min"], _d["lon_max"], _d["lat_min"], _d["lat_max"],
            _d["coast_traces"], _d["heatmap_trace"], _shape, theme=_theme,
        )
        ss_geo_patch_map = mo.ui.plotly(_fig)
    return (ss_geo_patch_map,)


@app.cell
def _(set_ss_patch, ss_geo_patch_map, ss_init):
    if ss_init is not None:
        _click = ss_geo_patch_map.value
        if isinstance(_click, list) and _click and "z" in _click[0]:
            set_ss_patch(int(_click[0]["z"]))
    return


@app.cell
def _(
    get_ss_patch,
    get_ss_spatial_filter,
    img_emb_tbl,
    patch_emb_tbl,
    ss_init,
    ss_metadata_filter,
    ss_n_similar_images,
    ss_n_similar_patches,
    ss_refine_factor,
    ss_search_mode,
    ss_selected_img_id,
):
    import pandas as _pd_ss

    if ss_init is None or not ss_selected_img_id:
        ss_top_df = None
    else:
        _patch_idx = get_ss_patch() if get_ss_patch() is not None else 0

        # Shared: fetch the query patch embedding
        _p_q = (
            patch_emb_tbl.search()
            .where(f"image_id = '{ss_selected_img_id}' AND patch_index = {_patch_idx}")
            .select(["embedding"])
            .limit(1)
            .to_pandas()
            .iloc[0]
        )

        # Shared: metadata and spatial filter clauses
        _allowed_ids = ss_metadata_filter.value["id"].tolist() if ss_metadata_filter.value is not None and len(ss_metadata_filter.value) > 0 else None
        _id_clause = ", ".join(f"'{i}'" for i in _allowed_ids) if _allowed_ids else None
        _spatial = get_ss_spatial_filter()
        _patch_clause = (
            f"patch_index IN ({', '.join(str(i) for i in _spatial)})"
            if _spatial else ""
        )

        if ss_search_mode.value == "Patch first":
            # Search patches directly — no image pre-filter
            _parts = []
            if _id_clause:
                _parts.append(f"image_id IN ({_id_clause})")
            if _patch_clause:
                _parts.append(_patch_clause)
            _where = " AND ".join(_parts) if _parts else None
            _search = (
                patch_emb_tbl.search(_p_q["embedding"], vector_column_name="embedding")
                .metric("cosine")
                .refine_factor(int(ss_refine_factor.value))
                .select(["image_id", "patch_index"])
                .limit(ss_n_similar_patches.value)
            )
            if _where:
                _search = _search.where(_where)
            ss_top_df = _search.to_pandas()

        else:
            # Image first: find similar images restricted to spatial region, then search patches
            _img_q = (
                img_emb_tbl.search()
                .where(f"image_id = '{ss_selected_img_id}'")
                .select(["embedding"])
                .limit(1)
                .to_pandas()
                .iloc[0]
            )

            # Restrict image candidates to those that have patches in the selected region
            if _spatial:
                _spatial_img_ids = (
                    patch_emb_tbl.search()
                    .where(_patch_clause)
                    .select(["image_id"])
                    .limit(100_000)
                    .to_pandas()["image_id"]
                    .unique()
                    .tolist()
                )
                if _allowed_ids:
                    _allowed_set = set(_allowed_ids)
                    _spatial_img_ids = [i for i in _spatial_img_ids if i in _allowed_set]
                _img_where = (
                    f"image_id IN ({', '.join(repr(i) for i in _spatial_img_ids)})"
                    if _spatial_img_ids else None
                )
                _no_candidates = not _spatial_img_ids
            elif _id_clause:
                _img_where = f"image_id IN ({_id_clause})"
                _no_candidates = False
            else:
                _img_where = None
                _no_candidates = False

            if _no_candidates:
                ss_top_df = _pd_ss.DataFrame(columns=["image_id", "patch_index", "_distance"])
            else:
                _search = (
                    img_emb_tbl.search(_img_q["embedding"], vector_column_name="embedding")
                    .metric("cosine")
                    .select(["image_id"])
                )
                if _img_where:
                    _search = _search.where(_img_where)
                _sim_ims = _search.limit(ss_n_similar_images.value).to_pandas()

                if _sim_ims.empty:
                    ss_top_df = _pd_ss.DataFrame(columns=["image_id", "patch_index", "_distance"])
                else:
                    _img_filter = ", ".join(f"'{i}'" for i in _sim_ims["image_id"].tolist())
                    _where = f"image_id IN ({_img_filter})"
                    if _patch_clause:
                        _where += f" AND {_patch_clause}"

                    ss_top_df = (
                        patch_emb_tbl.search(_p_q["embedding"], vector_column_name="embedding")
                        .where(_where)
                        .metric("cosine")
                        .refine_factor(int(ss_refine_factor.value))
                        .select(["image_id", "patch_index"])
                        .limit(ss_n_similar_patches.value)
                        .to_pandas()
                    )
    return (ss_top_df,)


@app.cell
def _(
    get_ss_spatial_filter,
    map_theme,
    mo,
    src_img_tbl,
    ss_available_ids,
    ss_init,
    ss_max_gallery,
    ss_metadata_filter,
    ss_similarity_toggle,
    ss_top_df,
):
    import io as _io_g
    from PIL import Image as _Image_g, ImageDraw as _ImageDraw_g
    import pandas as _pd_g

    if ss_top_df is None or ss_init is None:
        ss_gallery_ui = None
    else:
        _d = ss_init
        _MAX = int(ss_max_gallery.value)
        _n_rows = _d["n_rows"]
        _n_cols = _d["n_cols"]

        _groups = (
            ss_top_df.groupby("image_id")
            .agg(patch_index=("patch_index", list), _distance=("_distance", "min"))
            .sort_values("_distance")
            .head(_MAX)
        )

        _spatial_extent = {
            "lat_min": _d["lat_min"], "lat_max": _d["lat_max"],
            "lon_min": _d["lon_min"], "lon_max": _d["lon_max"],
        }
        _thumb_w, _thumb_h = compute_thumb_dimensions(_spatial_extent, base_size=192)

        _patch_dists = {
            (row["image_id"], int(row["patch_index"])): row["_distance"]
            for _, row in ss_top_df.iterrows()
        }

        _thumbs = []
        _date_map = {}
        for _img_id, _data in _groups.iterrows():
            _r = (
                src_img_tbl.search()
                .where(f"id = '{_img_id}'")
                .select(["image_blob", "dt"])
                .limit(1)
                .to_pandas()
                .iloc[0]
            )
            _date_map[_img_id] = _r["dt"]

            if ss_similarity_toggle.value:
                _matched = {
                    int(p): _patch_dists[(_img_id, int(p))]
                    for p in _data["patch_index"]
                    if (_img_id, int(p)) in _patch_dists
                }
                _blob = apply_similarity_overlay(_r["image_blob"], _matched, _n_rows, _n_cols)
            else:
                _im = _Image_g.open(_io_g.BytesIO(_r["image_blob"])).convert("RGB")
                _tw, _th = _im.size
                _patch_w = _tw // _n_cols
                _patch_h = _th // _n_rows
                _draw = _ImageDraw_g.Draw(_im)
                for _p in map(int, _data["patch_index"]):
                    _pr, _pc = _p // _n_cols, _p % _n_cols
                    _bx = (_pc * _patch_w, _pr * _patch_h, (_pc + 1) * _patch_w, (_pr + 1) * _patch_h)
                    _draw.rectangle(_bx, outline=(255, 80, 0), width=2)
                _buf = _io_g.BytesIO()
                _im.save(_buf, format="JPEG", quality=85)
                _blob = _buf.getvalue()

            # Resize to display dimensions before base64-encoding to keep HTML output small
            _im_t = _Image_g.open(_io_g.BytesIO(_blob)).resize((_thumb_w, _thumb_h), _Image_g.LANCZOS)
            _buf_t = _io_g.BytesIO()
            _im_t.save(_buf_t, format="JPEG", quality=82)
            _thumbs.append((f"{str(_r['dt'])[:10]}  ·  d={_data['_distance']:.2f}", _buf_t.getvalue(), _r["dt"]))

        _theme = "dark" if map_theme.value else "light"
        _n_patches = len(ss_top_df)
        _n_images = ss_top_df["image_id"].nunique()
        _n_shown = len(_groups)
        _cap = f" (capped at {_MAX})" if _n_images > _MAX else ""

        _sf = get_ss_spatial_filter() or []
        _filter_parts = []
        if _sf:
            _filter_parts.append(f"spatial filter active ({len(_sf)} region{'s' if len(_sf) != 1 else ''})")
        _meta_count = len(ss_metadata_filter.value) if ss_metadata_filter.value is not None else len(ss_available_ids)
        if _meta_count < len(ss_available_ids):
            _filter_parts.append(f"data filter: {_meta_count} of {len(ss_available_ids)} rows")
        _filter_note = "  ·  " + ",  ".join(_filter_parts) if _filter_parts else ""

        _status = mo.md(f"**{_n_patches} patches** across **{_n_images} images** — showing **{_n_shown}**{_cap}{_filter_note}")

        _, _gallery_html = render_thumbnail_gallery(
            _thumbs, _n_shown, _MAX, theme=_theme,
            thumb_w=_thumb_w, thumb_h=_thumb_h,
        )

        _df_merged = (
            ss_top_df.groupby("image_id")
            .agg(patch_indices=("patch_index", list), cosine_dists=("_distance", list))
            .reset_index()
        )
        _df_merged["date"] = _df_merged["image_id"].map(lambda x: str(_date_map.get(x, ""))[:10])
        _df_merged["best_dist"] = _df_merged["cosine_dists"].apply(min)
        _df_merged = (
            _df_merged[["date", "image_id", "patch_indices", "cosine_dists", "best_dist"]]
            .sort_values("best_dist")
            .reset_index(drop=True)
        )

        _visual_tab = mo.vstack([_status, mo.Html(_gallery_html)])
        _data_tab = mo.ui.table(_df_merged, selection=None)
        ss_gallery_ui = mo.vstack([
            mo.hstack([ss_similarity_toggle], justify="end"),
            mo.ui.tabs({"Visuals": _visual_tab, "Data": _data_tab}),
        ])
    return (ss_gallery_ui,)


@app.cell
def _():
    # Timeline chart temporarily disabled
    ss_timeline_ui = None
    return


@app.cell
def _(
    get_ss_patch,
    get_ss_spatial_filter,
    mo,
    set_ss_spatial_filter,
    ss_date_picker,
    ss_gallery_ui,
    ss_geo_patch_map,
    ss_init,
    ss_init_status,
    ss_load_button,
    ss_max_gallery,
    ss_metadata_filter,
    ss_n_similar_images,
    ss_n_similar_patches,
    ss_refine_factor,
    ss_search_mode,
    ss_similarity_toggle,
    ss_spatial_filter_map,
):
    _items = [mo.hstack([ss_load_button], justify="start")]
    if ss_init_status is not None:
        _items.append(ss_init_status)
    if ss_init is not None:
        _sel = get_ss_patch()
        _label_q = f"**Selected patch:** `{_sel}`" if _sel is not None else "*Click a patch to select it*"
        _active_sf = get_ss_spatial_filter() or []
        _label_s = (
            f"**Search region:** {len(_active_sf)} patch{'es' if len(_active_sf) != 1 else ''} selected"
            if _active_sf else "*Click patches to restrict the search region*"
        )
        _clear_btn = mo.ui.button(label="✕ Clear", on_click=lambda _: set_ss_spatial_filter(None))
        _search_panel = mo.ui.tabs({
            "Patch Query": mo.vstack([ss_date_picker, mo.md(_label_q), ss_geo_patch_map]),
            "Search Region": mo.vstack([
                mo.hstack([mo.md(_label_s), _clear_btn], align="center"),
                ss_spatial_filter_map,
            ]),
            "Data Filter": ss_metadata_filter,
            "Settings": mo.vstack([ss_search_mode, ss_n_similar_images, ss_n_similar_patches, ss_max_gallery, ss_refine_factor, ss_similarity_toggle]),
        })
        _gallery = ss_gallery_ui if ss_gallery_ui is not None else mo.md("")
        _s = f'<div style="flex:1 1 0;min-width:0;overflow:auto;">{_search_panel.text}</div>'
        _g = (
            f'<div style="flex:1 1 0;min-width:0;">'
            f'<div style="margin-top:8px;overflow:auto;">{_gallery.text}</div>'
            f'</div>'
        )
        _items.append(mo.Html(f'<div style="display:flex;align-items:flex-start;gap:8px;">{_s}{_g}</div>'))
    elif ss_gallery_ui is not None:
        _items.append(ss_gallery_ui)
    spatial_search_tab = mo.vstack(_items)
    return (spatial_search_tab,)


@app.cell
def _(mo):
    viz_url = mo.ui.text(
        value="",
        placeholder="pelican://... or /path/to/file.nc or /path/to/folder/",
        full_width=True,
    )
    viz_load_button = mo.ui.run_button(label="▶ Load dataset")
    viz_reset_minmax = mo.ui.run_button(label="↺ Reset min/max")
    return viz_load_button, viz_reset_minmax, viz_url


@app.cell
def _(mo, viz_url):
    """If the URL is a folder, list NetCDF files in a dropdown."""
    import os as _os_fc
    _path = viz_url.value.strip()
    _nc_exts = (".nc", ".nc4", ".netcdf")
    if _path and _os_fc.path.isdir(_path):
        _files = sorted([
            f for f in _os_fc.listdir(_path)
            if f.lower().endswith(_nc_exts)
        ])
        if _files:
            viz_file_picker = mo.ui.dropdown(
                options=_files,
                value=_files[0],
                label="Select NetCDF file",
            )
        else:
            viz_file_picker = None
    else:
        viz_file_picker = None
    return (viz_file_picker,)


@app.cell
def _(mo, viz_file_picker, viz_load_button, viz_url):
    """Load dataset — triggers on button click or Enter in the URL field."""
    get_viz_ds, set_viz_ds = mo.state(None)
    get_viz_err, set_viz_err = mo.state(None)

    # mo.ui.text submits (re-runs dependents) on Enter, same as button click
    if viz_load_button.value or viz_url.value.strip():
        try:
            import xarray as _xr
            import os as _os

            _url = viz_url.value.strip()

            # ── Determine source ────────────────────────────────────────────
            _is_netcdf = False
            _nc_path   = None
            _nc_exts   = (".nc", ".nc4", ".netcdf")

            if _url and _os.path.isdir(_url):
                # Folder — use the file picker selection
                if viz_file_picker is not None and viz_file_picker.value:
                    _nc_path   = _os.path.join(_url, viz_file_picker.value)
                    _is_netcdf = True
            elif _url and (_url.endswith(_nc_exts) or _os.path.exists(_url)):
                _nc_path   = _url
                _is_netcdf = True

            if _is_netcdf:
                # ── NetCDF via xarray ────────────────────────────────────────
                _nc = _xr.open_dataset(_nc_path, engine="netcdf4")
                _data_vars = list(_nc.data_vars)
                _dims      = dict(_nc.dims)
                _coords    = dict(_nc.coords)

                # Identify x, y, z, time dimensions heuristically
                def _find_dim(candidates, dims):
                    for c in candidates:
                        for d in dims:
                            if c in d.lower():
                                return d
                    return None

                _xdim = _find_dim(["lon","x","nx"], _dims)
                _ydim = _find_dim(["lat","y","ny"], _dims)
                _zdim = _find_dim(["depth","lev","z","nz","alt"], _dims)
                _tdim = _find_dim(["time","t"], _dims)

                _nx = int(_dims[_xdim]) if _xdim else 1
                _ny = int(_dims[_ydim]) if _ydim else 1
                _nz = int(_dims[_zdim]) if _zdim else 1
                _nt = int(_dims[_tdim]) if _tdim else 1

                # Extract 1-D lat/lon coordinate arrays if available
                _lon_coord = _find_dim(["lon","longitude","x"], _coords)
                _lat_coord = _find_dim(["lat","latitude","y"], _coords)
                _lon_vals = _nc[_lon_coord].values if _lon_coord else None
                _lat_vals = _nc[_lat_coord].values if _lat_coord else None

                # Flatten to 1-D if needed (some files store 2-D coords)
                if _lon_vals is not None and _lon_vals.ndim > 1:
                    _lon_vals = _lon_vals[0]  # take first row
                if _lat_vals is not None and _lat_vals.ndim > 1:
                    _lat_vals = _lat_vals[:, 0]  # take first column

                # Extract actual time coordinate values if available
                _time_vals = None
                if _tdim and _tdim in _nc.coords:
                    import pandas as _pd
                    try:
                        _tv = _nc[_tdim].values
                        _time_vals = [_pd.Timestamp(_t).strftime("%Y-%m-%d %H:%M") for _t in _tv]
                    except Exception:
                        _time_vals = [str(i) for i in range(_nt)]
                else:
                    _time_vals = [str(i) for i in range(_nt)]

                set_viz_ds({
                    "kind":      "netcdf",
                    "ds":        _nc,
                    "path":      _nc_path,
                    "url":       _nc_path,
                    "fields":    _data_vars,
                    "steps":     list(range(_nt)),
                    "time_vals": _time_vals,  # list of date strings
                    "box":       [[0, 0, 0], [_nx, _ny, _nz]],
                    "maxres":    32,
                    "dims":      {"x": _xdim, "y": _ydim, "z": _zdim, "t": _tdim},
                    "lon_vals":  _lon_vals,
                    "lat_vals":  _lat_vals,
                })
                set_viz_err(None)

            elif _url:
                # ── Remote OpenVisus dataset ─────────────────────────────────
                import openvisuspy as _ovp_viz
                _ds = _ovp_viz.LoadDataset(_url)
                _box   = _ds.getLogicBox()
                _steps = _ds.getTimesteps()
                _raw_fields = _ds.getFields()
                _fields = [f.name if hasattr(f, "name") else str(f) for f in _raw_fields]
                set_viz_ds({
                    "kind":   "ovp",
                    "ds":     _ds,
                    "url":    _url,
                    "box":    _box,
                    "steps":  _steps,
                    "fields": _fields,
                    "maxres": _ds.getMaxResolution(),
                })
                set_viz_err(None)
            else:
                set_viz_err("Please enter a URL or upload a file.")

        except Exception as _e:
            set_viz_ds(None)
            set_viz_err(str(_e))
    return get_viz_ds, get_viz_err


@app.cell
def _(get_viz_ds, mo):
    """Build controls once a dataset is loaded."""
    _meta = get_viz_ds()
    if _meta is None:
        viz_timestep   = mo.ui.text(value="", placeholder="YYYY-MM-DD HH:MM")
        viz_depth      = mo.ui.slider(start=0, stop=0,  value=0,  show_value=True)
        viz_resolution = mo.ui.slider(start=0, stop=40, value=28, show_value=True)
        viz_quality    = mo.ui.slider(start=-8, stop=0, value=-1, show_value=True)
        viz_x          = mo.ui.range_slider(start=0, stop=100, value=[40, 60], show_value=True)
        viz_y          = mo.ui.range_slider(start=0, stop=100, value=[40, 60], show_value=True)
        viz_field      = mo.ui.dropdown(options=["(none)"], value="(none)")
        viz_colormap   = mo.ui.dropdown(
            options=["viridis","plasma","inferno","magma","cividis","RdBu_r","coolwarm","turbo"],
            value="magma",
        )
    else:
        _steps  = _meta["steps"]
        _box    = _meta["box"]
        _maxres = _meta["maxres"]
        _z_max  = max(0, _box[1][2] - 1)
        _nx     = int(_box[1][0])
        _ny     = int(_box[1][1])
        # Default to center 20%
        _cx0, _cx1 = int(_nx * 0.4), int(_nx * 0.6)
        _cy0, _cy1 = int(_ny * 0.4), int(_ny * 0.6)
        _time_vals = _meta.get("time_vals")
        _kind_ctrl_t = _meta.get("kind", "ovp")
        if _kind_ctrl_t == "netcdf" and _time_vals:
            viz_timestep = mo.ui.text(
                value=_time_vals[0],
                placeholder="YYYY-MM-DD HH:MM",
            )
        else:
            viz_timestep = mo.ui.slider(
                start=int(_steps[0]), stop=int(_steps[-1]),
                value=int(_steps[0]),
                step=int(_steps[1] - _steps[0]) if len(_steps) > 1 else 1, show_value=True,
            )
        viz_depth = mo.ui.slider(
            start=0, stop=int(_z_max), value=0, show_value=True,
        )
        viz_resolution = mo.ui.slider(
            start=0, stop=int(_maxres), value=min(28, int(_maxres)), show_value=True,
        )
        viz_quality = mo.ui.slider(
            start=-8, stop=0, value=-1, show_value=True,
        )
        _lon_vals = _meta.get("lon_vals")
        _lat_vals = _meta.get("lat_vals")
        _kind_ctrl = _meta.get("kind", "ovp")
        if _kind_ctrl == "netcdf" and _lon_vals is not None and _lat_vals is not None:
            _lon_min, _lon_max = float(_lon_vals.min()), float(_lon_vals.max())
            _lat_min, _lat_max = float(_lat_vals.min()), float(_lat_vals.max())
            _lon_step = max(round((_lon_max - _lon_min) / _nx, 4), 0.0001)
            _lat_step = max(round((_lat_max - _lat_min) / _ny, 4), 0.0001)
            _lon_c0 = max(_lon_min, 260)
            _lon_c1 = min(_lon_max, 330)
            if _lon_c1 <= _lon_c0:
                _lon_c0 = _lon_min + (_lon_max - _lon_min) * 0.4
                _lon_c1 = _lon_min + (_lon_max - _lon_min) * 0.6
            _lat_c0 = max(_lat_min, 10)
            _lat_c1 = min(_lat_max, 40)
            if _lat_c1 <= _lat_c0:
                _lat_c0 = _lat_min + (_lat_max - _lat_min) * 0.4
                _lat_c1 = _lat_min + (_lat_max - _lat_min) * 0.6
            viz_x = mo.ui.range_slider(
                start=round(_lon_min, 4), stop=round(_lon_max, 4),
                value=[round(_lon_c0, 4), round(_lon_c1, 4)],
                step=_lon_step, show_value=True,
            )
            viz_y = mo.ui.range_slider(
                start=round(_lat_min, 4), stop=round(_lat_max, 4),
                value=[round(_lat_c0, 4), round(_lat_c1, 4)],
                step=_lat_step, show_value=True,
            )
        else:
            viz_x = mo.ui.range_slider(
                start=0, stop=_nx, value=[_cx0, _cx1], show_value=True,
            )
            viz_y = mo.ui.range_slider(
                start=0, stop=_ny, value=[_cy0, _cy1], show_value=True,
            )
        viz_field = mo.ui.dropdown(
            options=_meta["fields"], value=_meta["fields"][0],
        )
        viz_colormap = mo.ui.dropdown(
            options=["viridis","plasma","inferno","magma","cividis","RdBu_r","coolwarm","turbo"],
            value="magma",
        )
    return (
        viz_colormap,
        viz_depth,
        viz_field,
        viz_quality,
        viz_resolution,
        viz_timestep,
        viz_x,
        viz_y,
    )


@app.cell
def _(get_viz_ds, mo, np, viz_depth, viz_field, viz_timestep):
    """Compute min/max for the selected timestep and depth when field/date/reset changes."""
    get_viz_vmin, set_viz_vmin = mo.state(0.0)
    get_viz_vmax, set_viz_vmax = mo.state(1.0)

    _meta = get_viz_ds()
    if _meta is not None and _meta.get("kind") == "netcdf":
        try:
            _dims      = _meta["dims"]
            _time_vals = _meta.get("time_vals")
            _da        = _meta["ds"][viz_field.value]

            # Select the current timestep
            _sel = {}
            if _dims["t"]:
                if _time_vals and isinstance(viz_timestep.value, str):
                    _tv = viz_timestep.value.strip()
                    if _tv in _time_vals:
                        _sel[_dims["t"]] = _time_vals.index(_tv)
                    else:
                        _sel = None  # invalid date — skip
                else:
                    try:
                        _sel[_dims["t"]] = int(viz_timestep.value)
                    except (ValueError, TypeError):
                        _sel = None
            if _dims["z"]:
                _sel[_dims["z"]] = int(viz_depth.value)

            if _sel is not None:
                _slice = _da.isel(**_sel).values
                set_viz_vmin(round(float(np.nanpercentile(_slice, 2)), 4))
                set_viz_vmax(round(float(np.nanpercentile(_slice, 98)), 4))
        except Exception:
            pass
    return get_viz_vmax, get_viz_vmin


@app.cell
def _(get_viz_vmax, get_viz_vmin, mo):
    """Number fields initialised from state; user edits trigger re-render."""
    viz_vmin = mo.ui.number(value=get_viz_vmin(), label="Min")
    viz_vmax = mo.ui.number(value=get_viz_vmax(), label="Max")
    return viz_vmax, viz_vmin


@app.cell
def _(
    get_viz_ds,
    get_viz_err,
    map_theme,
    mo,
    np,
    plt,
    viz_colormap,
    viz_depth,
    viz_field,
    viz_file_picker,
    viz_load_button,
    viz_quality,
    viz_reset_minmax,
    viz_resolution,
    viz_timestep,
    viz_url,
    viz_vmax,
    viz_vmin,
    viz_x,
    viz_y,
):
    """Render a horizontal 2-D slice — auto-updates on any control change."""
    _meta = get_viz_ds()
    _err  = get_viz_err()
    _theme = "dark" if map_theme.value else "light"
    _colors = get_theme_colors(_theme)

    _picker_row = (
        mo.hstack([mo.md("**File:**"), viz_file_picker], justify="start")
        if viz_file_picker is not None else None
    )
    _header = mo.vstack([
        mo.Html(
            '<div style="display:flex;gap:8px;align-items:flex-end;width:100%">'
            + f'<div style="flex:3;min-width:0">{viz_url._repr_html_()}</div>'
            + f'<div style="flex:0 0 auto">{viz_load_button._repr_html_()}</div>'
            + '</div>'
        ),
        *([_picker_row] if _picker_row is not None else []),
    ])

    if _err is not None:
        visualize_tab = mo.vstack([
            _header,
            mo.callout(mo.md(f"**Load error:** `{_err}`"), kind="danger"),
        ])
    elif _meta is None:
        visualize_tab = mo.vstack([
            _header,
            mo.callout(
                mo.md(
                    "Enter a folder pathway to ERA5 NetCDF files\n"
                    # "Enter a remote Pelican dataset URL, a path to a local NetCDF file, or a folder to browse files.\n\n"
                    # "**Example datasets:**\n"
                    # "- `pelican://osg-htc.org/nasa/nsdf/climate1/llc4320/idx/theta/theta_llc4320_x_y_depth.idx` — Temperature\n"
                    # "- `pelican://osg-htc.org/nasa/nsdf/climate1/llc4320/idx/salt/salt_llc4320_x_y_depth.idx` — Salinity\n"
                    # "- `pelican://osg-htc.org/nasa/nsdf/climate2/llc4320/idx/w/w_llc4320_x_y_depth.idx` — Vertical velocity"
                ),
                kind="info",
            ),
        ])
    else:
        _box    = _meta["box"]
        _steps  = _meta["steps"]
        _fields = _meta["fields"]
        _maxres = _meta["maxres"]
        _nx, _ny, _nz = _box[1][0], _box[1][1], _box[1][2]

        _kind = _meta.get("kind", "ovp")
        _time_vals_info = _meta.get("time_vals")
        if _kind == "ovp":
            _extra = f"  ·  **Max resolution:** {_maxres}"
        elif _time_vals_info:
            _extra = f"  ·  **Dates:** {_time_vals_info[0]} → {_time_vals_info[-1]}"
        else:
            _extra = "  ·  **Source:** NetCDF"
        _info_md = mo.md(
            f"**Dimensions:** {_nx} × {_ny} × {_nz}  ·  "
            f"**Timesteps:** {len(_steps)}  ·  "
            f"**Fields:** {', '.join(_fields)}"
            + _extra
        )

        def _labeled(label, widget):
            return mo.vstack([mo.md(f"**{label}**"), widget], gap=0)

        if _kind == "netcdf":
            _controls = mo.vstack([
                mo.hstack([
                    _labeled("Variable", viz_field),
                    _labeled("Date (YYYY-MM-DD HH:MM)", viz_timestep),
                    _labeled("Colormap", viz_colormap),
                    _labeled("Detail", viz_quality),
                ], justify="start"),
                mo.hstack([
                    _labeled("Longitude range", viz_x),
                    _labeled("Latitude range", viz_y),
                    _labeled("Depth (z)", viz_depth),
                ], justify="start"),
                mo.hstack([
                    _labeled("Min value", viz_vmin),
                    _labeled("Max value", viz_vmax),
                    _labeled(" ", viz_reset_minmax),
                ], justify="start"),
            ])
        else:
            _controls = mo.vstack([
                mo.hstack([
                    _labeled("Variable", viz_field),
                    _labeled("Timestep", viz_timestep),
                    _labeled("Colormap", viz_colormap),
                    _labeled("Detail", viz_quality),
                ], justify="start"),
                mo.hstack([
                    _labeled("X range", viz_x),
                    _labeled("Y range", viz_y),
                    _labeled("Depth (z)", viz_depth),
                ], justify="start"),
                mo.hstack([
                    _labeled("Min value", viz_vmin),
                    _labeled("Max value", viz_vmax),
                    _labeled(" ", viz_reset_minmax),
                ], justify="start"),
                mo.hstack([
                    _labeled("Base resolution", viz_resolution),
                ], justify="start"),
            ])

        try:
            _ds   = _meta["ds"]
            _kind = _meta.get("kind", "ovp")
            _res  = int(viz_resolution.value)
            _q    = int(viz_quality.value)
            _z    = int(viz_depth.value)
            _x0, _x1 = int(viz_x.value[0]), int(viz_x.value[1])
            _y0, _y1 = int(viz_y.value[0]), int(viz_y.value[1])
            if _x1 <= _x0: _x1 = _x0 + 1
            if _y1 <= _y0: _y1 = _y0 + 1

            if _kind == "netcdf":
                # ── NetCDF slice via xarray ──────────────────────────────────
                _dims     = _meta["dims"]
                _lon_vals = _meta.get("lon_vals")
                _lat_vals = _meta.get("lat_vals")
                _var      = _ds[viz_field.value]
                _sel      = {}
                if _dims["t"]:
                    _time_vals = _meta.get("time_vals")
                    if _time_vals and isinstance(viz_timestep.value, str):
                        _tv = viz_timestep.value.strip()
                        if _tv not in _time_vals:
                            raise ValueError(f"Date '{_tv}' not found in dataset. Available range: {_time_vals[0]} → {_time_vals[-1]}")
                        _t_idx = _time_vals.index(_tv)
                    else:
                        _t_idx = int(viz_timestep.value)
                    _sel[_dims["t"]] = _t_idx
                if _dims["z"]:
                    _sel[_dims["z"]] = _z
                _var2d = _var.isel(**_sel) if _sel else _var

                # Convert lon/lat slider values back to index ranges
                if _lon_vals is not None and _lat_vals is not None:
                    # Handle both ascending and descending coordinate arrays
                    def _coord_slice(arr, lo, hi):
                        if arr[-1] > arr[0]:  # ascending
                            i0 = int(np.searchsorted(arr, lo, side='left'))
                            i1 = int(np.searchsorted(arr, hi, side='right'))
                        else:  # descending — flip, search, flip back
                            i0 = len(arr) - int(np.searchsorted(arr[::-1], hi, side='left'))
                            i1 = len(arr) - int(np.searchsorted(arr[::-1], lo, side='right'))
                        return max(0, i0), min(len(arr), max(i1, i0 + 1))
                    _xi0, _xi1 = _coord_slice(_lon_vals, viz_x.value[0], viz_x.value[1])
                    _yi0, _yi1 = _coord_slice(_lat_vals, viz_y.value[0], viz_y.value[1])
                else:
                    _xi0, _xi1 = _x0, _x1
                    _yi0, _yi1 = _y0, _y1

                if _dims["y"]:
                    _var2d = _var2d.isel({_dims["y"]: slice(_yi0, _yi1)})
                if _dims["x"]:
                    _var2d = _var2d.isel({_dims["x"]: slice(_xi0, _xi1)})
                _slice = _var2d.values.squeeze()

                # Downsample to approximate quality level (each -1 halves resolution)
                _step = max(1, 2 ** (-_q))
                _slice = _slice[::_step, ::_step]

                # Coordinate arrays for this slice, downsampled to match
                if _lon_vals is not None and _lat_vals is not None:
                    _plot_lon = _lon_vals[_xi0:_xi1][::_step]
                    _plot_lat = _lat_vals[_yi0:_yi1][::_step]
                else:
                    _plot_lon = None
                    _plot_lat = None
            else:
                # ── Remote OpenVisus dataset ─────────────────────────────────
                _reader = _ds.db if hasattr(_ds, "db") and hasattr(_ds.db, "read") else _ds
                if not hasattr(_reader, "read"):
                    raise AttributeError(
                        f"Cannot find read() on {type(_ds).__name__}. "
                        f"Available: {[m for m in dir(_ds) if not m.startswith('_')]}"
                    )
                _data = _reader.read(
                    logic_box=[[_x0, _y0, _z], [_x1, _y1, _z + 1]],
                    field=viz_field.value,
                    time=int(viz_timestep.value),
                    max_resolution=_res,
                    quality=_q,
                )
                if not isinstance(_data, np.ndarray):
                    _data = next(iter(_data))
                while isinstance(_data, np.ndarray) and _data.ndim > 2 and _data.shape[0] == 1:
                    _data = _data[0]
                _slice = _data

            _bg = _colors["bg"]
            _txt = _colors["text"]

            _fig, _ax = plt.subplots(figsize=(10, 5))
            _fig.patch.set_facecolor(_bg)
            _ax.set_facecolor(_bg)

            _vmin = float(viz_vmin.value) if viz_vmin.value is not None else float(np.nanpercentile(_slice, 2))
            _vmax = float(viz_vmax.value) if viz_vmax.value is not None else float(np.nanpercentile(_slice, 98))
            if _vmax <= _vmin:
                _vmax = _vmin + 1.0

            if _kind == "netcdf" and _plot_lon is not None and _plot_lat is not None:
                import matplotlib.colors as _mcolors
                import matplotlib.patches as _mpatches
                import cartopy.io.shapereader as _shpreader
                import shapely.vectorized as _shvec
                from shapely.ops import unary_union as _unary_union

                # ── Fixed-size figure with plain axes ────────────────────────
                plt.close(_fig)
                _fig, _ax = plt.subplots(figsize=(10, 5), dpi=100)
                _fig.patch.set_facecolor(_bg)
                _ax.set_facecolor("#444444")  # ocean — dark gray
                _ax.set_xlim(_plot_lon.min(), _plot_lon.max())
                _ax.set_ylim(_plot_lat.min(), _plot_lat.max())
                _ax.set_aspect("auto")  # fill the fixed figure size

                # ── Draw land polygons from Natural Earth ─────────────────────
                _land_shp = _shpreader.natural_earth(
                    resolution="110m", category="physical", name="land"
                )
                for _geom in _shpreader.Reader(_land_shp).geometries():
                    _ax.add_patch(_mpatches.PathPatch(
                        plt.matplotlib.path.Path.make_compound_path(
                            *[plt.matplotlib.path.Path(
                                np.column_stack([np.array(p.exterior.coords)[:, 0],
                                                 np.array(p.exterior.coords)[:, 1]])
                            ) for p in (
                                [_geom] if _geom.geom_type == "Polygon"
                                else list(_geom.geoms)
                            )]
                        ),
                        facecolor="#888888", edgecolor="none", zorder=1,
                        transform=_ax.transData,
                    ))

                # ── Data with linear alpha ────────────────────────────────────
                _norm = _mcolors.Normalize(vmin=_vmin, vmax=_vmax)
                _cmap_base = plt.get_cmap(viz_colormap.value)
                _colors_rgba = _cmap_base(np.linspace(0, 1, 256))
                _colors_rgba[:, 3] = np.linspace(0, 1, 256)
                _cmap_alpha = _mcolors.ListedColormap(_colors_rgba)

                _im = _ax.pcolormesh(
                    _plot_lon, _plot_lat, _slice,
                    cmap=_cmap_alpha, norm=_norm,
                    shading="auto", zorder=2,
                )

                # ── Lat/lon tick labels ───────────────────────────────────────
                _ax.set_xlabel("Longitude", color=_txt)
                _ax.set_ylabel("Latitude", color=_txt)
                _title_loc = (f"lon=[{viz_x.value[0]:.2f},{viz_x.value[1]:.2f}]  "
                              f"lat=[{viz_y.value[0]:.2f},{viz_y.value[1]:.2f}]")
            else:
                # Normalize to uint8 for faster matplotlib rendering
                _range = _vmax - _vmin if _vmax > _vmin else 1.0
                _slice_8 = np.clip((_slice - _vmin) / _range * 255, 0, 255).astype(np.uint8)
                _im = _ax.imshow(
                    _slice_8,
                    origin="lower",
                    cmap=viz_colormap.value,
                    aspect="auto",
                    vmin=0, vmax=255,
                    extent=[_x0, _x1, _y0, _y1],
                )
                _ax.set_xlabel("X", color=_txt)
                _ax.set_ylabel("Y", color=_txt)
                _title_loc = f"x=[{_x0},{_x1}]  y=[{_y0},{_y1}]"

            _cbar = _fig.colorbar(_im, ax=_ax, fraction=0.03, pad=0.02)
            _cbar.ax.set_facecolor("#444444")
            _cbar.ax.yaxis.set_tick_params(color=_txt)
            plt.setp(_cbar.ax.yaxis.get_ticklabels(), color=_txt)
            _cbar.set_label(viz_field.value, color=_txt)

            _ax.set_title(
                f"{viz_field.value}  ·  {viz_timestep.value}  ·  z={_z}  ·  "
                f"{_title_loc}  ·  q={_q}",
                color=_txt,
            )
            _ax.tick_params(colors=_txt)
            for _spine in _ax.spines.values():
                _spine.set_edgecolor(_colors["border"])

            if _kind != "netcdf":
                _fig.tight_layout()
            _plot_html = mo.as_html(_fig)
            plt.close(_fig)

            _shape_note = mo.md(
                f"Slice shape: **{_slice.shape[1]} × {_slice.shape[0]}**  ·  "
                f"min {float(np.nanmin(_slice)):.4g}  ·  "
                f"max {float(np.nanmax(_slice)):.4g}  ·  "
                f"mean {float(np.nanmean(_slice)):.4g}"
            )

            visualize_tab = mo.vstack([
                _header, _info_md, _controls, _shape_note, _plot_html,
            ])

        except Exception as _render_err:
            visualize_tab = mo.vstack([
                _header, _info_md, _controls,
                mo.callout(mo.md(f"**Render error:** `{_render_err}`"), kind="danger"),
            ])
    return (visualize_tab,)


@app.cell
def _(mo):
    audit_tab = mo.callout(
        mo.md("**Audit** — coming soon."), kind="neutral"
    )
    return (audit_tab,)


@app.cell
def _(
    audit_tab,
    dim_reduction_tab,
    explore_tab,
    mo,
    spatial_search_tab,
    visualize_tab,
):
    mo.ui.tabs({
        "Data": explore_tab,
        "Clustering": dim_reduction_tab,
        "Spatial Search": spatial_search_tab,
        "Visualize": visualize_tab,
        "Audit": audit_tab,
    })
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
