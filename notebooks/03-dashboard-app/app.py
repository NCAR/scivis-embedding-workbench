import marimo

__generated_with = "0.20.4"
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
    """Scan a LanceDB directory for experiment names by finding *_config.lance dirs."""
    from pathlib import Path
    p = Path(db_path)
    if not p.exists() or not p.is_dir():
        return []
    return [
        d.name[: -len("_config.lance")]
        for d in sorted(p.iterdir())
        if d.is_dir() and d.name.endswith("_config.lance")
    ]


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
def make_extent_map(lat_min, lat_max, lon_min, lon_max, num_patch_tokens, patch_size=16, theme="light", experiment=""):
    """Cartopy map cropped to spatial extent with patch grid line overlay."""
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import math
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
    n_side = int(math.sqrt(num_patch_tokens))
    img_size = n_side * patch_size
    lat_step = (lat_max - lat_min) / n_side
    lon_step = (lon_max - lon_min) / n_side
    for i in range(1, n_side):
        ax.plot([lon_min, lon_max], [lat_min + i * lat_step] * 2,
                transform=proj, color=colors["grid"], linewidth=0.4, zorder=3)
    for j in range(1, n_side):
        ax.plot([lon_min + j * lon_step] * 2, [lat_min, lat_max],
                transform=proj, color=colors["grid"], linewidth=0.4, zorder=3)
    _title = f"{img_size}×{img_size}px  |  {n_side}×{n_side} patch grid ({num_patch_tokens} patches)"
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
        db = lancedb.connect(embedding_db_path.value)
        config = load_config_dict(db, f"{experiment_selector.value}_config")
        _source_path = resolve_source_path(embedding_db_path.value, config.get("source_path", ""))
        if _source_path:
            source_db = lancedb.connect(_source_path)
            src_img_tbl = source_db.open_table(config["source"])
        else:
            src_img_tbl = None
        img_emb_tbl = db.open_table(config["tbl_img_emb"])
        patch_emb_tbl = db.open_table(config["tbl_patch_emb"])
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
    return (overview_tab,)


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
        if _ext and "num_patch_tokens" in config:
            _map_fig = make_extent_map(
                lat_min=float(_ext["lat_min"]), lat_max=float(_ext["lat_max"]),
                lon_min=float(_ext["lon_min"]), lon_max=float(_ext["lon_max"]),
                num_patch_tokens=int(config["num_patch_tokens"]),
                patch_size=int(config.get("patch_size", 16)),
                theme="dark" if map_theme.value else "light",
                experiment=experiment_selector.value,
            )
            _map_subtab = mo.as_html(_map_fig)
            plt.close(_map_fig)
        else:
            _map_subtab = mo.callout(mo.md("Spatial extent or `num_patch_tokens` not available."), kind="warn")

        explore_tab = mo.vstack([
            mo.hstack([embedding_db_path, experiment_selector, map_theme], justify="start"),
            mo.ui.tabs({"Source": _src_subtab, "Experiment": _exp_subtab, "Map": _map_subtab}),
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
        tooltip="Run Principal Component Analysis on loaded embeddings",
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
def _(mo, pca_tab):
    _umap_placeholder = mo.callout(
        mo.md("UMAP dimensionality reduction — coming soon."), kind="neutral"
    )
    dim_reduction_tab = mo.ui.tabs({"PCA": pca_tab, "UMAP": _umap_placeholder})
    return (dim_reduction_tab,)


@app.cell
def _(mo):
    spatial_search_tab = mo.callout(
        mo.md("**Spatial Search** — coming soon."), kind="neutral"
    )
    return (spatial_search_tab,)


@app.cell
def _(mo):
    visualize_tab = mo.callout(
        mo.md("**Visualize** — coming soon."), kind="neutral"
    )
    return (visualize_tab,)


@app.cell
def _(mo):
    def _chat_model(messages, config):
        return "Chat functionality coming soon."

    chat_tab = mo.ui.chat(_chat_model, prompts=["Tell me about this dataset"])
    return (chat_tab,)


@app.cell
def _(
    chat_tab,
    dim_reduction_tab,
    explore_tab,
    mo,
    overview_tab,
    spatial_search_tab,
    visualize_tab,
):
    mo.ui.tabs({
        "Overview": overview_tab,
        "Data": explore_tab,
        "Dimensionality Reduction": dim_reduction_tab,
        "Spatial Search": spatial_search_tab,
        "Visualize": visualize_tab,
        "Chat": chat_tab,
    })
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
