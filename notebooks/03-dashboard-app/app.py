import marimo

__generated_with = "0.20.4"
app = marimo.App(layout_file="layouts/app.grid.json")


@app.cell
def _():
    import marimo as mo
    import numpy as np
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
    is_dark = theme == "dark"
    colors = (
        {"land": "#3a3a3a", "ocean": "#1e3a5f", "coast": "#aaaaaa",
         "grid": "#666666", "text": "#e0e0e0", "bg": "#1a1a1a"}
        if is_dark else
        {"land": "#d4d4d4", "ocean": "#a8c8e8", "coast": "#555555",
         "grid": "#888888", "text": "#222222", "bg": "#ffffff"}
    )
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

    is_dark = theme == "dark"
    plotly_template = "plotly_dark" if is_dark else "plotly_white"

    cum = np.cumsum(evr) * 100
    per_comp = evr * 100
    components = np.arange(1, len(evr) + 1)
    _sample_label = f"{n_used:,} / {n_total:,}" if n_used < n_total else f"{n_total:,}"

    fig = go.Figure()

    bar_color = "#4FC3F7" if is_dark else "#1565C0"
    line_color = "#FF7043" if is_dark else "#C62828"

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

    _lx = int(components[-1])
    _lper = float(per_comp[-1])
    _lcum = float(cum[-1])
    _ann_bg = "rgba(30,30,30,0.85)" if is_dark else "rgba(255,255,255,0.88)"
    _ann_fg = "#e0e0e0" if is_dark else "#222222"
    fig.add_annotation(
        x=_lx, y=_lcum, yref="y2",
        text=f"PC {_lx}<br>{_lper:.2f}% var<br>{_lcum:.1f}% cum",
        showarrow=True, arrowhead=2, arrowwidth=1.5,
        arrowcolor=line_color,
        ax=-50, ay=-36,
        font=dict(size=10, color=_ann_fg),
        bgcolor=_ann_bg,
        bordercolor=line_color,
        borderwidth=1,
        borderpad=4,
        xanchor="right",
        yanchor="bottom",
    )

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
    map_theme = mo.ui.radio(
        options=["system", "light", "dark"],
        value=mo.app_meta().theme,
        label="Map theme",
        inline=True,
    )
    embedding_db_path = mo.ui.text(
        value="",
        placeholder="e.g. /data/lancedb/experiments/era5",
        label="Experiments DB path",
        full_width=True,
    )
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
                theme=map_theme.value,
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
def _(explore_tab, mo, overview_tab):
    mo.ui.tabs({
        "🏠 Overview": overview_tab,
        "🔍 Explore": explore_tab,
    })
    return


@app.cell
def _(mo):
    get_pca, set_pca = mo.state(None)
    return get_pca, set_pca


@app.cell
def _(mo):
    n_vectors = mo.ui.slider(2, 5000, value=1000, step=100, label="Max vectors")
    run_pca = mo.ui.run_button(label="Run PCA")
    mo.hstack([n_vectors, run_pca], justify="start")
    return n_vectors, run_pca


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
        mo.callout(mo.md("Configure options above and click Run PCA."), kind="info")
    elif img_emb_tbl is None:
        mo.callout(mo.md("No experiment loaded — enter a DB path above."), kind="warn")
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
    return


@app.cell
def _(get_pca, mo):
    _r = get_pca()
    mo.stop(_r is None)
    n_components = mo.ui.slider(
        1, len(_r["evr"]), value=len(_r["evr"]), step=1, label="Components to show",
    )
    n_components
    return (n_components,)


@app.cell
def _(get_pca, map_theme, mo, n_components):
    _r = get_pca()
    _fig = make_scree_plot(
        _r["evr"][:n_components.value],
        _r["n_total"], _r["emb_dim"], _r["n_used"], _r["backend"], map_theme.value,
    )
    mo.as_html(_fig)
    return


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
        vals = np.asarray(data_cols[axis_name], dtype=float)
        lo = min(info["range"])
        hi = max(info["range"])
        axis_mask = (vals >= lo) & (vals <= hi)
        if info.get("include_infnans", False):
            axis_mask |= ~np.isfinite(vals)
        mask &= axis_mask
    return np.where(mask)[0].tolist()


@app.function
def fetch_thumbnails_batch(src_img_tbl, image_ids: list, max_images: int = 50):
    """Batch-fetch thumbnail blobs, filenames, and timestamps by image id."""
    if not image_ids or src_img_tbl is None:
        return []
    image_ids = image_ids[:max_images]
    escaped = ", ".join(f"'{i}'" for i in image_ids)
    df = (
        src_img_tbl.search()
        .where(f"id IN ({escaped})")
        .select(["id", "filename", "thumb_blob", "dt"])
        .limit(max_images)
        .to_pandas()
    )
    return list(zip(df["filename"], df["thumb_blob"], df["dt"]))


@app.function
def render_thumbnail_gallery(thumbs, n_filtered, max_display, theme="light"):
    """Build HTML for a theme-aware thumbnail gallery with datetime labels."""
    import base64

    is_dark = theme == "dark"
    bg = "rgba(30,30,30,0.85)" if is_dark else "#ffffff"
    text = "#e0e0e0" if is_dark else "#222222"
    border = "#444444" if is_dark else "#cccccc"

    imgs = []
    for fname, blob, dt in thumbs:
        b64 = base64.b64encode(blob).decode()
        dt_str = dt.strftime("%Y-%m-%d %H:%M") if hasattr(dt, "strftime") else str(dt)
        imgs.append(
            f'<div style="display:inline-block;margin:3px;text-align:center">'
            f'<img src="data:image/jpeg;base64,{b64}" '
            f'style="width:64px;height:64px;border:1px solid {border};'
            f'border-radius:4px" title="{fname}"/>'
            f'<div style="font-size:9px;color:{text};max-width:64px;'
            f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap">'
            f'{dt_str}</div></div>'
        )

    count_msg = f"Showing {len(thumbs)} of {n_filtered} selected"
    if n_filtered > max_display:
        count_msg += f" (capped at {max_display})"

    gallery_html = (
        f'<div style="display:flex;flex-wrap:wrap;gap:4px;'
        f'max-height:300px;overflow-y:auto;background:{bg};'
        f'border-radius:8px;padding:8px;border:1px solid {border}">'
        + "".join(imgs)
        + "</div>"
    )
    return count_msg, gallery_html


@app.cell
def _(get_pca, mo):
    _r = get_pca()
    mo.stop(_r is None)
    _default = ",".join(str(i + 1) for i in range(min(6, len(_r["evr"]))))
    pc_axes_input = mo.ui.text(
        value=_default,
        label="PCs to display (comma-separated)",
    )
    pc_axes_input
    return (pc_axes_input,)


@app.cell(hide_code=True)
def _(get_pca, mo, np):


    _r = get_pca()
    mo.stop(_r is None)

    # Build options: datetime components + any numeric source columns
    _dt_options = ["dt:year", "dt:month", "dt:day", "dt:hour", "dt:minute", "dt:second"]
    _extra_options = list(_dt_options)
    _meta = _r.get("metadata")
    if _meta is not None and len(_meta) > 0:
        _skip = {"id", "filename", "dt"}
        for _col in _meta.columns:
            if _col not in _skip and np.issubdtype(_meta[_col].dtype, np.number):
                _extra_options.append(_col)

    extra_axes_select = mo.ui.multiselect(
        options=_extra_options,
        value=["dt:month"],
        label="Extra axes",
    )
    normalize_toggle = mo.ui.switch(value=False, label="Equal PC axis range")
    mo.hstack([extra_axes_select, normalize_toggle], justify="start")
    return extra_axes_select, normalize_toggle


@app.cell(hide_code=True)
def _(
    ParallelCoordinates,
    extra_axes_select,
    get_pca,
    mo,
    normalize_toggle,
    np,
    pc_axes_input,
    pl,
):


    _r = get_pca()
    mo.stop(_r is None)

    # Parse PC indices
    _max_pc = _r["scores"].shape[1]
    _indices = []
    for _tok in pc_axes_input.value.split(","):
        _tok = _tok.strip()
        if _tok.isdigit() and 1 <= int(_tok) <= _max_pc:
            _indices.append(int(_tok) - 1)
    mo.stop(
        len(_indices) == 0,
        mo.callout(mo.md("Enter valid PC numbers."), kind="warn"),
    )

    # Build extra columns from metadata
    _meta = _r.get("metadata")
    _extra_cols = {}
    if _meta is not None and len(_meta) > 0:
        for _sel in extra_axes_select.value:
            if _sel.startswith("dt:") and "dt" in _meta.columns:
                _part = _sel.split(":")[1]
                if hasattr(_meta["dt"].dt, _part):
                    _vals = getattr(_meta["dt"].dt, _part).astype(float)
                    _n_nan = int(_vals.isna().sum())
                    if _n_nan > 0:
                        print(f"[parcoord] WARNING: {_sel} has {_n_nan} NaN values "
                              f"out of {len(_vals)} rows (missing metadata)")
                    _extra_cols[_sel] = _vals.values
            elif _sel in _meta.columns:
                _extra_cols[_sel] = _meta[_sel].astype(float).values

    # Build PC columns
    _scores = _r["scores"]
    _pc_values = {f"PC{i + 1}": _scores[:, i] for i in _indices}

    # Sentinel rows to equalize PC axis ranges (values stay untouched)
    _sentinels = None
    if normalize_toggle.value:
        _all_vals = np.column_stack([_scores[:, i] for i in _indices])
        _gmin = float(_all_vals.min())
        _gmax = float(_all_vals.max())
        _sentinel_lo = {k: _gmin for k in _pc_values}
        _sentinel_hi = {k: _gmax for k in _pc_values}
        for k, v in _extra_cols.items():
            _sentinel_lo[k] = float(np.nanmin(v))
            _sentinel_hi[k] = float(np.nanmax(v))
        _sentinels = pl.DataFrame([_sentinel_lo, _sentinel_hi])

    # Combine: extra columns first, then PCs
    parcoord_data_cols = {**_extra_cols, **_pc_values}
    _df = pl.DataFrame(parcoord_data_cols)
    if _sentinels is not None:
        _df = pl.concat([_df, _sentinels], how="diagonal_relaxed")
    parcoord_widget = mo.ui.anywidget(ParallelCoordinates(_df, height=350))
    parcoord_widget
    return parcoord_data_cols, parcoord_widget


@app.cell(hide_code=True)
def _(
    get_pca,
    map_theme,
    mo,
    parcoord_data_cols,
    parcoord_widget,
    src_img_tbl,
):
    _r = get_pca()
    mo.stop(_r is None or src_img_tbl is None)

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
    _max_display = 50
    _thumbs = fetch_thumbnails_batch(src_img_tbl, _selected_ids, _max_display)
    _count_msg, _gallery_html = render_thumbnail_gallery(
        _thumbs, len(_filtered), _max_display, theme=map_theme.value,
    )
    mo.vstack([mo.md(f"**{_count_msg}**"), mo.Html(_gallery_html)])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
