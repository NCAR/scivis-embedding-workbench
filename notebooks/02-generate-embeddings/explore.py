import marimo

__generated_with = "0.20.4"
app = marimo.App(layout_file="layouts/explore.grid.json")


@app.cell
def _():
    import marimo as mo
    import lancedb
    import pandas as pd
    import io
    import math
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw

    return Image, io, lancedb, mo, plt


@app.cell
def _(mo):
    # --- DB PATH CONFIGURATION ---
    embedding_db_path = mo.ui.text(
        value="/Users/ncheruku/Documents/Work/sample_data/data/lancedb/experiments/era5",
        label="Experiments DB path",
        full_width=True,
    )
    source_db_path = mo.ui.text(
        value="/Users/ncheruku/Documents/Work/sample_data/data/lancedb/shared_source/era5_sample_images",
        label="Source DB path (project folder)",
        full_width=True,
    )
    mo.vstack([embedding_db_path, source_db_path])
    return embedding_db_path, source_db_path


@app.cell
def _(embedding_db_path):
    from pathlib import Path

    _p = Path(embedding_db_path.value)
    experiments = []
    if _p.exists() and _p.is_dir():
        for _d in sorted(_p.iterdir()):
            if _d.is_dir() and _d.name.endswith("_config.lance"):
                experiments.append(_d.name[: -len("_config.lance")])
    return (experiments,)


@app.cell
def _(experiments, mo):
    experiment_selector = mo.ui.dropdown(
        options=experiments,
        value=experiments[0] if experiments else None,
        label="Select Experiment",
    )
    experiment_selector
    return (experiment_selector,)


@app.cell
def _(alt):
    _chart = (
        alt.Chart([]) # <-- replace with data
        .mark_bar()
        .transform_aggregate(count="count()", groupby=["date"])
        .transform_window(
            rank="rank()",
            sort=[
                alt.SortField("count", order="descending"),
                alt.SortField("date", order="ascending"),
            ],
        )
        .transform_filter(alt.datum.rank <= 10)
        .encode(
            y=alt.Y(
                "date:N",
                sort="-x",
                axis=alt.Axis(title=None),
            ),
            x=alt.X("count:Q", title="Number of records"),
            tooltip=[
                alt.Tooltip("date:N"),
                alt.Tooltip("count:Q", format=",.0f", title="Number of records"),
            ],
        )
        .properties(width="container")
        .configure_view(stroke=None)
        .configure_axis(grid=False)
    )
    _chart
    return


@app.cell
def _():
    import altair as alt

    return (alt,)


@app.cell
def _(embedding_db_path, experiment_selector, lancedb, source_db_path):
    # Connect and Open Tables
    _db = lancedb.connect(embedding_db_path.value)
    _s_db = lancedb.connect(source_db_path.value)

    _conf_tbl = _db.open_table(f"{experiment_selector.value}_config")
    _df_conf = _conf_tbl.to_pandas()
    config = dict(zip(_df_conf["key"], _df_conf["value"]))

    _img_name = config.get("tbl_img_emb") or config.get("img_emb_table_current")
    _patch_name = config.get("tbl_patch_emb") or config.get("patch_emb_table_current")
    _src_name = config.get("source", "images")

    img_emb_tbl = _db.open_table(_img_name)
    patch_emb_tbl = _db.open_table(_patch_name)
    src_img_tbl = _s_db.open_table(_src_name)
    return config, img_emb_tbl, patch_emb_tbl, src_img_tbl


@app.cell
def _(img_emb_tbl, mo, patch_emb_tbl):
    import numpy as _np

    def _check_norms(tbl, label, n=50):
        _vecs = tbl.search().limit(n).to_pandas()["embedding"].tolist()
        _norms = [_np.linalg.norm(v) for v in _vecs]
        _mean, _mn, _mx = _np.mean(_norms), _np.min(_norms), _np.max(_norms)
        _normalized = "✅ normalized" if abs(_mean - 1.0) < 0.05 else "⚠️ NOT normalized"
        return f"**{label}**: mean norm = `{_mean:.4f}` (min `{_mn:.4f}`, max `{_mx:.4f}`) — {_normalized}"

    mo.callout(
        mo.md(
            "### Embedding diagnostics\n"
            + _check_norms(img_emb_tbl, "Image embeddings") + "\n\n"
            + _check_norms(patch_emb_tbl, "Patch embeddings")
        ),
        kind="info",
    )
    return


@app.cell
def _(mo):
    n_similar_images = mo.ui.number(start=1, stop=50, step=1, value=10, label="Similar images")
    n_similar_patches = mo.ui.number(start=10, stop=500, step=10, value=100, label="Max patches")
    max_gallery_display = mo.ui.number(start=4, stop=100, step=4, value=24, label="Gallery cap")
    similarity_overlay_toggle = mo.ui.switch(label="Similarity overlay")
    map_theme = mo.ui.switch(label="Dark mode")
    return (
        map_theme,
        max_gallery_display,
        n_similar_images,
        n_similar_patches,
        similarity_overlay_toggle,
    )


@app.cell
def _(src_img_tbl):
    _df = (
        src_img_tbl.search()
        .select(["id", "dt"])
        .to_pandas()
        .sort_values("dt")
        .reset_index(drop=True)
    )
    available_ids   = _df["id"].tolist()
    available_dates = [str(d)[:10] for d in _df["dt"].tolist()]
    return available_dates, available_ids


@app.cell
def _(available_dates, mo):
    _n = len(available_dates)
    date_picker = mo.ui.datetime(
        value=available_dates[0] if _n else None,
        start=available_dates[0] if _n else None,
        stop=available_dates[-1] if _n else None,
    )
    date_navigator = date_picker
    return date_navigator, date_picker


@app.cell
def _(available_dates, available_ids, date_picker):
    _s = str(date_picker.value)[:10] if date_picker.value else ""
    selected_img_id = available_ids[available_dates.index(_s)] if _s in available_dates else (available_ids[0] if available_ids else "")
    return (selected_img_id,)


@app.cell
def _(Image, io):
    # Constants and Utilities
    PATCH_SIZE = 16
    IMG_SIZE = 256

    def fetch_image_by_filename(table, filename):
        row = table.search().where(f"filename = '{filename}'").limit(1).to_pandas().iloc[0]
        return Image.open(io.BytesIO(row['image_blob'])).convert('RGB')

    def patch_box_from_index(idx, img_w=IMG_SIZE, img_h=IMG_SIZE, p_size=PATCH_SIZE):
        grid_w = img_w // p_size
        r, c = idx // grid_w, idx % grid_w
        return (c * p_size, r * p_size, (c + 1) * p_size, (r + 1) * p_size)


    return IMG_SIZE, PATCH_SIZE, patch_box_from_index


@app.cell
def _(Image, io, plt, selected_img_id, src_img_tbl):
    # --- IMAGE DISPLAY ---
    _row = src_img_tbl.search().where(f"id = '{selected_img_id}'").select(["image_blob"]).limit(1).to_pandas().iloc[0]
    _raw_img = Image.open(io.BytesIO(_row["image_blob"])).convert("RGB")
    _w, _h = _raw_img.size
    _scale = 4 / max(_w, _h)
    _fig, _ax = plt.subplots(figsize=(_w * _scale, _h * _scale))
    _ax.imshow(_raw_img)
    _ax.axis("off")
    _fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    _fig
    return


@app.cell
def _(mo, src_img_tbl):
    # --- METADATA FILTER ---
    _blob_cols = {"image_blob", "thumb_blob"}
    _meta_cols = [f.name for f in src_img_tbl.schema if f.name not in _blob_cols]
    _meta_df = src_img_tbl.search().select(_meta_cols).to_pandas()
    metadata_filter = mo.ui.dataframe(_meta_df)
    return (metadata_filter,)


@app.cell
def _(IMG_SIZE, PATCH_SIZE, mo):
    # --- PATCH INDEX INPUT ---
    _grid = IMG_SIZE // PATCH_SIZE
    target_patch_idx = mo.ui.number(
        start=0,
        stop=_grid * _grid - 1,
        step=1,
        value=0,
        label="Patch index",
    )
    target_patch_idx
    return (target_patch_idx,)


@app.cell
def _(
    get_selected_patch,
    get_spatial_filter,
    img_emb_tbl,
    metadata_filter,
    n_similar_images,
    n_similar_patches,
    patch_emb_tbl,
    selected_img_id,
    target_patch_idx,
):
    # --- HIERARCHICAL SEARCH ---
    # Map click sets get_selected_patch(); slider is the fallback
    _patch_idx = get_selected_patch() if get_selected_patch() is not None else target_patch_idx.value

    _img_q = img_emb_tbl.search().where(f"image_id = '{selected_img_id}'").select(["embedding"]).limit(1).to_pandas().iloc[0]

    _allowed_ids = metadata_filter.value["id"].tolist()
    _id_clause = ", ".join(f"'{i}'" for i in _allowed_ids)

    _sim_ims = (
        img_emb_tbl.search(_img_q["embedding"], vector_column_name="embedding")
        .metric("cosine")
        .where(f"image_id IN ({_id_clause})")
        .select(["image_id"])
        .limit(n_similar_images.value)
        .to_pandas()
    )
    _filter = ", ".join([f"'{_i}'" for _i in _sim_ims["image_id"].tolist()])

    _p_q = (
        patch_emb_tbl.search()
        .where(f"image_id = '{selected_img_id}' AND patch_index = {_patch_idx}")
        .select(["embedding"])
        .limit(1)
        .to_pandas()
        .iloc[0]
    )

    _spatial = get_spatial_filter()
    _patch_clause = (
        f" AND patch_index IN ({', '.join(str(i) for i in _spatial)})"
        if _spatial else ""
    )

    top_df = (
        patch_emb_tbl.search(_p_q["embedding"], vector_column_name="embedding")
        .where(f"image_id IN ({_filter}){_patch_clause}")
        .metric("cosine")
        .refine_factor(10)
        .select(["image_id", "patch_index"])
        .limit(n_similar_patches.value)
        .to_pandas()
    )
    return (top_df,)


@app.cell
def _(mo, top_df):
    mo.vstack([
        mo.md("**`top_df` debug** — raw columns and first 5 rows:"),
        mo.md(f"`columns`: `{list(top_df.columns)}`"),
        mo.md(f"`_distance` range: min=`{top_df['_distance'].min():.4f}` max=`{top_df['_distance'].max():.4f}` mean=`{top_df['_distance'].mean():.4f}`"),
        mo.as_html(top_df.head()),
    ])
    return


@app.cell
def _(
    get_selected_patch,
    mo,
    patch_emb_tbl,
    selected_img_id,
    target_patch_idx,
    top_df,
):
    import numpy as _np

    # --- Query patch vector ---
    _patch_idx = get_selected_patch() if get_selected_patch() is not None else target_patch_idx.value
    _q_vec = _np.array(
        patch_emb_tbl.search().where(f"image_id = '{selected_img_id}' AND patch_index = {_patch_idx}")
        .select(["embedding"]).limit(1).to_pandas().iloc[0]["embedding"],
        dtype=_np.float32,
    )

    # --- Check first 5 result patch vectors and compare ---
    _rows = []
    for _, _r in top_df.head(5).iterrows():
        _r_vec = _np.array(
            patch_emb_tbl.search().where(f"image_id = '{_r['image_id']}' AND patch_index = {int(_r['patch_index'])}")
            .select(["embedding"]).limit(1).to_pandas().iloc[0]["embedding"],
            dtype=_np.float32,
        )
        _dot   = float(_np.dot(_q_vec, _r_vec))
        _true_cosine = 1.0 - _dot          # valid since both are unit vectors
        _lancedb_d   = float(_r["_distance"])
        _rows.append({
            "patch_index": int(_r["patch_index"]),
            "lancedb _distance": round(_lancedb_d, 4),
            "true cosine dist": round(_true_cosine, 4),
            "d / 64 (num_sub_vectors)": round(_lancedb_d / 64, 4),
            "d / 128 (num_partitions)": round(_lancedb_d / 128, 4),
            "dot product": round(_dot, 4),
        })

    import pandas as _pd
    mo.vstack([
        mo.md("### Distance hypothesis check"),
        mo.md("Comparing LanceDB `_distance` against manually computed cosine distance for the top-5 patch results."),
        mo.as_html(_pd.DataFrame(_rows)),
    ])
    return


@app.cell
def _(
    IMG_SIZE,
    available_ids,
    get_spatial_filter,
    io,
    lat_max,
    lat_min,
    lon_max,
    lon_min,
    max_gallery_display,
    metadata_filter,
    mo,
    n_side,
    patch_box_from_index,
    similarity_overlay_toggle,
    src_img_tbl,
    top_df,
):
    from PIL import Image as _Image, ImageDraw as _ImageDraw

    # --- RESULTS VISUALIZATION ---
    _MAX_DISPLAY = int(max_gallery_display.value)
    _groups = top_df.groupby('image_id').agg({
        'patch_index': list,
        '_distance': 'min'
    }).sort_values('_distance').head(_MAX_DISPLAY)

    _spatial_extent = {"lat_min": lat_min, "lat_max": lat_max, "lon_min": lon_min, "lon_max": lon_max}
    _thumb_w, _thumb_h = compute_thumb_dimensions(_spatial_extent, base_size=320)

    # Pre-build {(image_id, patch_index): distance} for overlay mode
    _patch_dists = {
        (row["image_id"], int(row["patch_index"])): row["_distance"]
        for _, row in top_df.iterrows()
    }

    _thumbs = []
    _date_map = {}
    for _id, _data in _groups.iterrows():
        _r = src_img_tbl.search().where(f"id = '{_id}'").select(['image_blob', 'dt']).limit(1).to_pandas().iloc[0]
        _date_map[_id] = _r['dt']

        if similarity_overlay_toggle.value:
            # Overlay mode: matched patches bright, rest faded
            _matched_dists = {
                int(p): _patch_dists[(_id, int(p))]
                for p in _data['patch_index']
                if (_id, int(p)) in _patch_dists
            }
            _blob = apply_similarity_overlay(_r['image_blob'], _matched_dists, n_side)
        else:
            # Red box mode
            _im = _Image.open(io.BytesIO(_r['image_blob'])).convert('RGB')
            _tw, _th = _im.size
            _sx, _sy = _tw / IMG_SIZE, _th / IMG_SIZE
            _draw = _ImageDraw.Draw(_im)
            for _p in map(int, _data['patch_index']):
                _bx = patch_box_from_index(_p)
                _draw.rectangle(
                    (int(_bx[0]*_sx), int(_bx[1]*_sy), int(_bx[2]*_sx), int(_bx[3]*_sy)),
                    outline=(255, 80, 0), width=2
                )
            _buf = io.BytesIO()
            _im.save(_buf, format="JPEG", quality=85)
            _blob = _buf.getvalue()

        _thumbs.append((f"{str(_r['dt'])[:10]}  ·  d={_data['_distance']:.2f}", _blob, _r['dt']))

    _n_patches = len(top_df)
    _n_images  = top_df['image_id'].nunique()
    _n_shown   = len(_groups)

    _cap = f" (capped at {_MAX_DISPLAY})" if _n_images > _MAX_DISPLAY else ""

    _sf = get_spatial_filter() or []
    _filter_parts = []
    if _sf:
        _filter_parts.append(f"spatial filter active ({len(_sf)} region{'s' if len(_sf) != 1 else ''})")
    _meta_count = len(metadata_filter.value) if metadata_filter.value is not None else len(available_ids)
    if _meta_count < len(available_ids):
        _filter_parts.append(f"data filter: {_meta_count} of {len(available_ids)} rows")
    _filter_note = "  ·  " + ",  ".join(_filter_parts) if _filter_parts else ""

    _status = mo.md(
        f"**{_n_patches} patches** across **{_n_images} images** — showing **{_n_shown}**{_cap}{_filter_note}"
    )

    _, _gallery_html = render_thumbnail_gallery(
        _thumbs, _n_shown, _MAX_DISPLAY, theme="dark",
        thumb_w=_thumb_w, thumb_h=_thumb_h,
    )

    # --- Data tab: one row per image with lists of matched patches + distances ---
    import pandas as _pd
    _df_merged = (
        top_df.groupby("image_id")
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
    mo.vstack([
        mo.hstack([similarity_overlay_toggle], justify="end"),
        mo.ui.tabs({"Visuals": _visual_tab, "Data": _data_tab}),
    ])
    return


@app.function
def get_spatial_extent(src_img_tbl, config):
    """Read lat/lon bounds and patch grid size from table schema metadata + config."""
    import json, math
    lat_min, lat_max = 0.0, 10.0
    lon_min, lon_max = 0.0, 10.0
    n_side = 16
    try:
        raw_meta = src_img_tbl.schema.metadata or {}
        ds_info = json.loads(raw_meta.get(b"dataset_info", "{}")) if raw_meta else {}
        ext = ds_info.get("spatial_extent", {})
        if ext:
            lat_min = float(ext["lat_min"])
            lat_max = float(ext["lat_max"])
            lon_min = float(ext["lon_min"])
            lon_max = float(ext["lon_max"])
        npt = config.get("num_patch_tokens")
        if npt:
            n_side = int(math.sqrt(int(npt)))
    except Exception:
        pass
    return lat_min, lat_max, lon_min, lon_max, n_side


@app.function
def build_coastline_traces(lat_min, lat_max, lon_min, lon_max, n_side):
    """Return a list of go.Scatter coastline traces clipped to the given extent.

    Handles 0–360 lon convention by normalising to −180/+180 for Cartopy
    intersection checks, then shifting coords back for plotting.
    Per-geometry fault tolerance: one bad geometry never aborts the rest.
    """
    import numpy as np
    import plotly.graph_objects as go

    lat_step = (lat_max - lat_min) / n_side
    lon_step = (lon_max - lon_min) / n_side
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
def make_patch_heatmap(lat_min, lat_max, lon_min, lon_max, n_side):
    """Invisible N×N heatmap whose z values are flat patch indices (click target)."""
    import numpy as np
    import plotly.graph_objects as go

    lat_step = (lat_max - lat_min) / n_side
    lon_step = (lon_max - lon_min) / n_side
    z = np.arange(n_side * n_side).reshape(n_side, n_side)
    hm_x = [lon_min + (c + 0.5) * lon_step for c in range(n_side)]
    # y descending so z[0] (northernmost row) maps to the top of the plot
    hm_y = [lat_max - (r + 0.5) * lat_step for r in range(n_side)]
    return go.Heatmap(
        z=z, x=hm_x, y=hm_y,
        opacity=0.01,        # >0 so browser click events still fire
        showscale=False,
        colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
        hovertemplate="Patch %{z}  (%{y:.2f}°, %{x:.2f}°)<extra></extra>",
    )


@app.function
def make_selection_shape(patch_idx, lat_min, lat_max, lon_min, lon_max, n_side):
    """Return a red rectangle shape dict for the selected patch, or None."""
    if patch_idx is None:
        return None
    lat_step = (lat_max - lat_min) / n_side
    lon_step = (lon_max - lon_min) / n_side
    p_row = patch_idx // n_side
    p_col = patch_idx % n_side
    x0 = lon_min + p_col * lon_step
    y1 = lat_max - p_row * lat_step       # north edge
    y0 = y1 - lat_step                    # south edge
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
    coast_traces, heatmap_trace, selection_shape, theme="light",
):
    """Assemble the three-layer geo patch figure from pre-built components."""
    import plotly.graph_objects as go

    _is_dark = (theme == "dark")
    _bg   = "#1a1a1a" if _is_dark else "white"
    _text = "#e0e0e0" if _is_dark else "#222222"

    H, W = img_arr.shape[:2]
    fig = go.Figure()

    # Layer 1: satellite image anchored in lon/lat space (dy<0 → rows go southward)
    fig.add_trace(go.Image(
        z=img_arr,
        x0=lon_min, dx=(lon_max - lon_min) / W,
        y0=lat_max, dy=-(lat_max - lat_min) / H,
        hoverinfo="skip",
    ))

    # Layer 2: coastlines (pre-computed, only rebuilds when extent changes)
    for trace in coast_traces:
        fig.add_trace(trace)

    # Layer 3: invisible heatmap click target (pre-computed)
    fig.add_trace(heatmap_trace)

    shapes = [selection_shape] if selection_shape is not None else []
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
        margin=dict(l=65, r=10, t=40, b=60),
        plot_bgcolor=_bg,
        paper_bgcolor=_bg,
    )
    return fig


@app.function
def compute_thumb_dimensions(spatial_extent, base_size=192):
    """Compute (width_px, height_px) preserving geographic aspect ratio with cosine correction."""
    import math
    lat_range = abs(spatial_extent["lat_max"] - spatial_extent["lat_min"])
    lon_range = abs(spatial_extent["lon_max"] - spatial_extent["lon_min"])
    mean_lat = (spatial_extent["lat_min"] + spatial_extent["lat_max"]) / 2
    effective_lon = lon_range * math.cos(math.radians(mean_lat))
    if lat_range == 0 or effective_lon == 0:
        return base_size, base_size
    aspect = effective_lon / lat_range
    if aspect >= 1:
        return base_size, round(base_size / aspect)
    else:
        return round(base_size * aspect), base_size


@app.function
def apply_similarity_overlay(image_blob, matched_patch_distances, n_side, alpha_min=0.08, bg_color=(0, 0, 0)):
    """Fade non-matched patches toward bg_color; matched patches stay opaque.

    matched_patch_distances: {patch_index: cosine_distance} — lower = more similar.
    Matched patches graded [0.5, 1.0] by relative distance; non-matched get alpha_min.
    """
    import io
    import numpy as np
    from PIL import Image

    img = Image.open(io.BytesIO(image_blob)).convert("RGBA")
    iw, ih = img.size

    alpha_grid = np.full((n_side, n_side), alpha_min, dtype=np.float32)
    if matched_patch_distances:
        dists = np.array(list(matched_patch_distances.values()), dtype=np.float32)
        d_min, d_max = dists.min(), dists.max()
        for pidx, dist in matched_patch_distances.items():
            row, col = int(pidx) // n_side, int(pidx) % n_side
            norm = (dist - d_min) / (d_max - d_min + 1e-8)
            alpha_grid[row, col] = 1.0 - norm * 0.5   # grades to [0.5, 1.0]

    ph, pw = ih // n_side, iw // n_side
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
def render_thumbnail_gallery(thumbs, n_filtered, max_display, theme="light",
                             thumb_w=192, thumb_h=192):
    """Build HTML for a theme-aware thumbnail gallery with datetime labels."""
    import base64
    border = "#cccccc" if theme == "light" else "#444444"
    text   = "#222222" if theme == "light" else "#e0e0e0"
    bg     = "#ffffff" if theme == "light" else "rgba(30,30,30,0.85)"

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
            f'{fname}</div></div>'
        )

    count_msg = f"Showing {len(thumbs)} of {n_filtered} results"
    if n_filtered > max_display:
        count_msg += f" (capped at {max_display})"

    gallery_html = (
        f'<div style="display:flex;flex-wrap:wrap;gap:4px;align-content:flex-start;'
        f'height:780px;overflow-y:auto;background:{bg};'
        f'border-radius:8px;padding:8px;border:1px solid {border}">'
        + "".join(imgs)
        + "</div>"
    )
    return count_msg, gallery_html


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
def build_spatial_filter_shapes(selected_indices, lat_min, lat_max, lon_min, lon_max, n_side):
    """Return a list of Plotly rect shapes highlighting each selected patch."""
    if not selected_indices:
        return []
    lat_step = (lat_max - lat_min) / n_side
    lon_step = (lon_max - lon_min) / n_side
    shapes = []
    for idx in selected_indices:
        row, col = idx // n_side, idx % n_side
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


@app.function
def build_patch_grid_lines(lat_min, lat_max, lon_min, lon_max, n_side):
    """Dotted grid lines at patch boundaries."""
    import plotly.graph_objects as go

    lat_step = (lat_max - lat_min) / n_side
    lon_step = (lon_max - lon_min) / n_side

    xs, ys = [], []
    for i in range(n_side + 1):
        lat = lat_min + i * lat_step
        xs.extend([lon_min, lon_max, None])
        ys.extend([lat, lat, None])
    for j in range(n_side + 1):
        lon = lon_min + j * lon_step
        xs.extend([lon, lon, None])
        ys.extend([lat_min, lat_max, None])

    return go.Scatter(
        x=xs, y=ys,
        mode="lines",
        line=dict(color="rgba(100,100,100,0.3)", width=0.5, dash="dot"),
        showlegend=False,
        hoverinfo="skip",
        name="patch_grid",
    )


@app.cell
def _(config, src_img_tbl):
    lat_min, lat_max, lon_min, lon_max, n_side = get_spatial_extent(src_img_tbl, config)
    return lat_max, lat_min, lon_max, lon_min, n_side


@app.cell
def _(lat_max, lat_min, lon_max, lon_min, n_side):
    # Expensive layers — recomputed only when the geographic extent changes,
    # not on every image swap or patch click.
    coast_traces = build_coastline_traces(lat_min, lat_max, lon_min, lon_max, n_side)
    patch_heatmap_trace = make_patch_heatmap(lat_min, lat_max, lon_min, lon_max, n_side)
    return coast_traces, patch_heatmap_trace


@app.cell
def _(mo):
    get_spatial_filter, set_spatial_filter = mo.state(None)
    return get_spatial_filter, set_spatial_filter


@app.cell
def _(
    get_spatial_filter,
    lat_max,
    lat_min,
    lon_max,
    lon_min,
    map_theme,
    mo,
    n_side,
    patch_heatmap_trace,
):
    _theme = "dark" if map_theme.value else "light"
    _basemap_arr = render_basemap(lat_min, lat_max, lon_min, lon_max, theme=_theme)
    _active = get_spatial_filter() or []
    _fig_sf = build_geo_patch_figure(
        _basemap_arr, lon_min, lon_max, lat_min, lat_max,
        [], patch_heatmap_trace, None, theme=_theme,  # coastlines already in basemap image
    )
    _fig_sf.update_layout(
        shapes=build_spatial_filter_shapes(_active, lat_min, lat_max, lon_min, lon_max, n_side),
        uirevision="spatial_filter_map",
    )
    spatial_filter_map = mo.ui.plotly(_fig_sf)
    return (spatial_filter_map,)


@app.cell
def _(get_spatial_filter, set_spatial_filter, spatial_filter_map):
    _val = spatial_filter_map.value
    if isinstance(_val, list) and _val and "z" in _val[0]:
        _idx = int(_val[0]["z"])
        _cur = list(get_spatial_filter() or [])
        if _idx in _cur:
            _cur.remove(_idx)
        else:
            _cur.append(_idx)
        set_spatial_filter(_cur if _cur else None)
    return


@app.cell
def _(mo):
    get_selected_patch, set_selected_patch = mo.state(None)
    return get_selected_patch, set_selected_patch


@app.cell
def _(set_selected_patch, target_patch_idx):
    # Keep existing patch-index slider in sync with shared state
    set_selected_patch(target_patch_idx.value)
    return


@app.cell
def _(
    Image,
    coast_traces,
    get_selected_patch,
    io,
    lat_max,
    lat_min,
    lon_max,
    lon_min,
    map_theme,
    mo,
    n_side,
    patch_heatmap_trace,
    selected_img_id,
    src_img_tbl,
):
    import numpy as _np

    _theme = "dark" if map_theme.value else "light"
    _sel = get_selected_patch()
    _row = src_img_tbl.search().where(f"id = '{selected_img_id}'").select(["image_blob"]).limit(1).to_pandas().iloc[0]
    _img_arr = _np.array(Image.open(io.BytesIO(_row["image_blob"])).convert("RGB"))
    _shape = make_selection_shape(_sel, lat_min, lat_max, lon_min, lon_max, n_side)
    _fig = build_geo_patch_figure(
        _img_arr, lon_min, lon_max, lat_min, lat_max,
        coast_traces, patch_heatmap_trace, _shape, theme=_theme,
    )
    geo_patch_map = mo.ui.plotly(_fig)
    return (geo_patch_map,)


@app.cell
def _(geo_patch_map, set_selected_patch):
    _click = geo_patch_map.value
    if isinstance(_click, list) and _click and "z" in _click[0]:
        set_selected_patch(int(_click[0]["z"]))
    return


@app.cell
def _(
    date_navigator,
    geo_patch_map,
    get_selected_patch,
    get_spatial_filter,
    map_theme,
    max_gallery_display,
    metadata_filter,
    mo,
    n_similar_images,
    n_similar_patches,
    set_spatial_filter,
    spatial_filter_map,
):
    _sel = get_selected_patch()
    _label_q = f"**Selected patch:** `{_sel}`" if _sel is not None else "*Click a patch to select it*"

    _active_sf = get_spatial_filter() or []
    _n_sf = len(_active_sf)
    _label_s = (
        f"**Search region:** {_n_sf} patches selected"
        if _n_sf else "*Click patches to restrict the search region*"
    )
    _clear_btn = mo.ui.button(label="✕ Clear", on_click=lambda _: set_spatial_filter(None))

    mo.ui.tabs({
        "Patch Query": mo.vstack([date_navigator, mo.md(_label_q), geo_patch_map]),
        "Search Region": mo.vstack([
            mo.hstack([mo.md(_label_s), _clear_btn], align="center"),
            spatial_filter_map,
        ]),
        "Data Filter": metadata_filter,
        "Settings": mo.vstack([n_similar_images, n_similar_patches, max_gallery_display, map_theme]),
    })
    return


@app.cell
def _(patch_emb_tbl):
    patch_emb_tbl.list_indices()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
