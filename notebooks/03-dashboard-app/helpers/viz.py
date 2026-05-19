"""Viz-side helpers for the dashboard: matplotlib/cartopy figures, plotly traces,
PIL image compositing, themed HTML rendering.

Extracted from app.py to keep the marimo notebook focused on reactive UI cells.
No marimo dependency. Owns the shared `get_theme_colors` palette used by every
plotting/HTML helper in the dashboard.
"""


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


def render_thumbnail_gallery(thumbs, n_filtered, max_display, theme="light",
                             thumb_w=192, thumb_h=192, full_blobs=None):
    """Build HTML for a theme-aware thumbnail gallery with datetime labels.

    If `full_blobs` is a list aligned with `thumbs`, each thumbnail becomes
    clickable: clicking opens the corresponding full-resolution image in a
    pure-CSS lightbox overlay (hidden-checkbox sibling-selector technique —
    no JavaScript, so it survives marimo's HTML sanitizer). Click anywhere
    on the overlay to close.
    """
    import base64
    import uuid

    _c = get_theme_colors(theme)
    bg, text, border = _c["gallery_bg"], _c["text"], _c["border"]

    # Per-render id prefix keeps checkbox ids unique across re-renders / cells
    _render_id = uuid.uuid4().hex[:8]
    _cls = f"lbx-{_render_id}"   # scoped class to avoid global CSS collisions

    _has_any_full = (
        full_blobs is not None
        and any(fb is not None for fb in full_blobs[: len(thumbs)])
    )

    imgs = []
    for _i, (fname, blob, dt) in enumerate(thumbs):
        b64 = base64.b64encode(blob).decode()

        def _fmt_dt(_d):
            if _d is None:
                return "—"
            if hasattr(_d, "strftime"):
                try:
                    _s = _d.strftime("%Y-%m-%d %H:%M")
                except (ValueError, AttributeError):
                    _s = str(_d)
            else:
                _s = str(_d)
            return "—" if _s in ("", "NaT", "NaTType", "None") else _s

        dt_str = _fmt_dt(dt)

        _has_full = full_blobs is not None and _i < len(full_blobs) and full_blobs[_i] is not None
        if _has_full:
            _slot = f"{_render_id}-{_i}"
            _full_b64 = base64.b64encode(full_blobs[_i]).decode()
            # Order matters: <input> must come before label + overlay so the
            # `.lbx-cb:checked ~ .lbx-overlay` sibling selector can match.
            imgs.append(
                f'<span class="{_cls}-slot" style="display:inline-block;margin:3px;text-align:center;position:relative">'
                f'<input type="checkbox" class="{_cls}-cb" id="lb-{_slot}">'
                f'<label for="lb-{_slot}" class="{_cls}-thumb" title="{fname} — click to zoom">'
                f'<img src="data:image/jpeg;base64,{b64}" '
                f'style="width:{thumb_w}px;height:{thumb_h}px;object-fit:fill;'
                f'border:1px solid {border};border-radius:4px;display:block"/>'
                f'</label>'
                f'<div style="font-size:11px;color:{text};max-width:{thumb_w}px;'
                f'overflow:hidden;text-overflow:ellipsis;white-space:nowrap">'
                f'{dt_str}</div>'
                f'<label for="lb-{_slot}" class="{_cls}-overlay">'
                f'<img src="data:image/jpeg;base64,{_full_b64}"/>'
                f'</label>'
                f'</span>'
            )
        else:
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

    # Inject CSS only when at least one thumb has a full-res blob
    _style = ""
    if _has_any_full:
        _style = (
            f'<style>'
            f'.{_cls}-cb {{ display: none; }}'
            f'.{_cls}-thumb {{ cursor: zoom-in; display: inline-block; }}'
            f'.{_cls}-overlay {{ '
            f'display: none; position: fixed; inset: 0; '
            f'background: rgba(0,0,0,0.85); z-index: 2147483647; '
            f'align-items: center; justify-content: center; '
            f'cursor: zoom-out; '
            f'}}'
            f'.{_cls}-cb:checked ~ .{_cls}-overlay {{ display: flex; }}'
            f'.{_cls}-overlay img {{ '
            f'max-width: 95vw; max-height: 95vh; '
            f'border-radius: 4px; '
            f'box-shadow: 0 8px 32px rgba(0,0,0,0.5); '
            f'}}'
            f'</style>'
        )

    gallery_html = (
        _style
        + f'<div class="{_cls}" style="display:flex;flex-wrap:wrap;gap:4px;align-content:flex-start;'
        f'height:600px;overflow-y:auto;background:{bg};'
        f'border-radius:8px;padding:8px;border:1px solid {border}">'
        + "".join(imgs)
        + "</div>"
    )
    return count_msg, gallery_html


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
