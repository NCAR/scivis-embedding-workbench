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
    import matplotlib.ticker as mticker
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
    colors = get_theme_colors(theme)
    # A lon span of exactly 360 (e.g. 0-360 datasets) makes cartopy normalize
    # both endpoints to the same longitude, collapsing set_extent to a point.
    lon_max = min(lon_max, lon_min + 359.9)
    # data_crs is absolute lon/lat (what lon_min/lon_max/grid coords are expressed
    # in); the axes projection is centered on the extent's midpoint so lon_min
    # lands on the left edge and lon_max on the right, matching the source
    # image's own pixel convention (e.g. a 0-360 dataset has column 0 = lon 0,
    # not the Atlantic-centered -180..180 that a default-centered projection
    # would show).
    data_crs = ccrs.PlateCarree()
    proj = ccrs.PlateCarree(central_longitude=(lon_min + lon_max) / 2.0)
    fig, ax = plt.subplots(figsize=(8, 5), subplot_kw={"projection": proj})
    fig.patch.set_facecolor(colors["bg"])
    ax.set_facecolor(colors["bg"])
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=data_crs)
    ax.add_feature(cfeature.OCEAN.with_scale("110m"), facecolor=colors["ocean"], zorder=0)
    ax.add_feature(cfeature.LAND.with_scale("110m"),  facecolor=colors["land"],  zorder=1)
    ax.add_feature(cfeature.COASTLINE.with_scale("110m"), edgecolor=colors["coast"], linewidth=0.8, zorder=2)
    # NOTE: gridlines(draw_labels=True) triggers a cartopy 0.25 / matplotlib 3.11 bug
    # (shapely LinearRing exception in _draw_gridliner). Use axis ticks instead.
    ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax.yaxis.set_major_locator(mticker.MaxNLocator(nbins=5))
    ax.xaxis.set_major_formatter(LongitudeFormatter())
    ax.yaxis.set_major_formatter(LatitudeFormatter())
    ax.tick_params(colors=colors["text"], labelsize=8)
    n_rows, n_cols = spatial_h, spatial_w
    img_h = n_rows * patch_size
    img_w = n_cols * patch_size
    lat_step = (lat_max - lat_min) / n_rows
    lon_step = (lon_max - lon_min) / n_cols
    for i in range(1, n_rows):
        ax.plot([lon_min, lon_max], [lat_min + i * lat_step] * 2,
                transform=data_crs, color=colors["grid"], linewidth=0.4, zorder=3)
    for j in range(1, n_cols):
        ax.plot([lon_min + j * lon_step] * 2, [lat_min, lat_max],
                transform=data_crs, color=colors["grid"], linewidth=0.4, zorder=3)
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

    try:
        import cartopy.feature as cfeature
    except ImportError as e:
        raise ImportError(
            "cartopy is required for coastline rendering. "
            "Install with: pip install cartopy"
        ) from e

    try:
        coast_geoms = list(cfeature.COASTLINE.with_scale("110m").geometries())
    except Exception:
        coast_geoms = []

    # Coastline data is natively in [-180, 180]. Try each wraparound shift so
    # extents using a [0, 360) convention still pick up geometry that would
    # otherwise fall outside [lon_min, lon_max] (e.g. the Americas at native
    # negative longitude, which belong near the 180-360 side of the extent).
    traces = []
    for geom in coast_geoms:
        try:
            lines = list(geom.geoms) if hasattr(geom, "geoms") else [geom]
        except Exception:
            continue
        for line in lines:
            try:
                xy = np.array(line.coords)
                if xy.ndim != 2 or xy.shape[0] < 2:
                    continue
            except Exception:
                continue
            for shift in (0, 360, -360):
                x_shifted = xy[:, 0] + shift
                if x_shifted.max() < lon_min - buffer or x_shifted.min() > lon_max + buffer:
                    continue
                traces.append(go.Scatter(
                    x=x_shifted.tolist(),
                    y=xy[:, 1].tolist(),
                    mode="lines",
                    line=dict(color="white", width=1.5),
                    opacity=0.8,
                    showlegend=False,
                    hoverinfo="skip",
                    name="coastline",
                ))
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
    coast_traces, heatmap_trace, selection_shape, theme="light", target_w=1800,
):
    """Assemble the three-layer geo patch figure from pre-built components.

    target_w is the figure's actual pixel width; the height follows from the
    extent's aspect ratio. Both are set explicitly, because setting only the
    height (as this did originally) lets plotly shrink the width to the panel
    and `scaleanchor` then letterboxes the map back down to a ~166px strip.
    Raising target_w makes the map bigger; the panel it sits in scrolls
    horizontally when it no longer fits.
    """
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
        autosize=False,
        width=target_w,
        height=_fig_h,
        margin=dict(l=_l, r=_r, t=_t, b=_b),
        plot_bgcolor=_bg,
        paper_bgcolor=_bg,
    )
    return fig


def _open_scaled(image_blob, target_size=None, mode="RGB"):
    """Open a JPEG blob, optionally downscaled to target_size.

    Returns (image, scale) where scale is target_w / source_w (1.0 when no
    resize happened) so callers can scale line widths to match.

    Gallery thumbnails used to be annotated at full resolution, re-encoded to
    JPEG, decoded again and only then resized. Downscaling up front avoids that
    round-trip entirely, and `draft()` lets libjpeg do most of the reduction
    during DCT decode rather than decoding every full-resolution pixel first.
    """
    import io
    from PIL import Image

    img = Image.open(io.BytesIO(image_blob))
    if target_size is None:
        return img.convert(mode), 1.0

    tw, th = int(target_size[0]), int(target_size[1])
    src_w = img.size[0]
    # draft() only honours power-of-two reductions and only for JPEG; asking
    # for 2x the target keeps enough detail for a good LANCZOS finish.
    img.draft(mode if mode == "RGB" else "RGB", (tw * 2, th * 2))
    img = img.convert(mode).resize((tw, th), Image.LANCZOS)
    return img, (tw / src_w if src_w else 1.0)


def apply_similarity_overlay(image_blob, matched_patch_distances, n_rows, n_cols, alpha_min=0.08, bg_color=(0, 0, 0), target_size=None):
    """Fade non-matched patches toward bg_color; matched patches stay opaque."""
    import io
    import numpy as np
    from PIL import Image

    img, _ = _open_scaled(image_blob, target_size, mode="RGBA")
    iw, ih = img.size

    alpha_grid = np.full((n_rows, n_cols), alpha_min, dtype=np.float32)
    if matched_patch_distances:
        dists = np.array(list(matched_patch_distances.values()), dtype=np.float32)
        d_min, d_max = dists.min(), dists.max()
        for pidx, dist in matched_patch_distances.items():
            row, col = int(pidx) // n_cols, int(pidx) % n_cols
            norm = (dist - d_min) / (d_max - d_min + 1e-8)
            alpha_grid[row, col] = 1.0 - norm * 0.5

    # Map each pixel to its patch directly, rather than repeat-then-pad with an
    # integer patch size. Repeating by ih//n_rows / iw//n_cols truncates and the
    # edge padding smears the shortfall onto the last row/column, drifting by
    # nearly a whole patch at thumbnail sizes (same defect as the box grid in
    # annotate_patch_image). This is exact at any size.
    row_of = (np.arange(ih) * n_rows // ih).clip(0, n_rows - 1)
    col_of = (np.arange(iw) * n_cols // iw).clip(0, n_cols - 1)
    alpha_up = alpha_grid[row_of[:, None], col_of[None, :]]

    img_arr = np.array(img)
    img_arr[..., 3] = (alpha_up * 255).astype(np.uint8)
    masked = Image.fromarray(img_arr, "RGBA")
    bg = Image.new("RGBA", (iw, ih), bg_color + (255,))
    composite = Image.alpha_composite(bg, masked).convert("RGB")

    buf = io.BytesIO()
    composite.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def annotate_patch_image(image_blob, patch_indices, matched_distances, n_rows, n_cols, use_similarity, box_color="black", box_width=2, target_size=None):
    """Annotate an image with matched-patch highlighting: similarity fade or grid boxes.

    Shared by the Spatial Search gallery's small preview and its on-demand
    full-resolution view, so both render identical annotations. box_color
    (any PIL-recognized color string, e.g. a "#RRGGBB" hex) and box_width
    only apply to the grid-box mode — the similarity overlay has no
    outline to style.

    Pass target_size=(w, h) to annotate at thumbnail resolution; box_width is
    scaled by the same factor so a thumbnail looks like a shrunk copy of the
    full-resolution view rather than a version with proportionally fatter
    outlines. The full-resolution view omits target_size.
    """
    import io
    from PIL import Image, ImageDraw

    if use_similarity:
        return apply_similarity_overlay(
            image_blob, matched_distances, n_rows, n_cols, target_size=target_size
        )

    im, scale = _open_scaled(image_blob, target_size)
    iw, ih = im.size
    # Fractional patch size, rounded per edge. Integer division here truncates
    # (a 320px-wide thumbnail over 63 columns gives 5 instead of 5.079) and the
    # error accumulates across the grid — nearly a full patch of drift by the
    # right-hand edge, so boxes landed on the wrong patch. It happens to be
    # exact at full resolution, where the image is a whole number of patches
    # per side, which is why only the thumbnails were visibly wrong.
    patch_w = iw / n_cols
    patch_h = ih / n_rows
    box_width = max(1, round(box_width * scale))
    draw = ImageDraw.Draw(im)
    for p in map(int, patch_indices):
        pr, pc = p // n_cols, p % n_cols
        box = (
            round(pc * patch_w), round(pr * patch_h),
            round((pc + 1) * patch_w), round((pr + 1) * patch_h),
        )
        draw.rectangle(box, outline=box_color, width=box_width)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=85)
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

    # A lon span of exactly 360 (e.g. 0-360 datasets) makes cartopy normalize
    # both endpoints to the same longitude, collapsing set_extent to a point.
    lon_max_c = min(lon_max, lon_min + 359.9)
    # data_crs is absolute lon/lat; the axes projection is centered on the
    # extent's midpoint so lon_min lands on the left edge and lon_max on the
    # right, matching the source image's own pixel convention (e.g. a 0-360
    # dataset has column 0 = lon 0, not an Atlantic-centered -180..180 view).
    data_crs = ccrs.PlateCarree()
    proj = ccrs.PlateCarree(central_longitude=(lon_min + lon_max) / 2.0)
    aspect = (lon_max - lon_min) / max(lat_max - lat_min, 1e-6)
    target_h = max(1, int(target_w / aspect))

    fig, ax = plt.subplots(
        figsize=(target_w / 100, target_h / 100), dpi=100,
        subplot_kw={"projection": proj},
    )
    fig.patch.set_facecolor(_bg)
    ax.set_facecolor(_bg)
    ax.set_extent([lon_min, lon_max_c, lat_min, lat_max], crs=data_crs)
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
