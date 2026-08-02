"""Datashaded scatter of a 2-D projection.

`viz.py` builds HTML strings and imports no plotting library; this module is its
counterpart for holoviews objects. It returns a Panel pane and never a marimo
element, so it stays usable from a plain script.

The rendering path -- lazy rasterize, equal x/y scales, responsive width -- came
from `datashader_scatter.py`, which exists to prove it at ~1M points. The
comments below record the two failure modes that shaped it; both are silent, and
both cost real time to rediscover.
"""

_ACTIVATED = False

# Last view the user was looking at, per plot key. marimo re-runs the cell that
# builds the plot whenever any of its inputs change -- colour-by, colormap,
# background, width -- and each rebuild is a brand new bokeh figure with default
# ranges, so the zoom is lost. Bokeh range changes do sync back to Python, so
# the ranges of the outgoing figure can be read and handed to its replacement.
# Keyed so switching projection starts fresh rather than restoring a window that
# belonged to different data.
_LAST_VIEW = {}


def remember_view(key, figure):
    """Record a figure's current ranges under `key`, if they are set."""
    if figure is None:
        return
    xr, yr = figure.x_range, figure.y_range
    if None in (xr.start, xr.end, yr.start, yr.end):
        return
    _LAST_VIEW[key] = ((xr.start, xr.end), (yr.start, yr.end))


def forget_view(key=None):
    """Drop a remembered view, so the next build starts at the full extent."""
    if key is None:
        _LAST_VIEW.clear()
        _LIVE_FIGURE.clear()
    else:
        _LAST_VIEW.pop(key, None)
        _LIVE_FIGURE.pop(key, None)


# The figure currently on screen, so the *next* build can read its ranges.
_LIVE_FIGURE = {}


def _capture_hook(key):
    """Hook keeping a reference to the live figure for this key."""

    def hook(plot, element):
        _LIVE_FIGURE[key] = plot.state

    return hook


def activate():
    """Load the bokeh backend once.

    Idempotent because every `scatter_pane` call would otherwise re-fire the
    extension, which is slow and duplicates the JS payload.
    """
    global _ACTIVATED
    if _ACTIVATED:
        return
    import holoviews as hv
    import panel as pn

    hv.extension("bokeh")
    pn.extension()
    _ACTIVATED = True


# Label -> palette key. Labels are what the dropdown shows; keys are resolved to
# an explicit hex list by `resolve_cmap` so the raster and the hover overlay are
# always driven by one representation and cannot drift apart.
CONTINUOUS_CMAPS = {
    "Viridis": "viridis",
    "Inferno": "inferno",
    "Fire": "fire",
    "Blue-Magenta-Yellow": "bmy",
    "Blues": "kbc",
    "Coolwarm (diverging)": "coolwarm",
}

CATEGORICAL_CMAPS = {
    "Glasbey Light": "glasbey_light",
    "Glasbey Dark": "glasbey_dark",
    "Glasbey Category10": "glasbey_category10",
    "Category10": "category10",
}

# HDBSCAN writes -1 for unassigned points and the storm metadata writes -1 for
# "no storm". Either way it means absence, so it gets a neutral grey instead of
# a saturated hue that would make noise look like a real group.
NOISE_VALUE = -1
NOISE_COLOR = "#808080"


def cmap_options(kind: str) -> dict:
    """Dropdown options appropriate to a colour-by column's kind."""
    return dict(CATEGORICAL_CMAPS if kind == "categorical" else CONTINUOUS_CMAPS)


def default_cmap(kind: str, light_bg: bool = False) -> str:
    """A sensible starting palette for this kind of column and canvas."""
    if kind == "categorical":
        return "glasbey_dark" if light_bg else "glasbey_light"
    return "viridis"


def resolve_cmap(key: str, n: int = 256) -> list:
    """Palette key -> list of hex colours.

    Everything comes back in one form, because the raster and the overlay are
    driven by the same list and bokeh's legend rejects RGB tuples. colorcet is
    inconsistent about this -- `glasbey_light` is hex while `glasbey_category10`
    is float RGB triples -- so normalize rather than trusting the source.
    matplotlib colormaps are sampled to `n` steps.
    """
    import colorcet as cc
    from matplotlib.colors import to_hex

    if key == "category10":
        return [to_hex(c) for c in cc.glasbey_category10[:10]]
    palette = getattr(cc, key, None)
    if palette is not None:
        return [c if isinstance(c, str) else to_hex(c) for c in palette]

    from matplotlib import colormaps

    cmap = colormaps[key]
    return [to_hex(cmap(i / (n - 1))) for i in range(n)]


def category_color_key(values, cmap_key: str) -> dict:
    """Explicit {category: colour} for a categorical column.

    Left implicit, datashader colours by category order while holoviews colours
    by the hover sample's own ordering, and the raster and overlay disagree --
    the same points end up different colours in the two layers. One shared
    mapping is the fix.
    """
    palette = resolve_cmap(cmap_key)
    cats = list(values)
    colors = {}
    i = 0
    for c in cats:
        if c == NOISE_VALUE:
            colors[c] = NOISE_COLOR
        else:
            colors[c] = palette[i % len(palette)]
            i += 1
    return colors


def _categories(series):
    """Sorted unique values of a column, as a plain list."""
    return sorted(series.dropna().unique().tolist())


def scatter_pane(
    projection,
    color_by: str = "density",
    cmap: str = None,
    background: str = "#2b2b2b",
    hover_sample: int = 2000,
    seed: int = 0,
    pixel_ratio: float = 2.0,
    max_width: int = 800,
    hover_frame=None,
    tiles=None,
    tile_size: int = 56,
    tile_alpha: float = 0.9,
    hover_size: int = 7,
    hover_alpha: float = 0.75,
    hover_reach: int = 18,
    view_key=None,
    verbose: bool = True,
):
    """Datashaded scatter of `projection`, coloured by one of its columns.

    `color_by` is "density" or any column name; "density" needs nothing but x/y,
    so it is always valid even for a projection that was never clustered.

    `pixel_ratio` multiplies the aggregation grid relative to the frame's CSS
    size without changing how big the plot is on screen. It defaults to 2
    because a CSS pixel is half a physical pixel on a Retina display, so a ratio
    of 1 aggregates at half resolution and every bin is drawn as a 2x2 block.
    Raising it costs aggregation time as the square (2 -> 4x the bins), so 3 or
    4 is worth it only for a still figure.

    `max_width` caps how wide the frame gets on screen. The frame is square and
    sized off its width, so this bounds the height too, and it is a plain pixel
    cap rather than a viewport-relative one -- a vh-based cap collapses to
    nothing whenever the page lays out with an unknown viewport height. Crispness
    is unaffected: shrinking the frame shrinks the aggregation grid with it, so
    the bins-per-screen-pixel ratio stays wherever `pixel_ratio` put it.

    `hover_frame` supplies the overlay rows instead of sampling here. Pass one
    from `PatchExperiment.hover_frame(thumbnails=True)` and its `thumb` column
    becomes an image in the tooltip. Building it outside keeps it in its own
    cell, so changing a colour or a size does not rebuild the crops.

    `tiles` draws a few crops directly on the plot at their own coordinates --
    from `PatchExperiment.tile_frame()`, which picks them for spatial coverage.
    They are `tile_size` screen pixels square at `tile_alpha` opacity, above the
    raster. They are picked once for the full extent -- zooming in does not
    repick them, so a tight zoom may contain few or none.

    `view_key` preserves the zoom across rebuilds. marimo re-runs the cell that
    builds this plot whenever any control changes, and each rebuild is a fresh
    figure at the full extent -- so changing a colormap would throw away the
    region you had zoomed into. Given a key, the outgoing figure's ranges are
    read and applied to its replacement. Use something that identifies the data,
    such as the projection name: a different key starts at the full extent,
    which is what you want when the underlying points have changed.
    """
    import itertools

    import datashader as ds
    import holoviews as hv
    import holoviews.operation.datashader as hd
    import panel as pn

    activate()

    df = projection.df
    kind = projection.kind(color_by)
    light_bg = background.lower() in ("white", "#ffffff", "#fff")
    cmap = cmap or default_cmap(kind, light_bg)

    vdims = [c for c in (color_by,) if c in df.columns]
    color_key = None

    # Each kind needs its own aggregator: counts for density, a per-category
    # count so overlapping groups blend instead of overpainting, and a mean for
    # continuous columns.
    if kind == "density":
        palette = resolve_cmap(cmap)
        # Sequential ramps run dark -> bright, which only reads on a dark
        # canvas. On white, flip it so dense stays dark and sparse fades out.
        shade_kwargs = dict(
            cmap=list(reversed(palette)) if light_bg else palette, cnorm="log"
        )
        aggregator = ds.count()
    elif kind == "categorical":
        # count_cat requires a pandas Categorical; cluster, month and year are
        # plain integer columns, and passing them through raw fails inside
        # datashader rather than at the call site.
        df = df.assign(**{color_by: df[color_by].astype("category")})
        color_key = category_color_key(_categories(df[color_by]), cmap)
        aggregator = ds.count_cat(color_by)
        shade_kwargs = dict(color_key=color_key, cnorm="log")
    else:
        aggregator = ds.mean(color_by)
        shade_kwargs = dict(cmap=resolve_cmap(cmap), cnorm="linear")

    n_aggregations = itertools.count(1)

    def _shade(agg):
        # Logs every re-aggregation to the terminal, so zoom behaviour is
        # visible without guessing from the picture. If the counter climbs while
        # nothing is being touched, something is re-triggering the range stream.
        if verbose:
            x0, x1 = agg.range("x")
            y0, y1 = agg.range("y")
            print(
                f"[datashade] #{next(n_aggregations)} {color_by} window "
                f"x=({x0:.2f}, {x1:.2f}) y=({y0:.2f}, {y1:.2f})",
                flush=True,
            )
        return hd.shade(agg, **shade_kwargs)

    # rasterize stays lazy: it re-runs with the current view limits on every
    # zoom/pan. The grid is pinned rather than left to holoviews' PlotSize
    # stream, which only knows the frame size once the browser has laid the page
    # out: until that first size report arrives the raster is aggregated at
    # rasterize's 400x400 default and drawn as visible blocks. That is barely
    # noticeable at full extent but obvious after a rebuild that restored a
    # zoomed view, where the coarse grid is stretched over a small window -- so
    # every change of a dropdown looked pixelated until you nudged the zoom and
    # triggered a proper re-aggregation.
    #
    # The frame is square and capped at `max_width`, so its size is known here.
    # If the window is narrower than the cap the grid is finer than the frame,
    # which costs a little work and looks fine; too few bins is what looks bad.
    # width/height and pixel_ratio are both needed, and passing only one is the
    # bug this replaced. width/height are the *initial* grid, used before the
    # browser reports a frame size. Once it does, holoviews' PlotSize stream
    # takes over and recomputes the grid from the frame's CSS size -- which is
    # half the physical pixels on a Retina display, so without pixel_ratio the
    # raster silently drops to 2x2 blocks the moment the first size report
    # lands. With both, either path lands on the same resolution.
    #
    # width/height are in CSS pixels, like the frame size the stream reports:
    # pixel_ratio multiplies whichever of the two is in play, so pre-scaling
    # them here would square the factor and aggregate at 3200 instead of 1600.
    grid = max(64, int(max_width))
    points = hv.Points(df, ["x", "y"], vdims=vdims)
    raster = hd.rasterize(
        points,
        aggregator=aggregator,
        width=grid,
        height=grid,
        pixel_ratio=pixel_ratio,
    ).apply(_shade)

    overlay = _hover_overlay(
        df, color_by, kind, cmap, hover_sample, seed, light_bg, hover_frame,
        glyph_size=hover_size, alpha=hover_alpha, hit_size=hover_reach,
    )
    hooks = []
    if tiles is not None and len(tiles) and "thumb" in tiles.columns:
        hooks.append(_tile_hook(tiles, tile_size, tile_alpha))

    # Read the outgoing figure's ranges before building its replacement, then
    # capture the new one for the rebuild after this.
    xlim = ylim = None
    if view_key is not None:
        remember_view(view_key, _LIVE_FIGURE.get(view_key))
        xlim, ylim = _LAST_VIEW.get(view_key, (None, None))
        hooks.append(_capture_hook(view_key))

    combined = _style(
        raster * overlay, background, extra_hooks=hooks, xlim=xlim, ylim=ylim
    )
    return pn.pane.HoloViews(
        combined, sizing_mode="stretch_width", max_width=max_width
    )


# Extra columns worth showing in the tooltip when the frame carries them.
# `image_id` is deliberately absent: it is a 32-character hash that says nothing
# at a glance, and leaving it out keeps it off the wire too. `dt` stays, but is
# rendered as separate Date and Time rows rather than shown raw.
_TOOLTIP_EXTRAS = ("dt", "lat", "lon", "cluster", "patch_index")


def _hover_tool(sample, color_by):
    """A HoverTool listing the columns present, with the crop image if there is one."""
    from bokeh.models import HoverTool

    cols = [c for c in (color_by, *_TOOLTIP_EXTRAS) if c in sample.columns]
    cols = list(dict.fromkeys(cols))

    rows, formatters = [], {}
    for c in cols:
        if c == "thumb":
            continue
        if c == "dt":
            # Split by bokeh's datetime formatter rather than by adding two
            # string columns: the timestamp is already in the data source, so
            # this costs nothing extra on the wire.
            rows.append("<div><b>Date</b>: @{dt}{%F}</div>")
            rows.append("<div><b>Time</b>: @{dt}{%H:%M}</div>")
            formatters["@{dt}"] = "datetime"
        else:
            rows.append(f"<div><b>{c}</b>: @{{{c}}}</div>")

    img = ""
    if "thumb" in sample.columns:
        # Bokeh renders the tooltip as HTML, so a data URI in a column becomes
        # an inline image. width is set on the element rather than left to the
        # crop's own size, so a change of `scale` does not resize the tooltip.
        img = (
            '<div><img src="@thumb" width="132" '
            'style="display:block;margin-bottom:4px;border-radius:2px"></div>'
        )
    return HoverTool(
        tooltips=f'<div style="font-size:11px">{img}{"".join(rows)}</div>',
        formatters=formatters,
    )


def _hover_overlay(df, color_by, kind, cmap, hover_sample, seed, light_bg,
                   hover_frame=None, glyph_size: int = 7, alpha: float = 0.75,
                   hit_size: int = 18):
    """A small sample drawn as real glyphs, on top of the raster.

    Two jobs: something to hover, and a colorbar for continuous columns --
    neither of which a datashaded raster can provide, since it arrives at the
    browser as an image. Client-side, so it stays crisp when you zoom.

    `glyph_size` and `alpha` control how prominent the points look; `hit_size`
    controls how close the cursor has to get, via a transparent glyph beneath
    them. They are separate on purpose -- a comfortable hit target is much
    bigger than a dot you would want covering the data.

    `hover_frame` lets the caller supply the rows, so an expensive frame (one
    carrying patch crops) can be built once and reused across renders.
    """
    import holoviews as hv

    if hover_frame is not None:
        sample = hover_frame
    else:
        n = min(int(hover_sample), len(df))
        if n <= 0:
            sample = df.iloc[:0]
        else:
            sample = df.sample(n, random_state=seed)

    vdims = [
        c
        for c in dict.fromkeys((color_by, "thumb", *_TOOLTIP_EXTRAS))
        if c in sample.columns
    ]
    if not len(sample):
        return hv.Points([], ["x", "y"], vdims=vdims)

    # An invisible larger glyph underneath carries the hover. Bokeh hit-tests
    # against a glyph's geometry, not its rendered pixels, so a fully
    # transparent disc is a hit target -- which decouples how big these points
    # *look* from how close the cursor has to get. Without it the only way to
    # make hovering easier is to draw fatter dots over the data.
    halo = hv.Points(sample, ["x", "y"], vdims=vdims).opts(
        size=hit_size,
        alpha=0.0,
        tools=[_hover_tool(sample, color_by)],
        show_legend=False,
    )

    # A contrasting outline is what makes these read against a raster that is
    # bright in places and dark in others; a flat dot disappears into whichever
    # matches it.
    edge = "white" if light_bg else "black"
    shared = dict(
        size=glyph_size,
        alpha=alpha,
        line_color=edge,
        line_width=1,
        line_alpha=min(1.0, alpha + 0.2),
    )
    pts = hv.Points(sample, ["x", "y"], vdims=vdims)

    if kind == "continuous":
        return halo * pts.opts(
            color=color_by, cmap=resolve_cmap(cmap), colorbar=True, **shared
        )

    # Neutral for density and categorical alike. For categorical that is
    # deliberate: `hd.shade` builds its own hidden legend plot and reaches into
    # it for a colour mapper, and a second category-coloured layer gives
    # holoviews two candidates -- it picks the wrong one and dies with
    # KeyError: 'color_color_mapper', intermittently, depending on which plot
    # was built first. The raster underneath already shows each point's
    # category, and its legend covers every row rather than this sample.
    return halo * pts.opts(color="black" if light_bg else "white", **shared)


def _tile_data(tiles):
    """Tile frame -> the three columns the ImageURL glyph needs."""
    if tiles is None or not len(tiles):
        return dict(url=[], x=[], y=[])
    return dict(
        url=list(tiles["thumb"]), x=list(tiles["x"]), y=list(tiles["y"])
    )


def _tile_hook(tiles, size_px: int, alpha: float):
    """Hook drawing patch crops onto the plot at their own coordinates.

    A raw bokeh ImageURL glyph rather than holoviews RGB elements, for two
    reasons: it is one renderer regardless of tile count, and `w_units="screen"`
    keeps every tile the same size on screen however far you zoom. In data units
    tiles would swell to fill the view on the way in and vanish on the way out.

    Added last, so the tiles sit above the raster and the hover glyphs.
    """
    from bokeh.models import ColumnDataSource

    def hook(plot, element):
        fig = plot.state

        # Holoviews calls hooks again on every update, so adding the glyph
        # unconditionally stacks a new renderer on each re-render -- the tiles
        # would darken as the duplicates pile up at the same alpha. Keep a
        # handle and refresh its data instead.
        existing = plot.handles.get("patch_tiles")
        if existing is not None:
            existing.data_source.data = _tile_data(tiles)
            existing.glyph.update(w=size_px, h=size_px, global_alpha=alpha)
            return

        renderer = fig.image_url(
            url="url",
            x="x",
            y="y",
            w=size_px,
            h=size_px,
            w_units="screen",
            h_units="screen",
            anchor="center",
            global_alpha=alpha,
            source=ColumnDataSource(_tile_data(tiles)),
        )
        plot.handles["patch_tiles"] = renderer

    return hook


def _style(element, background: str, extra_hooks=(), xlim=None, ylim=None):
    """Frame styling shared by every scatter: grid, sizing, 1:1 data scales."""
    # Drawn at "overlay" level. The default is "underlay", which puts the grid
    # beneath the datashaded image -- and that image covers the whole frame, so
    # an underlaid grid is invisible no matter its alpha.
    gridstyle = {
        "grid_line_color": "black" if background.lower() == "white" else "white",
        "grid_line_alpha": 0.12,
        "grid_level": "overlay",
    }

    # 1:1 data scale, set straight on the bokeh figure. The holoviews option for
    # this is data_aspect=1 and it is the same constraint -- but routing it
    # through .opts() sends it into holoviews' layout solver, which downgrades
    # the plot to sizing_mode="fixed" and logs "responsive mode could not be
    # enabled". The hook runs after sizing is settled, so bokeh gets
    # match_aspect and the responsive width survives.
    def _equal_scales(plot, element):
        plot.state.match_aspect = True

    return element.opts(
        # responsive="width" + aspect=1 resolves to bokeh
        # sizing_mode="scale_width" with aspect_ratio=1: the frame takes the
        # full width of the cell and derives its height.
        responsive="width",
        aspect=1,
        hooks=[_equal_scales, *extra_hooks],
        bgcolor=background,
        show_grid=True,
        gridstyle=gridstyle,
        # padding=0 is load-bearing: with the default padding, every new raster
        # extent gets padded again, which nudges the axis range, which fires the
        # range stream, which re-aggregates -- a loop that never settles.
        padding=0,
        xlabel="UMAP 1",
        ylabel="UMAP 2",
        active_tools=["box_zoom"],
        tools=["box_zoom", "wheel_zoom", "pan", "reset"],
        # None leaves the axes at the data's own extent; a remembered view puts
        # the replacement figure back where the last one was looking.
        **({"xlim": xlim} if xlim else {}),
        **({"ylim": ylim} if ylim else {}),
    )
