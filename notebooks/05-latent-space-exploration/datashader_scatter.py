import marimo

__generated_with = "0.23.13"
app = marimo.App(width="full")


@app.cell
def _():
    import itertools

    import marimo as mo
    import numpy as np
    import pandas as pd
    import colorcet as cc
    import datashader as ds
    import holoviews as hv
    import holoviews.operation.datashader as hd
    import panel as pn

    hv.extension("bokeh")
    pn.extension()
    return cc, ds, hd, hv, itertools, mo, np, pd, pn


@app.cell
def _(mo):
    mo.md("""
    # Datashading 1.5M points

    Synthetic sanity check for the datashader rendering path, at the scale we
    expect from real patch projections.

    **Zoom re-aggregates.** The plot goes through `mo.ui.panel`, so box-zoom and
    wheel-zoom send the new view limits back to Python and datashader rebuilds the
    raster for that window. Detail keeps resolving as you go in.
    """)
    return


@app.cell
def _(mo):
    n_points = mo.ui.slider(
        start=100_000,
        stop=5_000_000,
        step=100_000,
        value=1_500_000,
        label="Points",
        show_value=True,
    )
    n_blobs = mo.ui.slider(
        start=1, stop=20, step=1, value=5, label="Blobs", show_value=True
    )
    seed = mo.ui.number(value=42, label="Seed")
    mo.vstack([n_points, n_blobs, seed])
    return n_blobs, n_points, seed


@app.cell
def _(cc, n_blobs, n_points, np, pd, seed):
    rng = np.random.default_rng(int(seed.value))

    _n = int(n_points.value)
    _k = int(n_blobs.value)

    centers = rng.uniform(-10, 10, size=(_k, 2))
    sigmas = rng.uniform(0.5, 2.5, size=(_k, 2))

    # Even split across blobs, remainder into the last one.
    counts = np.full(_k, _n // _k)
    counts[-1] += _n - counts.sum()

    labels = np.repeat(np.arange(_k), counts)
    xy = rng.normal(loc=centers[labels], scale=sigmas[labels])

    # A few extra per-point variables to color by. `radius` is distance from the
    # point's own blob center; `field` is a smooth spatial signal, so the two
    # shade very differently and make aggregator choice visible.
    _offsets = xy - centers[labels]
    df = pd.DataFrame(
        {
            "x": xy[:, 0],
            "y": xy[:, 1],
            "blob": pd.Categorical([f"blob {i}" for i in labels]),
            "radius": np.hypot(_offsets[:, 0], _offsets[:, 1]),
            "field": np.sin(xy[:, 0] / 2) * np.cos(xy[:, 1] / 2),
        }
    )
    # One explicit category -> color map, shared by the raster and the overlay.
    # Left implicit, datashader colors by category order and holoviews by the
    # sample's own ordering, and the two layers disagree.
    blob_colors = dict(zip(df["blob"].cat.categories, cc.glasbey_light))

    df
    return blob_colors, df


@app.cell
def _(mo):
    color_by = mo.ui.dropdown(
        options=["density", "blob", "radius", "field"],
        value="density",
        label="Color by",
    )
    sample_size = mo.ui.slider(
        start=0,
        stop=5000,
        step=250,
        value=1000,
        label="Hover sample",
        show_value=True,
    )
    # Dict options so the label is readable but `.value` is the literal color
    # bokeh wants. Dark gray rather than black: the low end of the fire ramp is
    # near-black, so pure black hides the sparse tail of every blob.
    background = mo.ui.dropdown(
        options={"Dark gray": "#2b2b2b", "White": "white"},
        value="Dark gray",
        label="Background",
    )
    mo.vstack([color_by, background, sample_size])
    return background, color_by, sample_size


@app.cell
def _(background, blob_colors, cc, color_by, df, ds, hd, hv, itertools):
    _light_bg = background.value == "white"

    # Each variable needs its own aggregator: counts for density, a per-category
    # count so overlapping blobs blend instead of overpainting, and a mean for
    # the continuous columns.
    _var = color_by.value
    if _var == "density":
        # fire runs black -> white, so it only reads against a dark canvas. On
        # white, flip it: dense stays dark, sparse fades into the background.
        _cmap = list(reversed(cc.fire)) if _light_bg else cc.fire
        _aggregator = ds.count()
        _shade_kwargs = dict(cmap=_cmap, cnorm="log")
    elif _var == "blob":
        _aggregator = ds.count_cat("blob")
        _shade_kwargs = dict(color_key=blob_colors, cnorm="log")
    else:
        # cc.bmy / "bmy" are the same palette, so the raster and the overlay
        # glyphs below stay on one color scale.
        _aggregator = ds.mean(_var)
        _shade_kwargs = dict(cmap=cc.bmy, cnorm="linear")

    _n_aggregations = itertools.count(1)

    def _shade(agg):
        # Logs every re-aggregation to the marimo terminal, so zoom behaviour is
        # visible without guessing from the picture. `agg` is a holoviews Image
        # element here, so the window comes from its dimension ranges. If the
        # counter climbs while you are not touching the plot, something is
        # re-triggering the range stream.
        _x0, _x1 = agg.range("x")
        _y0, _y1 = agg.range("y")
        print(
            f"[datashade] #{next(_n_aggregations)} {_var} window "
            f"x=({_x0:.2f}, {_x1:.2f}) y=({_y0:.2f}, {_y1:.2f})",
            flush=True,
        )
        return hd.shade(agg, **_shade_kwargs)

    # rasterize stays lazy: it re-runs with the current view limits on every
    # zoom/pan. No width/height, because the frame width is now the cell's width
    # and is only known after the browser lays the page out — the grid comes from
    # holoviews' PlotSize stream instead, which also means a re-aggregation on
    # every window resize and on every press of the height buttons.
    points = hv.Points(df, ["x", "y"], vdims=["blob", "radius", "field"])
    raster = hd.rasterize(points, aggregator=_aggregator).apply(_shade)
    return (raster,)


@app.cell
def _(background, blob_colors, color_by, df, hv, sample_size):
    # Cheap foreground: real glyphs for a random sample, purely so the plot has
    # something to hover. This layer is client-side, so it stays crisp on zoom.
    _cols = ["blob", "radius", "field"]
    if sample_size.value == 0:
        overlay = hv.Points([], ["x", "y"], vdims=_cols)
    else:
        _sample = df.sample(min(int(sample_size.value), len(df)), random_state=0)
        overlay = hv.Points(_sample, ["x", "y"], vdims=_cols)

    _var = color_by.value
    _shared = dict(size=4, alpha=0.5, tools=["hover"])
    if _var == "density":
        # The density overlay has no data-driven color, so it only has to stay
        # visible — invert it with the canvas.
        _glyph = "black" if background.value == "white" else "white"
        overlay = overlay.opts(color=_glyph, **_shared)
    elif _var == "blob":
        # Categorical color here gives the legend datashader's raster cannot.
        overlay = overlay.opts(
            color="blob", cmap=blob_colors, show_legend=True, **_shared
        )
    else:
        overlay = overlay.opts(color=_var, cmap="bmy", colorbar=True, **_shared)
    return (overlay,)


@app.cell
def _(background, mo, overlay, pn, raster):
    # A faint grid, drawn at "overlay" level. The default is "underlay", which
    # puts it beneath the datashaded image — and that image covers the whole
    # frame, so an underlaid grid is invisible no matter its alpha.
    _grid = {
        "grid_line_color": "black" if background.value == "white" else "white",
        "grid_line_alpha": 0.12,
        "grid_level": "overlay",
    }

    # 1:1 data scale, applied straight to the bokeh figure. The holoviews option
    # for it is data_aspect=1, and this is the same constraint — but routing it
    # through .opts() sends it into holoviews' layout solver, which downgrades
    # the plot to sizing_mode="fixed" and logs "responsive mode could not be
    # enabled". The hook runs after holoviews has finished sizing, so bokeh gets
    # match_aspect and the auto width survives.
    def _equal_scales(plot, element):
        plot.state.match_aspect = True

    # responsive="width" + aspect=1 resolves to bokeh sizing_mode="scale_width"
    # with aspect_ratio=1: the frame takes the full width of the cell and derives
    # its height. Height is not pinned to anything here.
    combined = (raster * overlay).opts(
        responsive="width",
        aspect=1,
        hooks=[_equal_scales],
        bgcolor=background.value,
        show_grid=True,
        gridstyle=_grid,
        # padding=0 is load-bearing: with the default padding, every new raster
        # extent gets padded again, which nudges the axis range, which fires the
        # range stream, which re-aggregates — a loop that never settles.
        padding=0,
        xlabel="x",
        ylabel="y",
        active_tools=["box_zoom"],
        tools=["box_zoom", "wheel_zoom", "pan", "reset"],
    )
    # The pane has to stretch as well, or the figure has no width to fill.
    plot = mo.ui.panel(pn.pane.HoloViews(combined, sizing_mode="stretch_width"))
    plot
    return


@app.cell
def _(color_by, df, mo, sample_size):
    mo.md(f"""
    **Points:** {len(df):,}  ·  **Colored by:** {color_by.value}  ·
    **Hover sample:** {sample_size.value:,}

    The frame fills the cell width. The x and y axes share one scale, so shapes
    are not skewed by the frame being wider than it is tall. See the
    `[datashade]` lines in the terminal for the window of each re-aggregation.

    Box-zoom into a dense region and the raster re-aggregates at that window, so
    zooming in buys real detail rather than magnified pixels. Reset returns to
    full extent.
    """)
    return


if __name__ == "__main__":
    app.run()
