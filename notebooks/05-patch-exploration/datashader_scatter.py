import marimo

__generated_with = "0.23.13"
app = marimo.App()


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
    plot_width = mo.ui.slider(
        start=400, stop=1600, step=50, value=900, label="Width", show_value=True
    )
    plot_height = mo.ui.slider(
        start=300, stop=1000, step=50, value=600, label="Height", show_value=True
    )
    sample_size = mo.ui.slider(
        start=0,
        stop=5000,
        step=250,
        value=1000,
        label="Hover sample",
        show_value=True,
    )
    mo.vstack([color_by, plot_width, plot_height, sample_size])
    return color_by, plot_height, plot_width, sample_size


@app.cell
def _(
    blob_colors,
    cc,
    color_by,
    df,
    ds,
    hd,
    hv,
    itertools,
    plot_height,
    plot_width,
):
    # Each variable needs its own aggregator: counts for density, a per-category
    # count so overlapping blobs blend instead of overpainting, and a mean for
    # the continuous columns.
    _var = color_by.value
    if _var == "density":
        _aggregator = ds.count()
        _shade_kwargs = dict(cmap=cc.fire, cnorm="log")
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
    # zoom/pan. width/height pin the aggregation grid, so holoviews never has to
    # ask the browser for the frame size.
    points = hv.Points(df, ["x", "y"], vdims=["blob", "radius", "field"])
    raster = hd.rasterize(
        points,
        aggregator=_aggregator,
        width=int(plot_width.value),
        height=int(plot_height.value),
    ).apply(_shade)
    return (raster,)


@app.cell
def _(blob_colors, color_by, df, hv, sample_size):
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
        overlay = overlay.opts(color="white", **_shared)
    elif _var == "blob":
        # Categorical color here gives the legend datashader's raster cannot.
        overlay = overlay.opts(
            color="blob", cmap=blob_colors, show_legend=True, **_shared
        )
    else:
        overlay = overlay.opts(color=_var, cmap="bmy", colorbar=True, **_shared)
    return (overlay,)


@app.cell
def _(mo, overlay, plot_height, plot_width, pn, raster):
    # frame_width/frame_height size the inner plot area. Sizing the frame rather
    # than the whole figure keeps the axes and colorbar from eating into the
    # canvas, and keeps the aggregation grid matched to the pixels on screen.
    combined = (raster * overlay).opts(
        frame_width=int(plot_width.value),
        frame_height=int(plot_height.value),
        bgcolor="black",
        # padding=0 is load-bearing: with the default padding, every new raster
        # extent gets padded again, which nudges the axis range, which fires the
        # range stream, which re-aggregates — a loop that never settles.
        padding=0,
        xlabel="x",
        ylabel="y",
        active_tools=["box_zoom"],
        tools=["box_zoom", "wheel_zoom", "pan", "reset"],
    )
    plot = mo.ui.panel(pn.pane.HoloViews(combined, sizing_mode="fixed"))
    plot
    return


@app.cell
def _(color_by, df, mo, plot_height, plot_width, sample_size):
    mo.md(f"""
    **Points:** {len(df):,}  ·  **Colored by:** {color_by.value}  ·
    **Grid:** {plot_width.value}×{plot_height.value}  ·
    **Hover sample:** {sample_size.value:,}

    Box-zoom into a dense region and the raster re-aggregates at that window —
    the grid stays {plot_width.value}×{plot_height.value} pixels, so zooming in
    buys real detail rather than magnified pixels. Reset returns to full extent.
    """)
    return


if __name__ == "__main__":
    app.run()
