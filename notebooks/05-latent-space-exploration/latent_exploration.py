import marimo

__generated_with = "0.23.13"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import IPython  # must be imported before lancedb to avoid circular import via tqdm→ipywidgets

    from helpers import PatchExperiment
    from helpers.controls import (
        background_dropdown,
        color_by_dropdown,
        color_picker,
        colormap_dropdown,
        display_options,
        display_panel,
        display_widgets,
        hover_point_controls,
        hover_sample_slider,
        projection_picker,
        resolution_slider,
        scatter_panel,
        thumbnail_checkbox,
        thumbnail_limit_slider,
        tile_controls,
        widget_values,
        width_buttons,
    )
    from helpers.data import list_experiments

    # holoviews/panel/datashader stay inside helpers.scatter -- the notebook
    # only ever handles the pane it returns.
    from helpers.region import explorer
    from helpers.scatter import default_cmap, scatter_pane

    return (
        PatchExperiment,
        background_dropdown,
        color_by_dropdown,
        color_picker,
        colormap_dropdown,
        default_cmap,
        display_options,
        display_panel,
        display_widgets,
        explorer,
        hover_point_controls,
        hover_sample_slider,
        list_experiments,
        mo,
        projection_picker,
        resolution_slider,
        scatter_pane,
        scatter_panel,
        thumbnail_checkbox,
        thumbnail_limit_slider,
        tile_controls,
        widget_values,
        width_buttons,
    )


@app.cell
def _(mo):
    embedding_db_path = mo.ui.text(
        value="/Users/ncheruku/Documents/Work/sample_data/data/lancedb/experiments/era5",
        placeholder="e.g. /data/lancedb/experiments/era5",
        label="Experiments DB path",
        full_width=True,
    )
    embedding_db_path
    return (embedding_db_path,)


@app.cell
def _(embedding_db_path, list_experiments, mo):
    _experiments = list_experiments(embedding_db_path.value)
    experiment_selector = mo.ui.dropdown(
        options=_experiments,
        value=_experiments[0] if _experiments else None,
        label="Experiment",
    )
    experiment_selector
    return (experiment_selector,)


@app.cell
def _(PatchExperiment, embedding_db_path, experiment_selector):
    if not embedding_db_path.value or experiment_selector.value is None:
        exp = None
    else:
        exp = PatchExperiment.open(embedding_db_path.value, experiment_selector.value)
    return (exp,)


@app.cell
def _(exp, mo):
    mo.md(
        exp.summary()
        if exp is not None
        else "Enter a DB path and pick an experiment."
    )
    return


@app.cell
def _(color_picker, display_panel, display_widgets):
    # Tile appearance for the region gallery below the projection. Both must be
    # cell-level names: marimo syncs widget values back to the instances it can
    # see in a cell, and only a UIElement group (not a plain dict) makes
    # dependent cells re-run. The picker is separate because anywidgets cannot
    # live inside mo.ui.dictionary.
    display = display_widgets()
    border_color = color_picker()
    display_panel(display, border_color)
    return border_color, display


@app.cell
def _(mo):
    mo.md("""
    ## Projection

    The 2-D projection of every patch in the table, datashaded.

    **Drag a box** to pull the patches from that region into the gallery below.
    The box queries the whole table, not the points drawn for hover, so a
    selection can hold hundreds of thousands of patches — page through them.

    **Zoom re-aggregates.** Wheel-zoom sends the new view limits back to Python
    and datashader rebuilds the raster for that window, so detail keeps
    resolving as you go in rather than turning into big pixels.
    """)
    return


@app.cell
def _(exp, projection_picker):
    _tables = exp.list_projections() if exp is not None else []
    projection_selector = projection_picker(_tables)
    projection_selector
    return (projection_selector,)


@app.cell
def _(exp, projection_selector):
    projection = (
        None
        if exp is None or projection_selector.value is None
        else exp.load_projection(projection_selector.value)
    )
    return (projection,)


@app.cell
def _(mo, projection):
    mo.md(
        projection.summary()
        if projection is not None
        else "No projection table in this experiment — expected one named `umap_*` "
        "beside the embedding tables."
    )
    return


@app.cell
def _(
    background_dropdown,
    color_by_dropdown,
    hover_point_controls,
    hover_sample_slider,
    mo,
    projection,
    resolution_slider,
    scatter_panel,
    thumbnail_checkbox,
    thumbnail_limit_slider,
    width_buttons,
):
    scatter_color_by = color_by_dropdown(
        projection.color_by_options() if projection is not None else ["density"]
    )
    scatter_background = background_dropdown()
    scatter_hover = hover_sample_slider()
    scatter_thumbs = thumbnail_checkbox()
    scatter_thumb_limit = thumbnail_limit_slider()
    scatter_resolution = resolution_slider()
    scatter_points = hover_point_controls()

    # The width lives in state because two buttons write to it. This cell must
    # not *read* it -- marimo does not re-run the cell that owns a state setter,
    # so the readout lives in the next cell instead.
    get_plot_width, set_plot_width = mo.state(800)
    narrower, wider = width_buttons(get_plot_width, set_plot_width)

    scatter_panel(
        scatter_color_by, scatter_background, scatter_hover,
        scatter_thumbs, scatter_thumb_limit, scatter_resolution,
        scatter_points,
    )
    return (
        get_plot_width,
        narrower,
        scatter_background,
        scatter_color_by,
        scatter_hover,
        scatter_points,
        scatter_resolution,
        scatter_thumb_limit,
        scatter_thumbs,
        wider,
    )


@app.cell
def _(
    colormap_dropdown,
    default_cmap,
    get_plot_width,
    mo,
    narrower,
    projection,
    scatter_background,
    scatter_color_by,
    scatter_panel,
    wider,
):
    # The colormap widget is rebuilt here rather than alongside the other
    # controls because its *options* depend on the colour-by kind. The choice
    # resets on a categorical <-> continuous switch, which is intended: the
    # previous palette would not have applied to the new aggregation.
    _kind = (
        projection.kind(scatter_color_by.value) if projection is not None else "density"
    )
    _light = scatter_background.value == "white"
    scatter_cmap = colormap_dropdown(_kind, value=default_cmap(_kind, _light))

    # The width readout has to be read in a cell that does not own the setter,
    # or it would never refresh.
    scatter_panel(
        scatter_cmap,
        mo.md("**Width**"),
        narrower,
        wider,
        mo.md(f"`{get_plot_width()} px`"),
    )
    return (scatter_cmap,)


@app.cell
def _(tile_controls):
    scatter_tiles = tile_controls()
    scatter_tiles
    return (scatter_tiles,)


@app.cell
def _(exp, projection, scatter_tiles):
    # Own cell, like the hover frame: the crops survive every change that is not
    # the tile count itself. Picked once over the full extent -- zooming does not
    # repick them.
    tile_frame = (
        None
        if projection is None or not scatter_tiles.value["show"]
        else exp.tile_frame(projection, n=int(scatter_tiles.value["count"]))
    )
    return (tile_frame,)


@app.cell
def _(exp, projection, scatter_hover, scatter_thumb_limit, scatter_thumbs):
    # Its own cell so the crops survive a change of colour, colormap, background
    # or width: marimo only re-runs this when the projection, the sample size or
    # the thumbnail toggle actually changes. Building it inside the plot cell
    # would rebuild every crop on every nudge of a control.
    hover_frame = (
        None
        if projection is None
        else exp.hover_frame(
            projection,
            n=int(scatter_hover.value),
            thumbnails=bool(scatter_thumbs.value),
            max_thumbnails=int(scatter_thumb_limit.value),
        )
    )
    return (hover_frame,)


@app.cell
def _(
    get_plot_width,
    hover_frame,
    projection,
    scatter_background,
    scatter_cmap,
    scatter_color_by,
    scatter_pane,
    scatter_points,
    scatter_resolution,
    scatter_tiles,
    tile_frame,
):
    # The plot is composed with the region gallery below rather than rendered on
    # its own: the bounds stream fires in Python without re-running any marimo
    # cell, so the gallery has to live in the same panel Column and update
    # through panel. See helpers/region.py.
    plot_pane = (
        None
        if projection is None
        else scatter_pane(
            projection,
            color_by=scatter_color_by.value,
            cmap=scatter_cmap.value,
            background=scatter_background.value,
            hover_frame=hover_frame,
            tiles=tile_frame,
            tile_size=int(scatter_tiles.value["size"]),
            tile_alpha=float(scatter_tiles.value["alpha"]),
            pixel_ratio=float(scatter_resolution.value),
            hover_size=int(scatter_points.value["size"]),
            # Zero opacity rather than dropping the layer: the colorbar for a
            # continuous colour-by hangs off these glyphs, and hovering still
            # works because the hit target is its own, already-invisible layer.
            hover_alpha=(
                float(scatter_points.value["opacity"])
                if scatter_points.value["show"]
                else 0.0
            ),
            hover_reach=int(scatter_points.value["reach"]),
            max_width=int(get_plot_width()),
            # Keeps the zoom across rebuilds; keyed on the table so
            # switching projection starts at the full extent.
            view_key=projection.name,
        )
    )
    return (plot_pane,)


@app.cell
def _(
    border_color,
    display,
    display_options,
    exp,
    explorer,
    mo,
    plot_pane,
    projection,
    widget_values,
):
    def _render_region(bounds, page, per_page):
        """One page of the selected region, as gallery HTML plus the total."""
        offsets = exp.region_offsets(projection, bounds)
        total = len(offsets)
        start = page * per_page
        sample = exp.patch_sample(projection, offsets[start : start + per_page])
        opts = display_options(widget_values(display, border_color))
        # n_examples caps the gallery, and the page is already the right size --
        # pin it so the gallery does not sub-sample the page.
        opts.n_examples = per_page
        return exp.gallery(sample, opts), total

    scatter = (
        mo.md("*Load an experiment with a projection table to see the map.*")
        if plot_pane is None
        else mo.ui.panel(explorer(plot_pane, _render_region))
    )
    scatter
    return


if __name__ == "__main__":
    app.run()
