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
        hover_sample_slider,
        loader_controls,
        projection_picker,
        scatter_panel,
        thumbnail_checkbox,
        thumbnail_limit_slider,
        widget_values,
        width_buttons,
    )
    from helpers.data import list_experiments

    # holoviews/panel/datashader stay inside helpers.scatter -- the notebook
    # only ever handles the pane it returns.
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
        hover_sample_slider,
        list_experiments,
        loader_controls,
        mo,
        projection_picker,
        scatter_pane,
        scatter_panel,
        thumbnail_checkbox,
        thumbnail_limit_slider,
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
def _(loader_controls):
    # Applied immediately: these drive a multi-second read of the patch table,
    # so they are kept out of the display form below.
    loader = loader_controls()
    loader
    return (loader,)


@app.cell
def _(exp, loader):
    sample = (
        None
        if exp is None
        else exp.load_patches(
            limit=int(loader.value["sample_size"]),
            random_sample=loader.value["random_sample"],
        )
    )
    return (sample,)


@app.cell
def _(exp, mo, sample):
    mo.md(
        exp.summary(sample)
        if sample is not None
        else "Enter a DB path and pick an experiment to load patches."
    )
    return


@app.cell
def _(color_picker, display_panel, display_widgets):
    # Both must be cell-level names: marimo syncs widget values back to the
    # instances it can see in a cell, and only a UIElement group (not a plain
    # dict) makes dependent cells re-run. The picker is separate because
    # anywidgets cannot live inside mo.ui.dictionary.
    display = display_widgets()
    border_color = color_picker()
    display_panel(display, border_color)
    return border_color, display


@app.cell
def _(border_color, display, display_options, exp, mo, sample, widget_values):
    mo.Html(
        exp.gallery(sample, display_options(widget_values(display, border_color)))
        if sample is not None
        else "<em>Load an experiment to see patch crops.</em>"
    )
    return


@app.cell
def _(mo):
    mo.md("""
    ## Projection

    The 2-D projection of these patches, datashaded. This reads the **whole**
    projection table, independent of the sample size above — that slider limits
    an expensive 768-dim embedding read, which does not apply to a table of
    coordinates.

    **Zoom re-aggregates.** Box-zoom or wheel-zoom sends the new view limits
    back to Python and datashader rebuilds the raster for that window, so detail
    keeps resolving as you go in rather than turning into big pixels.
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
    hover_sample_slider,
    mo,
    projection,
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

    # The width lives in state because two buttons write to it. This cell must
    # not *read* it -- marimo does not re-run the cell that owns a state setter,
    # so the readout lives in the next cell instead.
    get_plot_width, set_plot_width = mo.state(800)
    narrower, wider = width_buttons(get_plot_width, set_plot_width)

    scatter_panel(
        scatter_color_by, scatter_background, scatter_hover,
        scatter_thumbs, scatter_thumb_limit,
    )
    return (
        get_plot_width,
        narrower,
        scatter_background,
        scatter_color_by,
        scatter_hover,
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
    mo,
    projection,
    scatter_background,
    scatter_cmap,
    scatter_color_by,
    scatter_pane,
):
    if projection is None:
        scatter = mo.md("*Load an experiment with a projection table to see the map.*")
    else:
        scatter = mo.ui.panel(
            scatter_pane(
                projection,
                color_by=scatter_color_by.value,
                cmap=scatter_cmap.value,
                background=scatter_background.value,
                hover_frame=hover_frame,
                max_width=int(get_plot_width()),
            )
        )
    scatter
    return


if __name__ == "__main__":
    app.run()
