import marimo

__generated_with = "0.23.13"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import IPython  # must be imported before lancedb to avoid circular import via tqdm→ipywidgets

    from helpers import PatchExperiment
    from helpers.controls import (
        color_picker,
        display_options,
        display_panel,
        display_widgets,
        loader_controls,
        widget_values,
    )
    from helpers.data import list_experiments

    return (
        PatchExperiment,
        color_picker,
        display_options,
        display_panel,
        display_widgets,
        list_experiments,
        loader_controls,
        mo,
        widget_values,
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


if __name__ == "__main__":
    app.run()
