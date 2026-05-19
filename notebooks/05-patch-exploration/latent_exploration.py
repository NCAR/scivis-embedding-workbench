import marimo

__generated_with = "0.23.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import IPython  # must be imported before lancedb to avoid circular import via tqdm→ipywidgets
    import lancedb
    import numpy as np

    from helpers.data import list_experiments, open_experiment

    return list_experiments, mo, open_experiment


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
    if _experiments:
        experiment_selector = mo.ui.dropdown(
            options=_experiments, value=_experiments[0], label="Experiment"
        )
    else:
        experiment_selector = mo.ui.dropdown(options=[], label="Experiment")
    experiment_selector
    return (experiment_selector,)


@app.cell
def _(embedding_db_path, experiment_selector, open_experiment):
    if not embedding_db_path.value or experiment_selector.value is None:
        config, patch_emb_tbl = None, None
    else:
        config, patch_emb_tbl = open_experiment(
            embedding_db_path.value, experiment_selector.value
        )
    return config, patch_emb_tbl


@app.cell
def _(patch_emb_tbl):
    # Cheap access check — no full materialization. Replace this cell with the
    # real loader (e.g. cuML UMAP input) once access is confirmed.
    if patch_emb_tbl is None:
        n_patches, sample_row = None, None
    else:
        n_patches = patch_emb_tbl.count_rows()
        sample_row = patch_emb_tbl.head(1).to_pylist()[0]
    return n_patches, sample_row


@app.cell
def _(config, mo, n_patches, sample_row):
    if n_patches is None:
        status = mo.callout(
            mo.md("Enter a DB path and pick an experiment to load patches."),
            kind="info",
        )
    else:
        _dim = len(sample_row["embedding"])
        status = mo.md(
            f"**Model:** `{config.get('model_name', '?')}`  ·  "
            f"**Patches:** {n_patches:,}  ·  "
            f"**Embedding dim:** {_dim}  ·  "
            f"**Sample patch:** `image_id={sample_row['image_id']}`, "
            f"`patch_index={sample_row['patch_index']}`"
        )
    status
    return


if __name__ == "__main__":
    app.run()
