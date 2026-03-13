import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import lancedb
    import io
    import math
    import pandas as pd
    import matplotlib.pyplot as plt
    from PIL import Image, ImageDraw

    return Image, ImageDraw, io, lancedb, math, mo, pd, plt


@app.function
def list_experiments(db_path: str) -> list:
    """Scan a LanceDB directory for experiment names by finding *_config.lance dirs."""
    from pathlib import Path
    p = Path(db_path)
    if not p.exists() or not p.is_dir():
        return []
    return [
        d.name[: -len("_config.lance")]
        for d in sorted(p.iterdir())
        if d.is_dir() and d.name.endswith("_config.lance")
    ]


@app.function
def load_config_dict(db, config_table_name: str) -> dict:
    """Load a config table into a Python dict keyed by the 'key' column."""
    tbl = db.open_table(config_table_name)
    df = tbl.to_pandas()
    return dict(zip(df["key"], df["value"]))


@app.function
def resolve_source_path(experiments_db_path: str, source_path_from_config: str) -> str:
    """
    Resolve source_path from config to an absolute path.
    - If already absolute and exists → return as-is
    - If relative → walk up from experiments_db_path until the join exists
    - Returns None if unresolvable
    """
    from pathlib import Path
    p = Path(source_path_from_config)
    if p.is_absolute():
        return str(p) if p.exists() else None
    candidate = Path(experiments_db_path)
    for _ in range(10):
        candidate = candidate.parent
        resolved = candidate / source_path_from_config
        if resolved.exists():
            return str(resolved)
    return None


@app.function
def fetch_image_by_filename(src_img_tbl, filename: str):
    """Load a PIL image from src_img_tbl by filename."""
    import io
    from PIL import Image
    row = src_img_tbl.search().where(f"filename = '{filename}'").limit(1).to_pandas().iloc[0]
    return Image.open(io.BytesIO(row["image_blob"])).convert("RGBA")


@app.function
def overlay_grid(img, patch_size: int = 16, color=(255, 255, 255, 50), width: int = 1):
    """Draw a patch grid overlay on a PIL RGBA image."""
    from PIL import Image, ImageDraw
    w, h = img.size
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    for x in range(0, w + 1, patch_size):
        d.line([(x, 0), (x, h)], fill=color, width=width)
    for y in range(0, h + 1, patch_size):
        d.line([(0, y), (w, y)], fill=color, width=width)
    return Image.alpha_composite(img, overlay)


@app.function
def patch_box_from_index(patch_index: int, img_w: int = 256, img_h: int = 256, patch_size: int = 16):
    """Return (x0, y0, x1, y1) bounding box for a patch index."""
    grid_w = img_w // patch_size
    grid_h = img_h // patch_size
    n_patches = grid_w * grid_h
    if patch_index < 0 or patch_index >= n_patches:
        raise ValueError(f"patch_index {patch_index} out of range [0, {n_patches - 1}]")
    row = patch_index // grid_w
    col = patch_index % grid_w
    x0 = col * patch_size
    y0 = row * patch_size
    return (x0, y0, x0 + patch_size, y0 + patch_size)


@app.function
def highlight_patch(img, box, fill=(255, 0, 0, 0), outline=(255, 0, 0, 220), outline_width: int = 2):
    """Draw a highlighted rectangle over a patch on a PIL RGBA image."""
    from PIL import Image, ImageDraw
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    d.rectangle(box, fill=fill, outline=outline, width=outline_width)
    return Image.alpha_composite(img, overlay)


@app.cell
def _(mo):
    embedding_db_path = mo.ui.text(
        value="",
        placeholder="e.g. /data/lancedb/experiments/era5",
        label="Experiments DB path",
        full_width=True,
    )
    embedding_db_path
    return (embedding_db_path,)


@app.cell
def _(embedding_db_path, mo):
    _experiments = list_experiments(embedding_db_path.value)

    if not _experiments:
        experiment_selector = mo.ui.dropdown(options=[], label="Experiment")
        mo.stop(True, mo.callout(
            mo.md(f"No experiments found at `{embedding_db_path.value}`"),
            kind="warn",
        ))
    else:
        experiment_selector = mo.ui.dropdown(
            options=_experiments,
            value=_experiments[0],
            label="Experiment",
        )

    experiment_selector
    return (experiment_selector,)


@app.cell
def _(embedding_db_path, experiment_selector, lancedb, mo):
    mo.stop(experiment_selector.value is None, mo.callout(
        mo.md("Select an experiment above."), kind="info"
    ))

    _exp = experiment_selector.value
    db = lancedb.connect(embedding_db_path.value)

    config = load_config_dict(db, f"{_exp}_config")

    _raw_source_path = config.get("source_path", "")
    _source_path = resolve_source_path(embedding_db_path.value, _raw_source_path)
    mo.stop(_source_path is None, mo.callout(
        mo.md(f"Could not resolve source DB path from config `source_path`: `{_raw_source_path}`"),
        kind="danger",
    ))

    source_db = lancedb.connect(_source_path)
    img_emb_tbl = db.open_table(config["tbl_img_emb"])
    patch_emb_tbl = db.open_table(config["tbl_patch_emb"])
    src_img_tbl = source_db.open_table(config["source"])
    return config, img_emb_tbl, patch_emb_tbl, src_img_tbl


@app.cell
def _(config, img_emb_tbl, mo, patch_emb_tbl, pd, src_img_tbl):
    import json

    def scroll(tbl, height="240px"):
        return mo.Html(
            f'<div style="max-height:{height};overflow-y:auto">'
            + mo.as_html(tbl).text
            + "</div>"
        )

    def schema_df(tbl):
        return pd.DataFrame([{"Column": f.name, "Type": str(f.type)} for f in tbl.schema])

    # Source tab: nested tabs — Metadata first, Columns second
    _raw_meta = src_img_tbl.schema.metadata or {}
    _dataset_info = json.loads(_raw_meta.get(b"dataset_info", "{}"))

    def flatten(d, prefix=""):
        rows = []
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                rows.extend(flatten(v, key))
            else:
                rows.append({"Key": key, "Value": str(v)})
        return rows

    _src_tab = mo.ui.tabs({
        "Metadata": scroll(mo.ui.table(pd.DataFrame(flatten(_dataset_info)), selection=None))
        if _dataset_info
        else mo.md("*(no schema metadata)*"),
        "Columns": scroll(mo.ui.table(schema_df(src_img_tbl), selection=None)),
    })

    # Index info helper
    def index_info(tbl):
        indices = list(tbl.list_indices())
        if not indices:
            return mo.md("*(no indexes)*")
        rows = []
        for idx in indices:
            stats = tbl.index_stats(idx.name)
            row = {
                "Name": idx.name,
                "Type": getattr(idx, "index_type", getattr(idx, "type", "?")),
                "Columns": ", ".join(idx.columns),
            }
            if stats is not None:
                row["Indexed rows"] = getattr(stats, "num_indexed_rows", "—")
                row["Unindexed rows"] = getattr(stats, "num_unindexed_rows", "—")
            rows.append(row)
        return mo.ui.table(pd.DataFrame(rows), selection=None)

    # Experiment tab: nested tabs for Config / Image Embeddings / Patch Embeddings
    _img_n = img_emb_tbl.count_rows()
    _patch_n = patch_emb_tbl.count_rows()
    _cfg_df = pd.DataFrame(list(config.items()), columns=["Key", "Value"])
    _exp_tab = mo.ui.tabs({
        "Config": scroll(mo.ui.table(_cfg_df, selection=None)),
        f"Image Embeddings ({_img_n:,})": mo.vstack([
            mo.md("### Schema"),
            scroll(mo.ui.table(schema_df(img_emb_tbl), selection=None)),
            mo.md("### Indexes"),
            index_info(img_emb_tbl),
        ]),
        f"Patch Embeddings ({_patch_n:,})": mo.vstack([
            mo.md("### Schema"),
            scroll(mo.ui.table(schema_df(patch_emb_tbl), selection=None)),
            mo.md("### Indexes"),
            index_info(patch_emb_tbl),
        ]),
    })

    mo.ui.tabs({"Source": _src_tab, "Experiment": _exp_tab})
    return


@app.cell
def _(mo, src_img_tbl):
    _filenames = (
        src_img_tbl.search()
        .select(["filename"])
        .limit(10_000)
        .to_pandas()["filename"]
        .sort_values()
        .tolist()
    )
    selected_filename = mo.ui.dropdown(
        options=_filenames,
        value=_filenames[0] if _filenames else None,
        label="Image",
    )
    selected_filename
    return (selected_filename,)


@app.cell
def _(mo):
    patch_index_slider = mo.ui.slider(
        start=0,
        stop=255,
        value=135,
        label="Patch index",
        show_value=True,
    )
    patch_index_slider
    return (patch_index_slider,)


@app.cell
def _(mo, patch_index_slider, selected_filename, src_img_tbl):
    mo.stop(selected_filename.value is None, mo.callout(
        mo.md("Select an image above."), kind="info"
    ))

    _PATCH_SIZE = 16
    _IMG_SIZE = 256

    _img = fetch_image_by_filename(src_img_tbl, selected_filename.value)
    _patch_index = patch_index_slider.value
    _box = patch_box_from_index(_patch_index, _IMG_SIZE, _IMG_SIZE, _PATCH_SIZE)
    _img_vis = overlay_grid(_img, _PATCH_SIZE)
    img_highlighted = highlight_patch(_img_vis, _box)

    patch_index = _patch_index
    img_highlighted
    return (patch_index,)


@app.cell
def _(mo, patch_emb_tbl, patch_index, selected_filename, src_img_tbl):
    mo.stop(selected_filename.value is None)

    _image_id = (
        src_img_tbl.search()
        .where(f"filename = '{selected_filename.value}'")
        .select(["id"])
        .limit(1)
        .to_pandas()
        .iloc[0]["id"]
    )

    _q = (
        patch_emb_tbl.search()
        .where(f"image_id = '{_image_id}' AND patch_index = {patch_index}")
        .select(["patch_id", "image_id", "patch_index", "embedding"])
        .limit(1)
        .to_pandas()
        .iloc[0]
    )

    top_df = (
        patch_emb_tbl.search(_q["embedding"])
        .metric("cosine")
        .select(["patch_id", "image_id", "patch_index"])
        .limit(100)
        .to_pandas()
    )

    top_df
    return (top_df,)


@app.cell
def _(Image, ImageDraw, io, math, mo, pd, plt, src_img_tbl, top_df):
    mo.stop(top_df is None or top_df.empty)

    _PATCH, _BASE = 16, 256
    _u = top_df.groupby("image_id")["patch_index"].apply(list).head(30)
    _thumbs, _rows = [], []

    for _img_id, _pidxs in _u.items():
        _r = (
            src_img_tbl.search()
            .where(f"id = '{_img_id}'")
            .select(["image_blob", "dt"])
            .limit(1)
            .to_pandas()
            .iloc[0]
        )
        _im = Image.open(io.BytesIO(_r["image_blob"])).convert("RGB")
        _tw, _th = _im.size
        _sx, _sy = _tw / _BASE, _th / _BASE
        _grid_w = _BASE // _PATCH
        _d = ImageDraw.Draw(_im)
        for _pidx in map(int, _pidxs):
            _rr, _cc = _pidx // _grid_w, _pidx % _grid_w
            _x0, _y0 = _cc * _PATCH, _rr * _PATCH
            _box = (int(_x0 * _sx), int(_y0 * _sy), int((_x0 + _PATCH) * _sx), int((_y0 + _PATCH) * _sy))
            _d.rectangle(_box, outline=(0, 0, 0), width=max(2, int(round(min(_sx, _sy)))))
        _thumbs.append(_im)
        _rows.append({"image_id": _img_id, "dt": _r["dt"], "n_patches": len(_pidxs), "patch_indices": _pidxs})

    _dates_df = pd.DataFrame(_rows)
    _n = len(_thumbs)
    _cols = min(6, _n)
    _rows_n = math.ceil(_n / _cols)

    _fig, _axes = plt.subplots(_rows_n, _cols, figsize=(_cols * 2.2, _rows_n * 2.2))
    _axes_flat = _axes.flat if _n > 1 else [_axes]
    for _i, (_im, _ax) in enumerate(zip(_thumbs, _axes_flat)):
        _ax.imshow(_im)
        _ax.set_title(f"{_dates_df.loc[_i, 'n_patches']} | {str(_dates_df.loc[_i, 'dt'])[:19]}", fontsize=8)
        _ax.axis("off")
    for _ax in list(_axes_flat)[_n:]:
        _ax.axis("off")
    plt.tight_layout()
    plt.gca()

    mo.vstack([mo.pyplot(_fig), mo.ui.table(_dates_df, selection=None)])
    return


if __name__ == "__main__":
    app.run()
