import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import lancedb

    return lancedb, mo


@app.function
def get_metadata_value(table, key_name, value_column="value"):
    """
    Retrieves a single value from a LanceDB table given a key.

    Args:
        table: The opened LanceDB table object.
        key_name: The string key to look for (e.g., 'tbl_img_emb').
        value_column: The name of the column containing the data.
    """
    result = table.search().where(f"key='{key_name}'").select([value_column]).to_pandas()
    if not result.empty:
        return result[value_column].iloc[0]
    return None


@app.function
def list_experiments(db_path: str) -> list:
    """Scan a LanceDB directory for experiment names by finding *_config.lance dirs."""
    from pathlib import Path
    p = Path(db_path)
    if not p.exists() or not p.is_dir():
        return []
    experiments = []
    for d in sorted(p.iterdir()):
        if d.is_dir() and d.name.endswith("_config.lance"):
            experiments.append(d.name[: -len("_config.lance")])
    return experiments


@app.function
def load_config_dict(db, config_table_name: str) -> dict:
    """Load a config table into a Python dict keyed by the 'key' column."""
    tbl = db.open_table(config_table_name)
    df = tbl.to_pandas()
    return dict(zip(df["key"], df["value"]))


@app.function
def get_table_name(config: dict, *keys) -> str:
    """Try multiple config keys in order, return value of the first match."""
    for key in keys:
        if key in config and config[key]:
            return config[key]
    return None


@app.cell
def _(mo):
    embedding_db_path = mo.ui.text(
        value="/Users/ncheruku/Documents/Work/sample_data/data/lancedb/experiments/era5",
        label="Experiments DB path",
        full_width=True,
    )
    source_db_path = mo.ui.text(
        value="/Users/ncheruku/Documents/Work/sample_data/data/lancedb/shared_source",
        label="Source DB path",
        full_width=True,
    )
    return embedding_db_path, source_db_path


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
def _(embedding_db_path, experiment_selector, lancedb, mo, source_db_path):
    mo.stop(experiment_selector.value is None, mo.callout(
        mo.md("Select an experiment above."), kind="info"
    ))

    _exp = experiment_selector.value
    db = lancedb.connect(embedding_db_path.value)
    source_db = lancedb.connect(source_db_path.value)

    config = load_config_dict(db, f"{_exp}_config")
    db.open_table(f"{_exp}_config")

    # Support both naming conventions (setup_experiment keys and legacy keys)
    _img_emb_name = get_table_name(config, "tbl_img_emb", "img_emb_table_current")
    _patch_emb_name = get_table_name(config, "tbl_patch_emb", "patch_emb_table_current")
    _src_tbl_name = config.get("source", "era5_sample_images")

    db.open_table(_img_emb_name)
    patch_emb_tbl = db.open_table(_patch_emb_name)
    src_img_tbl = source_db.open_table(_src_tbl_name)
    return config, patch_emb_tbl, src_img_tbl


@app.cell
def _(config, mo):
    import pandas as pd

    _df = pd.DataFrame(list(config.items()), columns=["Key", "Value"])
    mo.vstack([
        mo.md("### Experiment Config"),
        mo.ui.table(_df, selection=None),
    ])
    return


@app.cell
def _(src_img_tbl):
    src_img_tbl.schema
    return


@app.cell
def _():
    # patch_emb_tbl.schema
    return


@app.cell
def _():
    # img_emb_tbl.schema
    return


@app.cell
def _():
    # config_tbl.schema
    return


@app.cell
def _(mo):
    FILENAME = mo.ui.text(value="20161009_rgb.jpeg", label="Image filename")
    FILENAME
    return (FILENAME,)


@app.cell
def _(FILENAME, src_img_tbl):
    import io
    from PIL import Image, ImageDraw
    PATCH_SIZE = 16
    IMG_SIZE = 256
    DRAW_GRID = True

    def fetch_image_by_filename(src_img_tbl, filename: str):
        """Load image from src_img_tbl using filename instead of id."""
        row = src_img_tbl.search().where(f"filename = '{filename}'").limit(1).to_pandas().iloc[0]
        return Image.open(io.BytesIO(row['image_blob'])).convert('RGBA')

    def overlay_grid(img: Image.Image, patch_size: int=PATCH_SIZE, color=(255, 255, 255, 50), width: int=1) -> Image.Image:
        w, h = img.size
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        for x in range(0, w + 1, patch_size):
            d.line([(x, 0), (x, h)], fill=color, width=width)
        for y in range(0, h + 1, patch_size):
            d.line([(0, y), (w, y)], fill=color, width=width)
        return Image.alpha_composite(img, overlay)

    def patch_box_from_index(patch_index: int, img_w: int=IMG_SIZE, img_h: int=IMG_SIZE, patch_size: int=PATCH_SIZE):
        grid_w = img_w // patch_size
        grid_h = img_h // patch_size
        n_patches = grid_w * grid_h
        if patch_index < 0 or patch_index >= n_patches:
            raise ValueError(f'patch_index {patch_index} out of range [0, {n_patches - 1}]')
        row = patch_index // grid_w
        col = patch_index % grid_w
        x0 = col * patch_size
        y0 = row * patch_size
        x1 = x0 + patch_size
        y1 = y0 + patch_size
        return (x0, y0, x1, y1)

    def highlight_patch(img: Image.Image, box, fill=(255, 0, 0, 0), outline=(255, 0, 0, 220), outline_width: int=2) -> Image.Image:
        overlay = Image.new('RGBA', img.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)
        d.rectangle(_box, fill=fill, outline=outline, width=outline_width)
        return Image.alpha_composite(img, overlay)

    img = fetch_image_by_filename(src_img_tbl, FILENAME.value)
    print('image size:', img.size)
    if img.size != (IMG_SIZE, IMG_SIZE):
        print(f'WARNING: expected {(IMG_SIZE, IMG_SIZE)} but got {img.size}')
    img_vis = img
    if DRAW_GRID:
        img_vis = overlay_grid(img_vis, PATCH_SIZE)
    patch_index = 135
    print('patch_index:', patch_index)
    _box = patch_box_from_index(patch_index, IMG_SIZE, IMG_SIZE, PATCH_SIZE)
    img_hl = highlight_patch(img_vis, _box)
    img_hl
    return Image, ImageDraw, io, patch_index


@app.cell
def _(FILENAME, patch_emb_tbl, patch_index, src_img_tbl):
    image_id = src_img_tbl.search().where(f"filename = '{FILENAME.value}'").select(["id"]).limit(1).to_pandas().iloc[0]["id"]

    q = (
        patch_emb_tbl.search()
        .where(f"image_id = '{image_id}' AND patch_index = {patch_index}")
        .select(["patch_id", "image_id", "patch_index", "embedding"])
        .limit(1)
        .to_pandas()
        .iloc[0]
    )

    query_emb = q["embedding"]

    top_df = patch_emb_tbl.search(query_emb).metric("cosine").select(["patch_id", "image_id", "patch_index"]).limit(100).to_pandas()

    top_df
    return (top_df,)


@app.cell
def _(Image, ImageDraw, io, src_img_tbl, top_df):
    import math
    import matplotlib.pyplot as plt
    import pandas as pd
    PATCH, BASE = (16, 256)
    u = top_df.groupby('image_id')['patch_index'].apply(list).head(30)
    thumbs, rows = ([], [])
    for img_id, pidxs in u.items():
        r = src_img_tbl.search().where(f"id = '{img_id}'").select(['image_blob', 'dt']).limit(1).to_pandas().iloc[0]
        im = Image.open(io.BytesIO(r['image_blob'])).convert('RGB')
        tw, th = im.size
        sx, sy = (tw / BASE, th / BASE)
        grid_w = BASE // PATCH
        d = ImageDraw.Draw(im)
        for pidx in map(int, pidxs):
            rr, cc = (pidx // grid_w, pidx % grid_w)
            x0, y0 = (cc * PATCH, rr * PATCH)
            _box = (int(x0 * sx), int(y0 * sy), int((x0 + PATCH) * sx), int((y0 + PATCH) * sy))
            d.rectangle(_box, outline=(0, 0, 0), width=max(2, int(round(min(sx, sy)))))
        thumbs.append(im)
        rows.append({'image_id': img_id, 'dt': r['dt'], 'n_patches': len(pidxs), 'patch_indices': pidxs})
    dates_df = pd.DataFrame(rows)
    n = len(thumbs)
    cols = min(6, n)
    rows_n = math.ceil(n / cols)
    plt.figure(figsize=(cols * 2.2, rows_n * 2.2))
    for i, im in enumerate(thumbs, 1):
        ax = plt.subplot(rows_n, cols, i)
        ax.imshow(im)
        ax.set_title(f"{dates_df.loc[i - 1, 'n_patches']} | {str(dates_df.loc[i - 1, 'dt'])[:19]}", fontsize=8)
        ax.axis('off')
    plt.tight_layout()
    plt.show()
    dates_df
    return


if __name__ == "__main__":
    app.run()
