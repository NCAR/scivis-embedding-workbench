import marimo

__generated_with = "0.23.13"
app = marimo.App(width="full")


@app.cell
def _(mo):
    mo.md(r"""
    # DINOv3 Attention Maps

    Inspect where a DINOv3 ViT actually looks when it encodes a frame from the
    shared source table.

    Pick a source token — a patch on the image, the `CLS` token, or one of the
    register tokens — and see the attention it sends to every patch, laid out as a
    **layer x head** grid, with a head-summary column on the right.

    Preprocessing and model construction are imported from
    `02-generate-embeddings/helpers/v5_dino_embeddings_lancedb.py`, so the token
    geometry here is the one the stored embeddings were built on: a rectangular
    resize to `IMAGE_H x IMAGE_W` with **no centre crop**, and
    `dynamic_img_size=True`.
    """)
    return


@app.cell
def _():
    import io
    import os
    import sys
    from pathlib import Path

    import altair as alt
    import lancedb
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import pyarrow.compute as pc
    import torch
    from PIL import Image

    # Reuse the embedding pipeline's own preprocessing and model loader rather
    # than reimplementing them, so geometry can't silently drift out of parity.
    _helpers_root = mo.notebook_dir().parent / "02-generate-embeddings"
    if str(_helpers_root) not in sys.path:
        sys.path.insert(0, str(_helpers_root))

    from helpers.v5_dino_embeddings_lancedb import (
        build_model,
        build_rect_transform,
        resolve_model_data_config,
    )

    return (
        Image,
        Path,
        alt,
        build_model,
        build_rect_transform,
        io,
        lancedb,
        mo,
        np,
        os,
        pc,
        pd,
        plt,
        resolve_model_data_config,
        torch,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Pick a frame
    """)
    return


@app.cell
def _(Path, mo, os):
    # Same widget as the Local file tab, in directory mode: the source DB is a
    # directory, so browsing to it beats typing two path fragments. initial_path
    # lands on shared_source so the project folders are the listed choices;
    # override with SCIVIS_SHARED_SOURCE (on Casper, a /glade path).
    shared_source_default = os.environ.get(
        "SCIVIS_SHARED_SOURCE",
        "/Users/ncheruku/Documents/Work/sample_data/data/lancedb/shared_source",
    )
    default_project = os.environ.get(
        "SCIVIS_SOURCE_PROJECT", "era5_hrly_2016_2018_images"
    )

    db_dir_ui = mo.ui.file_browser(
        initial_path=(
            shared_source_default
            if Path(shared_source_default).is_dir()
            else str(Path.home())
        ),
        selection_mode="directory",
        multiple=False,
        label="Source LanceDB directory",
    )
    return db_dir_ui, default_project, shared_source_default


@app.cell
def _(Path, db_dir_ui, default_project, lancedb, shared_source_default):
    # Deliberately no mo.stop in the database chain: a missing or broken DB must
    # degrade to a message inside the Database tab, not kill the cells the Local
    # file tab depends on.
    if db_dir_ui.value:
        source_uri = Path(db_dir_ui.value[0].path)
    else:
        # Nothing picked yet — fall back to the convention used by
        # 02-generate-embeddings so the tab is useful on first load.
        source_uri = Path(shared_source_default).expanduser() / default_project

    source_db = None
    db_tables = []
    db_error = None

    if not source_uri.is_dir():
        db_error = (
            f"**Source DB not found:**\n\n`{source_uri}`\n\nPick the LanceDB "
            "directory above — the one holding `*.lance` tables, matching "
            "`SOURCE_URI` in `02-generate-embeddings`."
        )
    else:
        try:
            source_db = lancedb.connect(str(source_uri))
            # list_tables() returns a ListTablesResponse, not names — iterating
            # it yields ("tables", [...]) pairs. table_names() is deprecated but
            # is the call that returns plain strings.
            if hasattr(source_db, "list_tables"):
                db_tables = list(source_db.list_tables().tables)
            else:
                db_tables = list(source_db.table_names())
            if not db_tables:
                db_error = f"No tables in `{source_uri}`."
        except Exception as _exc:
            db_error = f"Could not open `{source_uri}`:\n\n```\n{_exc}\n```"
    return db_error, db_tables, source_db, source_uri


@app.cell
def _(db_tables, mo):
    table_ui = mo.ui.dropdown(
        options=db_tables,
        value=("images" if "images" in db_tables else db_tables[0]) if db_tables else None,
        label="Table",
    )
    return (table_ui,)


@app.cell
def _(source_db, table_ui):
    # Only the index columns, never the blobs: a full scan of id/filename/dt is a
    # few MB even at 100k rows, whereas the image blobs are ~600 KB each. This is
    # what lets the date picker see the whole table instead of a paged window.
    src_tbl = None
    index = None
    index_note = None

    if source_db is not None and table_ui.value:
        src_tbl = source_db.open_table(table_ui.value)
        _blob_cols = {"image_blob", "thumb_blob"}
        _cols = [f.name for f in src_tbl.schema if f.name not in _blob_cols]
        _df = src_tbl.to_lance().scanner(columns=_cols).to_table().to_pandas()
        if _df.empty:
            index_note = f"Table `{table_ui.value}` is empty."
        elif "dt" not in _df.columns:
            index_note = (
                f"Table `{table_ui.value}` has no `dt` column, so it cannot be "
                "browsed by date. Use the **Local file** tab instead."
            )
        else:
            _df["_date"] = _df["dt"].dt.date
            _df["_time"] = _df["dt"].dt.strftime("%H:%M")
            index = _df
    return index, index_note, src_tbl


@app.cell
def _(index, mo):
    _dates = sorted(index["_date"].unique()) if index is not None else []
    if _dates:
        date_ui = mo.ui.date(
            start=_dates[0], stop=_dates[-1], value=_dates[0], label="Date"
        )
    else:
        date_ui = mo.ui.date(label="Date")
    return (date_ui,)


@app.cell
def _(date_ui, index, mo):
    if index is not None:
        _times = sorted(index[index["_date"] == date_ui.value]["_time"].unique())
    else:
        _times = []
    time_ui = mo.ui.dropdown(
        options=_times, value=_times[0] if _times else None, label="Time (UTC)"
    )
    return (time_ui,)


@app.cell
def _(date_ui, index, mo, time_ui):
    # A datetime can map to several rows — ensemble members, variants, reruns —
    # so resolve date+time to a set of candidates and let the member be chosen
    # explicitly rather than silently taking the first match.
    members = None
    member_ui = mo.ui.dropdown(options=[], value=None, label="File")

    if index is not None and time_ui.value:
        members = index[
            (index["_date"] == date_ui.value) & (index["_time"] == time_ui.value)
        ].reset_index(drop=True)
        if len(members):
            if len(members) > 1:
                _opts = {
                    f"{r.filename}  ({r.id[:8]})": int(i) for i, r in members.iterrows()
                }
                _label = f"Member ({len(members)} at this time)"
            else:
                _opts = {str(members.iloc[0].filename): 0}
                _label = "File"
            member_ui = mo.ui.dropdown(
                options=_opts, value=next(iter(_opts)), label=_label
            )
    return member_ui, members


@app.cell
def _(
    date_ui,
    db_dir_ui,
    db_error,
    index,
    index_note,
    member_ui,
    mo,
    source_uri,
    table_ui,
    time_ui,
):
    _problem = db_error or index_note
    _resolved = mo.md(f"Reading `{source_uri}`")

    if _problem:
        db_panel = mo.vstack([db_dir_ui, mo.callout(mo.md(_problem), kind="warn")])
    else:
        db_panel = mo.vstack(
            [
                db_dir_ui,
                _resolved,
                mo.hstack([table_ui, date_ui, time_ui], justify="start", gap=1),
                member_ui,
                mo.md(f"*{len(index):,} frames in table*"),
            ]
        )
    return (db_panel,)


@app.cell
def _(Path, mo, os):
    # Browses the filesystem of the machine running marimo, so on Casper this
    # reaches glade paths directly rather than uploading from the laptop.
    file_ui = mo.ui.file_browser(
        initial_path=os.environ.get("SCIVIS_IMAGE_DIR", str(Path.home())),
        filetypes=[".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"],
        multiple=False,
        label="Choose an image",
    )
    file_panel = mo.vstack(
        [
            mo.md(
                "Any image readable by the marimo process — useful for one-off "
                "frames that are not in a table yet. It goes through the same "
                "rectangular resize as the database path, so the token geometry "
                "matches."
            ),
            file_ui,
        ]
    )
    return file_panel, file_ui


@app.cell
def _(db_panel, file_panel, mo):
    source_tabs = mo.ui.tabs({"Database": db_panel, "Local file": file_panel})
    source_tabs
    return (source_tabs,)


@app.cell
def _(
    Image,
    Path,
    file_ui,
    io,
    member_ui,
    members,
    mo,
    pc,
    source_tabs,
    src_tbl,
):
    if source_tabs.value == "Local file":
        mo.stop(not file_ui.value, mo.md("*Choose an image file.*"))
        _path = Path(file_ui.value[0].path)
        source_image = Image.open(_path).convert("RGB")
        image_label = _path.name
        image_id = str(_path)
        blob_kb = _path.stat().st_size / 1e3
        frame_meta = []
    else:
        mo.stop(
            src_tbl is None or members is None or not len(members),
            mo.md("*No frame selected — see the **Database** tab.*"),
        )
        _row = members.iloc[int(member_ui.value)]
        image_id = str(_row["id"])
        _df = (
            src_tbl.to_lance()
            .scanner(
                columns=["id", "image_blob"], filter=pc.field("id").isin([image_id])
            )
            .to_table()
            .to_pandas()
        )
        mo.stop(
            _df.empty,
            mo.callout(mo.md(f"No blob for id `{image_id}`."), kind="danger"),
        )
        _blob = _df.iloc[0]["image_blob"]
        if not isinstance(_blob, (bytes, bytearray)):
            # Lance blob columns can come back as buffer-like objects.
            _blob = bytes(_blob.data) if hasattr(_blob, "data") else bytes(_blob)
        source_image = Image.open(io.BytesIO(_blob)).convert("RGB")
        image_label = str(_row.get("filename", image_id))
        blob_kb = len(_blob) / 1e3
        frame_meta = [
            f"{c} = **{_row[c]}**"
            for c in ("hurricane_present", "n_storms", "max_wind_kts", "max_category")
            if c in members.columns
        ]
    return blob_kb, frame_meta, image_id, image_label, source_image


@app.cell
def _(blob_kb, frame_meta, image_id, image_label, mo, source_image):
    # Preview before the forward pass: confirm the frame is the intended one
    # without paying for a model run.
    mo.hstack(
        [
            mo.md(
                f"`{image_label}`\n\n"
                f"{source_image.width}x{source_image.height} px, "
                f"{blob_kb:.0f} KB  \n`{image_id}`\n\n"
                + ("  \n".join(frame_meta) if frame_meta else "")
            ),
            mo.image(source_image, width=820),
        ],
        justify="start",
        gap=2,
        widths=[1, 2],
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. Capture attention
    """)
    return


@app.cell
def _(mo):
    model_ui = mo.ui.dropdown(
        options=[
            "vit_small_patch16_dinov3",
            "vit_base_patch16_dinov3",
            "vit_large_patch16_dinov3",
        ],
        value="vit_base_patch16_dinov3",
        label="Model",
    )
    # Defaults match IMAGE_H / IMAGE_W in 02-generate-embeddings (ERA5 is 7:2).
    image_h_ui = mo.ui.number(value=256, start=16, stop=1024, step=16, label="Image H")
    image_w_ui = mo.ui.number(value=896, start=16, stop=2048, step=16, label="Image W")
    device_ui = mo.ui.dropdown(
        options=["auto", "cuda", "mps", "cpu"], value="auto", label="Device"
    )
    run_btn = mo.ui.run_button(label="Run forward pass")

    mo.vstack(
        [
            mo.hstack([model_ui, device_ui], justify="start", gap=1),
            mo.hstack([image_h_ui, image_w_ui, run_btn], justify="start", gap=1),
        ]
    )
    return device_ui, image_h_ui, image_w_ui, model_ui, run_btn


@app.cell
def _(image_h_ui, image_w_ui, mo, model_ui):
    patch_size = 16
    mo.stop(
        int(image_h_ui.value) % patch_size or int(image_w_ui.value) % patch_size,
        mo.callout(
            mo.md(f"`IMAGE_H` and `IMAGE_W` must both be multiples of {patch_size}."),
            kind="danger",
        ),
    )

    _rows = int(image_h_ui.value) // patch_size
    _cols = int(image_w_ui.value) // patch_size
    _n_tok = _rows * _cols + 5
    _size = model_ui.value.split("_")[1]
    _heads = {"small": 6, "base": 12, "large": 16}[_size]
    _layers = {"small": 12, "base": 12, "large": 24}[_size]
    est_mb = _layers * _heads * _n_tok * _n_tok * 2 / 1e6

    mo.callout(
        mo.md(
            f"Grid **{_rows} x {_cols}** = {_rows * _cols} patch tokens (+5 prefix). "
            f"Attention cache approx **{est_mb:.0f} MB** at float16 — it grows with "
            f"the *square* of token count, so widen `IMAGE_W` with that in mind."
        ),
        kind="info" if est_mb < 1500 else "warn",
    )
    return (patch_size,)


@app.cell
def _(
    build_model,
    build_rect_transform,
    device_ui,
    image_h_ui,
    image_w_ui,
    mo,
    model_ui,
    np,
    patch_size,
    resolve_model_data_config,
    run_btn,
    source_image,
    torch,
):
    mo.stop(not run_btn.value, mo.md("*Press **Run forward pass** to begin.*"))

    if device_ui.value != "auto":
        _dev = device_ui.value
    elif torch.cuda.is_available():
        _dev = "cuda"
    elif torch.backends.mps.is_available():
        _dev = "mps"
    else:
        _dev = "cpu"

    _cfg = resolve_model_data_config(model_ui.value)
    _tfm = build_rect_transform(
        _cfg["mean"],
        _cfg["std"],
        int(image_h_ui.value),
        int(image_w_ui.value),
        _cfg.get("interpolation", "bicubic"),
    )
    _x = _tfm(source_image).unsqueeze(0)
    _model = build_model(model_ui.value).to(_dev).eval()

    # Undo the normalisation to recover exactly the pixels the model saw, so the
    # patch grid drawn below lines up with tokens rather than with the source file.
    _mean = torch.tensor(_cfg["mean"]).view(3, 1, 1)
    _std = torch.tensor(_cfg["std"]).view(3, 1, 1)
    img_disp = (_x[0] * _std + _mean).clamp(0, 1).permute(1, 2, 0).numpy()

    # fused_attn off + hook on attn_drop's input gives the true post-softmax
    # matrix. Recomputing q @ k.T from qkv would drop the rotary position
    # embeddings DINOv3 applies to q and k — negligible for the CLS row, but
    # materially wrong for patch rows and severely wrong for register rows.
    _buf = []
    _handles = []
    for _blk in _model.blocks:
        _blk.attn.fused_attn = False
        _handles.append(
            _blk.attn.attn_drop.register_forward_hook(
                # float16: the cache is quadratic in token count and only ever
                # drives a colour map, so float32 precision buys nothing.
                lambda _m, _inp, _out: _buf.append(
                    _inp[0][0].to(torch.float16).cpu().numpy()
                )
            )
        )
    try:
        with torch.no_grad():
            _model.forward_features(_x.to(_dev))
    finally:
        for _h in _handles:
            _h.remove()

    attn = np.stack(_buf)  # (layer, head, token, token)
    n_prefix = _model.num_prefix_tokens
    num_layers, num_heads, n_tokens, _ = attn.shape
    grid_rows = int(image_h_ui.value) // patch_size
    grid_cols = int(image_w_ui.value) // patch_size
    device = _dev

    if grid_rows * grid_cols != n_tokens - n_prefix:
        raise ValueError(
            f"Grid {grid_rows}x{grid_cols} does not match {n_tokens - n_prefix} "
            f"patch tokens (n_tokens={n_tokens}, n_prefix={n_prefix})"
        )
    return (
        attn,
        device,
        grid_cols,
        grid_rows,
        img_disp,
        n_prefix,
        num_heads,
        num_layers,
    )


@app.cell
def _(
    attn,
    device,
    grid_cols,
    grid_rows,
    mo,
    model_ui,
    n_prefix,
    num_heads,
    num_layers,
):
    mo.callout(
        mo.md(
            f"`{model_ui.value}` on **{device}** — {num_layers} layers x "
            f"{num_heads} heads, {attn.shape[2]} tokens "
            f"({n_prefix} prefix + {grid_rows * grid_cols} patches), "
            f"{grid_rows}x{grid_cols} grid, {attn.nbytes / 1e6:.0f} MB cached."
        ),
        kind="success",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Select a token and read its attention

    Controls and the located frame sit above; the clickable patch grid sits
    directly on top of the layer-by-head maps, so a click and its result stay
    within one eyeline.
    """)
    return


@app.cell
def _(cmap_ui, dark_ui, mo, pct_ui, scale_ui, source_ui, summary_ui, vmax_ui):
    mo.vstack(
        [
            mo.hstack([source_ui, summary_ui], justify="start", gap=2),
            mo.hstack(
                [scale_ui, pct_ui, vmax_ui, cmap_ui, dark_ui],
                justify="start",
                gap=2,
            ),
        ]
    )
    return


@app.cell
def _(
    grid_cols,
    grid_rows,
    image_label,
    img_disp,
    plt,
    sel_col,
    sel_row,
    theme,
):
    # Full-resolution locator for the selected patch. It lives above the picker so
    # it never sits between the patch grid and the maps that grid drives.
    _aspect = img_disp.shape[0] / img_disp.shape[1]
    _fig, _ax = plt.subplots(figsize=(9, max(1.4, 9 * _aspect)))
    _fig.patch.set_facecolor(theme["bg"])
    _ax.set_facecolor(theme["bg"])
    _ax.imshow(img_disp)
    _ax.set_xticks([])
    _ax.set_yticks([])
    _ax.set_title(image_label, fontsize=9, color=theme["fg"])
    for _s in _ax.spines.values():
        _s.set_color(theme["muted"])

    if sel_row is not None:
        _ph = img_disp.shape[0] / grid_rows
        _pw = img_disp.shape[1] / grid_cols
        _ax.add_patch(
            plt.Rectangle(
                (sel_col * _pw, sel_row * _ph),
                _pw,
                _ph,
                fill=False,
                edgecolor=theme["accent"],
                linewidth=2,
            )
        )
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(alt, grid_cols, grid_rows, img_disp, mo, np, pd):
    # The clickable selector is the image at token resolution: one rect per patch,
    # filled with that patch's mean colour. A click is therefore an exact token
    # pick, with no pixel-to-index arithmetic to get wrong.
    _tiles = img_disp.reshape(
        grid_rows,
        img_disp.shape[0] // grid_rows,
        grid_cols,
        img_disp.shape[1] // grid_cols,
        3,
    )
    _rgb = (_tiles.mean(axis=(1, 3)) * 255).astype(np.uint8)

    _grid = pd.DataFrame(
        [
            {"row": r, "col": c, "hex": "#{:02x}{:02x}{:02x}".format(*_rgb[r, c])}
            for r in range(grid_rows)
            for c in range(grid_cols)
        ]
    )

    _w = 980

    # marimo's point selection dims every unselected mark, which makes the frame
    # unreadable and the next patch hard to aim at. Pinning opacity cancels that.
    #
    # The highlight is drawn on the locator above rather than on this chart: an
    # in-chart outline needs a selection param of our own, and adding one
    # alongside marimo's chart_selection stops selection registering at all.
    # Deriving the outline from patch_chart.value instead would be a cycle
    # (chart -> value -> chart), which marimo rejects.
    patch_chart = mo.ui.altair_chart(
        alt.Chart(_grid)
        .mark_rect()
        .encode(
            x=alt.X("col:O", axis=None),
            y=alt.Y("row:O", axis=None),
            fill=alt.Fill("hex:N", scale=None, legend=None),
            opacity=alt.value(1.0),
            tooltip=["row:O", "col:O"],
        )
        .properties(
            width=_w,
            height=max(60, int(_w * grid_rows / grid_cols)),
            title="Click a patch",
        ),
        chart_selection="point",
    )
    patch_chart
    return (patch_chart,)


@app.cell
def _(mo, n_prefix):
    source_ui = mo.ui.radio(
        options=["patch", "CLS"] + [f"R-{i}" for i in range(1, n_prefix)],
        value="patch",
        label="Source token",
        inline=True,
    )
    # Scaling to the maximum is useless here: a few heads dump almost all their
    # mass on a single sink token, so max is ~1.0 and every other tile renders
    # flat. A high percentile keeps the sinks clipped and the structure visible.
    scale_ui = mo.ui.radio(
        options=["percentile", "fixed"],
        value="percentile",
        label="Colour scale",
        inline=True,
    )
    pct_ui = mo.ui.slider(
        start=90.0, stop=100.0, step=0.1, value=99.0,
        label="Percentile", show_value=True,
    )
    vmax_ui = mo.ui.slider(
        start=0.0005, stop=0.05, step=0.0005, value=0.005,
        label="Fixed max", show_value=True,
    )
    cmap_ui = mo.ui.dropdown(
        options=["viridis", "magma", "cividis", "inferno"],
        value="viridis",
        label="Colour map",
    )
    # Plain mean is faithful to total mass but is dragged around by sink heads;
    # the median across heads answers "what does a typical head do here".
    summary_ui = mo.ui.radio(
        options=["mean", "median"],
        value="mean",
        label="Head summary",
        inline=True,
    )
    return cmap_ui, pct_ui, scale_ui, source_ui, summary_ui, vmax_ui


@app.cell
def _(mo):
    # mo.app_meta().theme reports "light" under `marimo run` even when the UI is
    # dark, so it is not a reliable default — start dark and let the switch win.
    dark_ui = mo.ui.switch(value=True, label="Dark plots")
    return (dark_ui,)


@app.cell
def _(grid_cols, grid_rows, n_prefix, patch_chart, source_ui):
    # Prefix tokens are laid out as [CLS, R-1, ..., R-n], so "R-k" is token k.
    if source_ui.value == "patch":
        _sel = patch_chart.value
        if _sel is not None and len(_sel):
            sel_row = int(_sel["row"].iloc[0])
            sel_col = int(_sel["col"].iloc[0])
        else:
            sel_row, sel_col = grid_rows // 2, grid_cols // 2
        token_idx = n_prefix + sel_row * grid_cols + sel_col
        source_label = f"patch (row {sel_row}, col {sel_col})"
    else:
        sel_row = sel_col = None
        token_idx = 0 if source_ui.value == "CLS" else int(source_ui.value.split("-")[1])
        source_label = f"{source_ui.value} token"
    return sel_col, sel_row, source_label, token_idx


@app.cell
def _(dark_ui, plt):
    # marimo renders figures through savefig, whose facecolor/edgecolor rcParams
    # default to white and would override fig.patch. "auto" makes savefig honour
    # the figure's own colours, which is what makes dark mode take effect at all.
    plt.rcParams.update({"savefig.facecolor": "auto", "savefig.edgecolor": "auto"})

    # One palette drives every figure so the mosaic, the preview and both
    # colourbars stay consistent when the theme flips.
    if dark_ui.value:
        theme = {
            "bg": "#15171c",
            "fg": "#d6dae1",
            "muted": "#8b94a3",
            "accent": "#ff5c5c",
        }
    else:
        theme = {
            "bg": "#ffffff",
            "fg": "#1a1a1a",
            "muted": "#555555",
            "accent": "#d62728",
        }
    return (theme,)


@app.cell
def _(
    attn,
    cmap_ui,
    grid_cols,
    grid_rows,
    n_prefix,
    np,
    num_heads,
    num_layers,
    pct_ui,
    plt,
    scale_ui,
    source_label,
    summary_ui,
    theme,
    token_idx,
    vmax_ui,
):
    # marimo re-renders rather than mutating artists, so the tiles are composited
    # into one array and drawn with a single imshow per panel. Calling imshow once
    # per tile is what made the ipywidgets version sluggish on every click.
    _maps = (
        attn[:, :, token_idx, n_prefix:]
        .astype(np.float32)
        .reshape(num_layers, num_heads, grid_rows, grid_cols)
    )

    # Head summary. Each head's row is already a distribution over all tokens, so
    # the mean across heads is one too — deliberately NOT renormalised per tile,
    # which would destroy comparability across layers.
    if summary_ui.value == "median":
        _summary = np.median(_maps, axis=1)
    else:
        _summary = _maps.mean(axis=1)

    _gap = 1

    def _mosaic(stack):
        # stack is (layers, columns, rows, cols); columns is heads for the main
        # panel and 1 for the summary panel.
        _n = stack.shape[1]
        _out = np.full(
            (num_layers * (grid_rows + _gap) - _gap, _n * (grid_cols + _gap) - _gap),
            np.nan,
            dtype=np.float32,
        )
        for _i in range(num_layers):
            for _j in range(_n):
                _r0 = _i * (grid_rows + _gap)
                _c0 = _j * (grid_cols + _gap)
                _out[_r0 : _r0 + grid_rows, _c0 : _c0 + grid_cols] = stack[_i, _j]
        return _out

    _heads_mos = _mosaic(_maps)
    _sum_mos = _mosaic(_summary[:, None, :, :])

    def _vmax_for(arr):
        if scale_ui.value == "percentile":
            return max(float(np.nanpercentile(arr, pct_ui.value)), 1e-9)
        return max(float(vmax_ui.value), 1e-9)

    # The summary is ~1/num_heads the dynamic range of a peaked head, so it gets
    # its own scale and colourbar — on the shared one it would read as empty.
    _v_heads = _vmax_for(_heads_mos)
    _v_sum = _vmax_for(_sum_mos)
    _cmap = plt.get_cmap(cmap_ui.value).with_extremes(bad=theme["bg"])

    _fig_w = 18
    _h_px = _heads_mos.shape[0]
    _w_px = _heads_mos.shape[1] + (grid_cols + _gap) * 1.6
    _fig = plt.figure(figsize=(_fig_w, max(3.5, _fig_w * _h_px / _w_px)))
    _fig.patch.set_facecolor(theme["bg"])
    _gs = _fig.add_gridspec(1, 2, width_ratios=[num_heads, 1.35], wspace=0.06)

    _ax = _fig.add_subplot(_gs[0, 0])
    _axs = _fig.add_subplot(_gs[0, 1])

    _im = _ax.imshow(
        _heads_mos, cmap=_cmap, vmin=0.0, vmax=_v_heads, interpolation="nearest"
    )
    _ims = _axs.imshow(
        _sum_mos, cmap=_cmap, vmin=0.0, vmax=_v_sum, interpolation="nearest"
    )

    _ax.set_xticks([_j * (grid_cols + _gap) + grid_cols / 2 for _j in range(num_heads)])
    _ax.set_xticklabels([f"H{_j + 1}" for _j in range(num_heads)], fontsize=8)
    _ax.set_yticks([_i * (grid_rows + _gap) + grid_rows / 2 for _i in range(num_layers)])
    _ax.set_yticklabels([f"L{_i + 1}" for _i in range(num_layers)], fontsize=8)

    _axs.set_xticks([grid_cols / 2])
    _axs.set_xticklabels([summary_ui.value], fontsize=9)
    _axs.set_yticks([])

    for _a in (_ax, _axs):
        _a.set_facecolor(theme["bg"])
        _a.tick_params(length=0, colors=theme["fg"])
        for _s in _a.spines.values():
            _s.set_visible(False)

    _clip = f" (clipped at p{pct_ui.value:g})" if scale_ui.value == "percentile" else ""
    for _mappable, _a, _lab in (
        (_im, _ax, f"Per-head attention{_clip}"),
        (_ims, _axs, f"Head {summary_ui.value}{_clip}"),
    ):
        _cb = _fig.colorbar(_mappable, ax=_a, fraction=0.02, pad=0.012)
        _cb.set_label(_lab, color=theme["fg"], fontsize=9)
        _cb.ax.yaxis.set_tick_params(color=theme["fg"], labelcolor=theme["fg"], labelsize=8)
        _cb.outline.set_edgecolor(theme["muted"])

    _fig.suptitle(
        f"Attention from {source_label} to all patches",
        fontsize=13,
        color=theme["fg"],
    )
    _fig.tight_layout(rect=(0, 0, 1, 0.97))
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Reading the grid

    Early layers usually look local or near-uniform; structure concentrates in the
    middle and late layers. The register tokens (`R-1`..`R-n`) exist to absorb the
    high-norm global artefacts that would otherwise contaminate patch tokens — a
    register that looks like a diffuse wash over the whole frame is expected.

    **The summary column.** `mean` is faithful to total attention mass but is
    dragged around by sink heads; `median` shows what a typical head does. The
    bottom cell of this column, with `CLS` selected, is exactly the quantity the
    pipeline stores as the `attention_map` field — so it doubles as a check on
    what is already in the experiments DB.

    **Attention is not the embedding.** `patch_embeddings` holds the *last block's
    output tokens* (post-norm, L2-normalised), not attention. The layer-12 row
    from a patch shows which patches were mixed into that patch's stored vector,
    which is the closest interpretive link — but the residual stream means the
    stored vector is mostly its own running representation plus that one update.

    **On the colour scale.** Several heads behave as attention sinks, so the raw
    maximum is close to `1.0` while the median is around `1e-4`. Scaling to the
    max renders everything else flat; the default clips at a percentile instead.
    """)
    return


if __name__ == "__main__":
    app.run()
