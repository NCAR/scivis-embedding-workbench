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
    import bisect
    import io
    import math
    import os
    import re
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
        bisect,
        build_model,
        build_rect_transform,
        io,
        lancedb,
        math,
        mo,
        np,
        os,
        pc,
        pd,
        plt,
        re,
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
    # One ordered list of timesteps drives both fields. Stepping is by dt, not by
    # calendar day, so the arrows roll across midnight and never land on a
    # timestamp the table does not have.
    if index is not None:
        stamps = sorted(
            index[["_date", "_time"]].drop_duplicates().itertuples(index=False, name=None)
        )
    else:
        stamps = []
    get_ts_i, set_ts_i = mo.state(0)
    return get_ts_i, set_ts_i, stamps


@app.cell
def _(mo, set_ts_i, stamps):
    # Deliberately does NOT read get_ts_i. marimo will not re-run the cell that
    # defines the element whose callback set the state, so anything computed
    # from the getter here would go stale — that is what left the ← button
    # permanently disabled. A functional updater removes the need to read it,
    # so this cell has no dependency on the state and the buttons stay valid.
    _n = len(stamps)

    def _step(delta):
        # on_change, not on_click: on_click computes the button's own value.
        # Pairing it with an incrementing value guarantees a change each click,
        # which is what makes on_change fire.
        def _fn(_v):
            set_ts_i(lambda i: max(0, min(i + delta, _n - 1)))

        return _fn

    prev_step = mo.ui.button(
        label="←",
        value=0,
        on_click=lambda _c: _c + 1,
        on_change=_step(-1),
        tooltip="Previous timestep",
    )
    next_step = mo.ui.button(
        label="→",
        value=0,
        on_click=lambda _c: _c + 1,
        on_change=_step(+1),
        tooltip="Next timestep",
    )
    return next_step, prev_step


@app.cell
def _(bisect, get_ts_i, mo, set_ts_i, stamps):
    # Its own cell so it re-runs on every state change, including steps that
    # roll past midnight into the next day.
    _i = max(0, min(get_ts_i(), len(stamps) - 1)) if stamps else 0

    def _on_date(d):
        # Jump to the first timestep on the chosen day. (d,) sorts before any
        # (d, time), so bisect_left lands on that day's first entry.
        if d is None or not stamps:
            return
        _j = max(0, min(bisect.bisect_left(stamps, (d,)), len(stamps) - 1))
        if _j != get_ts_i():
            set_ts_i(_j)

    if stamps:
        date_ui = mo.ui.date(
            start=stamps[0][0],
            stop=stamps[-1][0],
            value=stamps[_i][0],
            label="Date",
            on_change=_on_date,
        )
    else:
        date_ui = mo.ui.date(label="Date")
    return (date_ui,)


@app.cell
def _(get_ts_i, mo, stamps):
    # Position readout, in a state-reading cell so it stays current. Replaces
    # disabling the arrows at the ends, which cannot be done reliably from the
    # cell that defines them.
    _i = max(0, min(get_ts_i(), len(stamps) - 1)) if stamps else 0
    step_pos = mo.md(f"*{_i + 1:,} / {len(stamps):,}*" if stamps else "")
    return (step_pos,)


@app.cell
def _(bisect, get_ts_i, mo, set_ts_i, stamps):
    _i = max(0, min(get_ts_i(), len(stamps) - 1)) if stamps else 0
    _cur_date, _cur_time = stamps[_i] if stamps else (None, None)
    _times = [t for d, t in stamps if d == _cur_date]

    def _on_time(t):
        if t is None or not stamps:
            return
        _j = bisect.bisect_left(stamps, (_cur_date, t))
        _j = max(0, min(_j, len(stamps) - 1))
        if _j != get_ts_i():
            set_ts_i(_j)

    time_ui = mo.ui.dropdown(
        options=_times,
        value=_cur_time,
        label="Time (UTC)",
        on_change=_on_time,
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
    next_step,
    prev_step,
    source_uri,
    step_pos,
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
                mo.hstack(
                    [table_ui, prev_step, date_ui, time_ui, next_step, step_pos],
                    justify="start",
                    gap=1,
                ),
                member_ui,
                mo.md(f"*{len(index):,} frames in table*"),
            ]
        )
    return (db_panel,)


@app.cell
def _(Path, mo, os):
    _exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"]

    # Two inputs because they read different filesystems. Upload goes through the
    # OS file dialog on whichever machine the browser is on, so it has typing and
    # search; it cannot see server-side paths. The browser reads the filesystem
    # of the marimo process, which is what reaches /glade on Casper, but means
    # scrolling. Upload takes precedence when both are set.
    upload_ui = mo.ui.file(
        filetypes=_exts, multiple=False, kind="area", label="Upload an image"
    )
    file_ui = mo.ui.file_browser(
        initial_path=os.environ.get("SCIVIS_IMAGE_DIR", str(Path.home())),
        filetypes=_exts,
        multiple=False,
        label="…or browse the machine running marimo",
    )

    file_panel = mo.vstack(
        [
            mo.md(
                "A one-off frame that is not in a table yet. Either way it goes "
                "through the same rectangular resize as the database path, so the "
                "token geometry matches."
            ),
            upload_ui,
            mo.md(
                "*Upload reads from **your** machine (max 100 MB); the browser "
                "below reads from the machine running marimo — use it for "
                "`/glade` paths on Casper.*"
            ),
            file_ui,
        ]
    )
    return file_panel, file_ui, upload_ui


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
    upload_ui,
):
    if source_tabs.value == "Local file":
        mo.stop(
            not upload_ui.value and not file_ui.value,
            mo.md("*Upload an image, or browse to one.*"),
        )
        if upload_ui.value:
            # Uploaded bytes, from the machine the browser is running on.
            _blob = upload_ui.contents()
            source_image = Image.open(io.BytesIO(_blob)).convert("RGB")
            image_label = upload_ui.name()
            image_id = f"upload:{image_label}"
            blob_kb = len(_blob) / 1e3
        else:
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
    size_mode_ui = mo.ui.radio(
        options=["from image", "pipeline", "manual"],
        value="from image",
        label="Input size",
        inline=True,
    )
    device_ui = mo.ui.dropdown(
        options=["auto", "cuda", "mps", "cpu"], value="auto", label="Device"
    )
    # Deriving size from the image makes it easy to ask for a cache that will not
    # fit: a 1200x1200 frame is 5.6k tokens, which is ~9 GB at float16. This is
    # the ceiling the run refuses to cross, rather than discovering it by OOM.
    max_cache_ui = mo.ui.number(
        value=4000, start=100, stop=64000, step=500, label="Max cache (MB)"
    )
    run_btn = mo.ui.run_button(label="Run forward pass")
    return device_ui, max_cache_ui, model_ui, run_btn, size_mode_ui


@app.cell
def _(mo, patch_size, source_image):
    # Only consulted in "manual". Seeded from the selected frame's own size,
    # snapped down to the patch grid, so the boxes start at something valid for
    # this image rather than at ERA5's 256x896. Defined in their own cell so
    # that re-seeding on a new frame does not also reset the model and device
    # dropdowns.
    def _snap(v):
        return max(patch_size, (int(v) // patch_size) * patch_size)

    _h, _w = _snap(source_image.height), _snap(source_image.width)

    # The ceiling has to clear the seeded value, otherwise mo.ui.number raises
    # on construction for any frame larger than the cap and takes every
    # downstream cell with it. Derived from the image for the same reason the
    # value is.
    image_h_ui = mo.ui.number(
        value=_h, start=patch_size, stop=max(2048, _h), step=patch_size, label="H"
    )
    image_w_ui = mo.ui.number(
        value=_w, start=patch_size, stop=max(4096, _w), step=patch_size, label="W"
    )
    return image_h_ui, image_w_ui


@app.cell
def _(
    device_ui,
    image_h_ui,
    image_w_ui,
    max_cache_ui,
    mo,
    model_ui,
    run_btn,
    size_mode_ui,
):
    # Laid out in its own cell so the H/W boxes can be shown only when they are
    # actually read. Reading size_mode_ui.value in the cell that defines it would
    # recreate the element on every change and reset the selection.
    _size_row = [size_mode_ui]
    if size_mode_ui.value == "manual":
        _size_row += [image_h_ui, image_w_ui]

    mo.vstack(
        [
            mo.hstack(
                [model_ui, device_ui, max_cache_ui, run_btn], justify="start", gap=1
            ),
            mo.hstack(_size_row, justify="start", gap=1),
        ]
    )
    return


@app.cell
def _(model_ui, re):
    # Read the stride off the model name rather than assuming 16, so a patch14
    # entry in the dropdown would not silently mis-slice the token grid. The
    # value is re-checked against the real model once it is built.
    _m = re.search(r"patch(\d+)", model_ui.value)
    patch_size = int(_m.group(1)) if _m else 16
    return (patch_size,)


@app.cell
def _(image_h_ui, image_w_ui, patch_size, size_mode_ui, source_image):
    # Deriving from the image avoids the silent distortion of forcing an
    # arbitrary picture into the pipeline's 7:2 frame; "pipeline" keeps the
    # geometry the stored embeddings were built on, which is what makes maps
    # comparable across experiments.
    def _snap(v):
        return max(patch_size, (int(v) // patch_size) * patch_size)

    if size_mode_ui.value == "from image":
        image_h = _snap(source_image.height)
        image_w = _snap(source_image.width)
    elif size_mode_ui.value == "pipeline":
        image_h, image_w = 256, 896
    else:
        image_h = _snap(image_h_ui.value)
        image_w = _snap(image_w_ui.value)
    return image_h, image_w


@app.cell
def _(image_h, image_w, mo, model_ui, patch_size, size_mode_ui, source_image):
    _rows = image_h // patch_size
    _cols = image_w // patch_size
    _n_tok = _rows * _cols + 5
    _size = model_ui.value.split("_")[1]
    _heads = {"small": 6, "base": 12, "large": 16}[_size]
    _layers = {"small": 12, "base": 12, "large": 24}[_size]
    _dim = {"small": 384, "base": 768, "large": 1024}[_size]
    # Attention is quadratic in tokens; descriptors (4 facets) are linear, so
    # attention is always the term that decides whether this fits.
    _attn_mb = _layers * _heads * _n_tok * _n_tok * 2 / 1e6
    _desc_mb = _layers * 4 * _n_tok * _dim * 2 / 1e6
    est_mb = _attn_mb + _desc_mb

    _native = f"{source_image.width}x{source_image.height}"
    _used = f"{image_w}x{image_h}"
    _resized = "" if _used == _native else f" (source is {_native})"
    _warn = (
        ""
        if size_mode_ui.value == "from image"
        else "  \nNot the image's own size, so it is being resized without "
        "preserving aspect ratio."
    )

    mo.callout(
        mo.md(
            f"Feeding **{_used}** px{_resized} at patch {patch_size} → grid "
            f"**{_rows} x {_cols}** = {_rows * _cols} patch tokens (+5 prefix). "
            f"Cache approx **{est_mb:,.0f} MB** at float16 "
            f"({_attn_mb:,.0f} attention + {_desc_mb:,.0f} descriptors) — "
            f"attention grows with the *square* of token count." + _warn
        ),
        kind="info" if est_mb < 1500 else "warn",
    )
    return


@app.cell
def _(
    build_model,
    build_rect_transform,
    device_ui,
    image_h,
    image_id,
    image_label,
    image_w,
    max_cache_ui,
    mo,
    model_ui,
    np,
    patch_size,
    resolve_model_data_config,
    run_btn,
    source_image,
    torch,
):
    # run_button resets to False once the cells referencing it have run, so any
    # later change of frame lands here and the maps below go blank until the pass
    # is re-run. Name the pending frame so that is obvious rather than looking
    # like the views simply failed to update.
    mo.stop(
        not run_btn.value,
        mo.callout(
            mo.md(
                f"Press **Run forward pass** to capture `{image_label}`.\n\n"
                "Everything below is computed only for the frame the pass ran on."
            ),
            kind="neutral",
        ),
    )

    _n_tok = (image_h // patch_size) * (image_w // patch_size) + 5
    _size = model_ui.value.split("_")[1]
    _layers = {"small": 12, "base": 12, "large": 24}[_size]
    _dim = {"small": 384, "base": 768, "large": 1024}[_size]
    _est_mb = (
        _layers * {"small": 6, "base": 12, "large": 16}[_size] * _n_tok * _n_tok * 2
        + _layers * 4 * _n_tok * _dim * 2
    ) / 1e6
    mo.stop(
        _est_mb > float(max_cache_ui.value),
        mo.callout(
            mo.md(
                f"**{_est_mb:,.0f} MB** would be cached for a "
                f"{image_w}x{image_h} input ({_n_tok:,} tokens), over the "
                f"{float(max_cache_ui.value):,.0f} MB ceiling.\n\nAttention is "
                "quadratic in token count. Either switch **Input size** to "
                "`pipeline`/`manual` and pick something smaller, or raise "
                "**Max cache** if you know it fits."
            ),
            kind="danger",
        ),
    )

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
        image_h,
        image_w,
        _cfg.get("interpolation", "bicubic"),
    )
    _x = _tfm(source_image).unsqueeze(0)
    _model = build_model(model_ui.value).to(_dev).eval()

    # patch_size was read off the model name; confirm it against the real model
    # before it is used to slice the token grid.
    _ps = getattr(getattr(_model, "patch_embed", None), "patch_size", None)
    if isinstance(_ps, (tuple, list)):
        _ps = _ps[0]
    if _ps is not None and int(_ps) != patch_size:
        raise ValueError(
            f"Model reports patch size {int(_ps)}, but {patch_size} was inferred "
            f"from the name `{model_ui.value}`."
        )

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

    # Descriptors for the PCA section, captured in this same pass. Heads are
    # concatenated back into one 768-d vector per token, matching how the
    # dense-descriptor paper builds them — so there is no head axis here.
    #
    # q/k are taken BEFORE RoPE: the paper's model had no rotary embedding, so
    # its "keys" are the unrotated content projection, and post-RoPE keys would
    # be dominated by position. This is the same qkv reconstruction that is
    # WRONG for attention (it drops RoPE) but correct for descriptors — do not
    # conflate the two.
    _desc = {"token": [], "q": [], "k": [], "v": []}

    _tok_raw = []

    def _keep(store):
        # Block outputs are kept in fp32 and normalised after the pass: the final
        # LayerNorm has not been applied at this point, and it is not a scalar
        # rescale, so skipping it would NOT give the pipeline's patch tokens.
        del store

        def _hook(_m, _inp, _out):
            _tok_raw.append(_out[0].detach().float().cpu())

        return _hook

    def _keep_qkv(module):
        @torch.no_grad()
        def _hook(_m, _inp, _out):
            _t = _inp[0]
            _b, _n, _c = _t.shape
            _h = module.num_heads
            _qkv = (
                module.qkv(_t)
                .reshape(_b, _n, 3, _h, _c // _h)
                .permute(2, 0, 3, 1, 4)
            )
            _q, _k, _v = module.q_norm(_qkv[0]), module.k_norm(_qkv[1]), _qkv[2]
            for _name, _tensor in (("q", _q), ("k", _k), ("v", _v)):
                # (B, heads, N, head_dim) -> (N, heads*head_dim)
                _flat = _tensor[0].permute(1, 0, 2).reshape(_n, _c)
                _desc[_name].append(_flat.to(torch.float16).cpu().numpy())

        return _hook

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
        _handles.append(_blk.register_forward_hook(_keep(_desc["token"])))
        _handles.append(_blk.attn.register_forward_hook(_keep_qkv(_blk.attn)))

    try:
        with torch.no_grad():
            _model.forward_features(_x.to(_dev))
    finally:
        for _h in _handles:
            _h.remove()

    # Apply the model's final norm to every block output. Without it the
    # last-layer token facet is pre-norm and does NOT match what the pipeline
    # stores: measured cosine against forward_features output was only 0.79
    # after L2 normalisation. Applying it to every layer also keeps layers
    # mutually comparable, which is what timm's get_intermediate_layers(norm=True)
    # does.
    with torch.no_grad():
        _desc["token"] = [
            _model.norm(_t.to(_dev)).to(torch.float16).cpu().numpy()
            for _t in _tok_raw
        ]

    # {facet: (layer, token, dim)}
    descriptors = {_kk: np.stack(_vv) for _kk, _vv in _desc.items()}
    attn = np.stack(_buf)  # (layer, head, token, token)
    n_prefix = _model.num_prefix_tokens
    num_layers, num_heads, n_tokens, _ = attn.shape
    grid_rows = image_h // patch_size
    grid_cols = image_w // patch_size
    device = _dev

    if grid_rows * grid_cols != n_tokens - n_prefix:
        raise ValueError(
            f"Grid {grid_rows}x{grid_cols} does not match {n_tokens - n_prefix} "
            f"patch tokens (n_tokens={n_tokens}, n_prefix={n_prefix})"
        )
    captured_label = image_label
    captured_id = image_id
    return (
        attn,
        captured_id,
        captured_label,
        descriptors,
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
    captured_id,
    captured_label,
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
            f"Captured from **{captured_label}** — `{captured_id}`  \n"
            f"`{model_ui.value}` on **{device}** — {num_layers} layers x "
            f"{num_heads} heads, {attn.shape[2]} tokens "
            f"({n_prefix} prefix + {grid_rows * grid_cols} patches), "
            f"{grid_rows}x{grid_cols} grid, {attn.nbytes / 1e6:.0f} MB cached."
        ),
        kind="success",
    )
    return


@app.cell
def _():
    # Shared by the picker chart and the locator figure so both tabs render at
    # the same width and switching between them cannot shift the layout.
    PATCH_CHART_W = 980
    return (PATCH_CHART_W,)


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
    PATCH_CHART_W,
    grid_cols,
    grid_rows,
    img_disp,
    io,
    mo,
    plt,
    sel_col,
    sel_row,
    theme,
):
    # Full-resolution locator for the selected patch. Returned rather than shown
    # here so it can be tabbed against the picker just above the gallery.
    #
    # Built to the picker's exact geometry — same width, same image aspect, same
    # title strip — so switching tabs does not resize the block and shove the
    # gallery up and down. PATCH_CHART_W is shared with the chart itself.
    # The chart renders at its declared width plus ~10px of chrome; a matplotlib
    # figure comes back as an <img> that stretches to the container instead. So
    # build the PNG at the chart's *rendered* size and pin it with mo.image,
    # rather than letting the two find different widths.
    # No title inside the figure: matplotlib would draw it in DejaVu Sans while
    # the chart's title comes from Vega in the browser font. Both captions are
    # rendered as markdown above the panels instead, so the fonts match and each
    # panel is pure content of identical height.
    _render_w = PATCH_CHART_W + 10
    _total_h = max(60, int(PATCH_CHART_W * grid_rows / grid_cols))

    _fig = plt.figure(figsize=(_render_w / 100, _total_h / 100), dpi=100)
    _fig.patch.set_facecolor(theme["bg"])
    # Axes placed by hand rather than via tight_layout: the point is to hit a
    # known pixel box, which tight_layout would renegotiate.
    _ax = _fig.add_axes([0.0, 0.0, 1.0, 1.0])
    _ax.set_facecolor(theme["bg"])
    # aspect="auto" fills the axes exactly. The image aspect already equals the
    # grid aspect, so nothing is distorted and no letterboxing remains.
    _ax.imshow(img_disp, aspect="auto")
    _ax.set_xticks([])
    _ax.set_yticks([])
    for _s in _ax.spines.values():
        _s.set_visible(False)
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
    _buf = io.BytesIO()
    _fig.savefig(_buf, format="png", dpi=100, facecolor=theme["bg"])
    plt.close(_fig)
    locator_fig = mo.image(_buf.getvalue(), width=_render_w)
    return (locator_fig,)


@app.cell
def _(PATCH_CHART_W, alt, grid_cols, grid_rows, img_disp, mo, np, pd):
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

    _h = max(60, int(PATCH_CHART_W * grid_rows / grid_cols))

    # marimo dims unselected marks by rewriting the opacity channel on the
    # frontend: encoding.opacity becomes {condition: selected -> o, value: o/5}.
    # The channel is hardcoded, and the rewrite spreads over whatever we set, so
    # an `opacity` in this encoding cannot survive it.
    #
    # Layered specs are exempt — marimo's walker returns layer charts untouched.
    # So the colours live in a base layer that is never rewritten, and a second,
    # invisible layer carries the selection and draws the red outline. The param
    # is named `select_point`, the name marimo reads its value from.
    _picked = alt.selection_point(
        fields=["row", "col"], empty=False, name="select_point"
    )
    _xy = {"x": alt.X("col:O", axis=None), "y": alt.Y("row:O", axis=None)}

    _colours = (
        alt.Chart(_grid)
        .mark_rect()
        .encode(**_xy, fill=alt.Fill("hex:N", scale=None, legend=None))
    )
    _hit = (
        alt.Chart(_grid)
        .mark_rect(fillOpacity=0)
        .encode(
            **_xy,
            stroke=alt.condition(_picked, alt.value("#ff2d55"), alt.value(None)),
            strokeWidth=alt.condition(_picked, alt.value(3), alt.value(0)),
            tooltip=["row:O", "col:O"],
        )
        .add_params(_picked)
    )

    patch_chart = mo.ui.altair_chart(
        alt.layer(_colours, _hit).properties(
            width=PATCH_CHART_W, height=_h
        )
        # Vega-Lite draws a view border by default (config.view.stroke = #ddd).
        # The locator panel has its spines hidden, so drop it here to match.
        .configure_view(stroke=None),
        chart_selection="point",
    )
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
def _(captured_label, locator_fig, mo, patch_chart):
    # One at a time rather than stacked: both are wide, and tabbing keeps
    # whichever is showing within an eyeline of the gallery below. Element
    # values survive while a tab is hidden, so the patch selection still drives
    # the maps even when the preview tab is open.
    # Captions live here as markdown rather than inside each panel, so both use
    # the same font and both tabs are one text line plus identically sized
    # content.
    mo.ui.tabs(
        {
            "Image preview": mo.vstack(
                [mo.md(f"`{captured_label}`"), locator_fig], gap=0.4
            ),
            "Select a patch": mo.vstack(
                [mo.md("Click a patch"), patch_chart], gap=0.4
            ),
        },
        value="Select a patch",
    )
    return


@app.cell
def _(
    attn,
    captured_label,
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
        f"Attention from {source_label} to all patches — {captured_label}",
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


@app.cell
def _(mo):
    mo.md(r"""
    ## 5. PCA of patch descriptors

    Attention answers *where a token looks*. This answers *how the resulting
    features organise spatially*, following
    [Deep ViT Features as Dense Visual Descriptors](https://arxiv.org/abs/2112.05814)
    (Amir, Gandelsman, Bagon, Dekel), whose central claim is that the choice of
    **facet** matters — so all four are shown together rather than one at a time.

    Per row: PCA over that facet's patch descriptors, PC1 thresholded into a
    foreground, then a **second PCA re-fit on the survivors** mapped to RGB.

    `CLS` and the registers are excluded — no grid position, and they would
    dominate PC1 as high-norm outliers. Fits are **per frame**, so colours mean
    nothing across dates. Heads are concatenated into one vector per token by
    default, as the paper's extractor does; **Head** isolates a single head of
    q/k/v instead — a contiguous column slice, so it costs nothing. Block
    outputs have no head decomposition, so the tokens row always shows all
    heads and is labelled as such.

    This section is independent of the source-token selection above: attention
    takes one token as a query, PCA is a property of the whole set.
    """)
    return


@app.cell
def _(mo, num_heads, num_layers):
    # Layers are discrete blocks, not a continuum — a dropdown says that, a
    # slider implies interpolation between them.
    layer_ui = mo.ui.dropdown(
        options={str(_i): _i for _i in range(1, num_layers + 1)},
        value=str(num_layers),
        label="Layer",
    )
    # q/k/v are (heads, N, head_dim) internally and are cached head-major, so a
    # single head is a contiguous column slice — no recompute. The paper always
    # concatenates, so "all" is the faithful default and a single head is
    # exploratory. Tokens are a block output with no head decomposition.
    head_ui = mo.ui.dropdown(
        options={
            "all (concatenated)": -1,
            **{str(_i + 1): _i for _i in range(num_heads)},
        },
        value="all (concatenated)",
        label="Head",
    )
    n_comp_ui = mo.ui.number(
        value=50, start=4, stop=200, step=1, label="Components (scree)"
    )
    invert_ui = mo.ui.switch(value=False, label="Invert foreground")
    rgb_pcs_ui = mo.ui.radio(
        options=["2,3,4", "1,2,3"], value="2,3,4", label="RGB components", inline=True
    )
    # Named for what it scales, not for the statistic it uses: "percentile"
    # previously read as if it were related to the keep-% thresholds, which it
    # is not. This only sets how the RGB channels are stretched.
    clip_ui = mo.ui.radio(
        options=["robust 2-98%", "min-max"],
        value="robust 2-98%",
        label="RGB scaling",
        inline=True,
    )
    mo.hstack(
        [layer_ui, head_ui, n_comp_ui, invert_ui, rgb_pcs_ui, clip_ui],
        justify="start",
        gap=2,
    )
    return clip_ui, head_ui, invert_ui, layer_ui, n_comp_ui, rgb_pcs_ui


@app.cell
def _(np):
    # Display order; keys first is the paper's default facet.
    FACETS = [("tokens", "token"), ("keys", "k"), ("queries", "q"), ("values", "v")]

    def exact_pca(X, k):
        """Deterministic PCA by full SVD of the centred matrix.

        Not run_pca_best: its MPS path is torch.pca_lowrank, a randomized
        algorithm with no seed, so two calls on identical input returned scores
        differing by up to 0.4 — every re-render recoloured every row. At
        896x768 the exact SVD costs ~100 ms, so approximating buys nothing.
        """
        _Xc = X - X.mean(axis=0, keepdims=True)
        _U, _S, _ = np.linalg.svd(_Xc, full_matrices=False)
        _var = _S**2
        _k = int(min(k, _U.shape[1]))
        _scores = _U[:, :_k] * _S[:_k]
        # Signs are arbitrary; pin each component so its largest-magnitude
        # score is positive, so colours stay put across layers and re-renders.
        _idx = np.argmax(np.abs(_scores), axis=0)
        _sign = np.sign(_scores[_idx, np.arange(_scores.shape[1])])
        _sign[_sign == 0] = 1.0
        return (_var / _var.sum())[:_k], (_scores * _sign).astype(np.float32)

    return FACETS, exact_pca


@app.cell
def _(
    FACETS,
    descriptors,
    exact_pca,
    head_ui,
    layer_ui,
    n_comp_ui,
    n_prefix,
    np,
    num_heads,
):
    # Stage 1, for every facet. Depends only on layer, head and component count,
    # so moving a threshold slider below does NOT redo this work.
    stage1 = {}
    for _label, _key in FACETS:
        _raw = descriptors[_key][int(layer_ui.value) - 1][n_prefix:].astype(np.float32)
        _head = int(head_ui.value)
        _sub = ""
        if _head >= 0 and _key != "token":
            _hd = _raw.shape[1] // num_heads
            _raw = _raw[:, _head * _hd : (_head + 1) * _hd]
            _sub = f"head {_head + 1}"
        elif _head >= 0:
            # Block outputs are not split by head; say so rather than silently
            # showing something not comparable with the other rows.
            _sub = "all heads"
        _X = _raw / np.clip(np.linalg.norm(_raw, axis=1, keepdims=True), 1e-12, None)
        _n = int(min(n_comp_ui.value, _X.shape[0], _X.shape[1]))
        _evr, _sc = exact_pca(_X, _n)
        stage1[_label] = {
            "X": _X,
            "evr": np.asarray(_evr, dtype=np.float64) * 100.0,
            "scores": _sc,
            "sub": _sub,
        }
    return (stage1,)


@app.cell
def _(FACETS, math, mo, np, stage1):
    # One threshold per facet, in that facet's own PC1 units. Bounds are taken
    # from each facet's actual PC1 range and rounded to the step so the readout
    # stays short — a raw score prints ~17 digits.
    def _slider_for(label):
        _pc1 = stage1[label]["scores"][:, 0]
        _lo_raw, _hi_raw = float(_pc1.min()), float(_pc1.max())
        _step = max(round((_hi_raw - _lo_raw) / 100.0, 4), 1e-4)
        _lo = math.floor(_lo_raw / _step) * _step
        _hi = math.ceil(_hi_raw / _step) * _step
        _mid = round(float(np.median(_pc1)) / _step) * _step
        return mo.ui.slider(
            start=round(_lo, 4),
            stop=round(_hi, 4),
            step=_step,
            value=round(min(max(_mid, _lo), _hi), 4),
            label=f"{label} PC1 >",
            show_value=True,
        )

    # Derived from the data, so these reset when the layer changes — the price
    # of showing real PC1 values instead of percentages.
    # Displayed next to each facet's own plot rather than as one row up here.
    thresh_ui = mo.ui.dictionary({_label: _slider_for(_label) for _label, _ in FACETS})
    return (thresh_ui,)


@app.cell
def _(
    FACETS,
    captured_label,
    clip_ui,
    exact_pca,
    grid_cols,
    grid_rows,
    head_ui,
    invert_ui,
    layer_ui,
    mo,
    np,
    num_heads,
    plt,
    rgb_pcs_ui,
    stage1,
    theme,
    thresh_ui,
):
    _cols_sel = (1, 2, 3) if rgb_pcs_ui.value == "2,3,4" else (0, 1, 2)
    _need = max(_cols_sel) + 1
    _fig_w = 15.0

    _head_note = (
        f"head {int(head_ui.value) + 1} of {num_heads}"
        if int(head_ui.value) >= 0
        else "heads concatenated"
    )
    _blocks = [
        mo.md(
            f"**{captured_label}** · layer {int(layer_ui.value)} · {_head_note}"
        )
    ]

    for _label, _ in FACETS:
        _X = stage1[_label]["X"]
        _pc1 = stage1[_label]["scores"][:, 0]
        _thr = float(thresh_ui.value[_label])
        _mask = _pc1 <= _thr if invert_ui.value else _pc1 >= _thr

        # One figure per facet so its slider can sit directly above it; a single
        # combined figure cannot have UI elements interleaved between its rows.
        _fig, _axes = plt.subplots(
            1, 2, figsize=(_fig_w, (_fig_w / 2) * grid_rows / grid_cols + 0.55)
        )
        _fig.patch.set_facecolor(theme["bg"])

        # Greyscale: PC1 is a single scalar field being split by a threshold, so
        # a monochrome ramp reads as "more/less" without inventing hue
        # structure, and keeps it visually distinct from the RGB panel.
        _axes[0].imshow(
            _pc1.reshape(grid_rows, grid_cols), cmap="gray", interpolation="nearest"
        )
        _sub = stage1[_label]["sub"]
        _axes[0].set_title(
            f"{_label}{' · ' + _sub if _sub else ''} — PC1",
            color=theme["fg"],
            fontsize=10,
        )

        _rgb = np.zeros((grid_rows * grid_cols, 3), dtype=np.float32)
        _note = ""
        if int(_mask.sum()) >= max(8, _need):
            _n2 = int(min(max(_need, 4), int(_mask.sum()), _X.shape[1]))
            _e2, _s2 = exact_pca(_X[_mask], _n2)
            _comp = _s2[:, list(_cols_sel)]
            if clip_ui.value == "min-max":
                _lo, _hi = _comp.min(axis=0), _comp.max(axis=0)
            else:
                _lo = np.percentile(_comp, 2, axis=0)
                _hi = np.percentile(_comp, 98, axis=0)
            _rgb[_mask] = np.clip(
                (_comp - _lo) / np.clip(_hi - _lo, 1e-12, None), 0, 1
            )
        else:
            _note = "  (too few patches to re-fit)"

        _axes[1].imshow(_rgb.reshape(grid_rows, grid_cols, 3), interpolation="nearest")
        _axes[1].set_title(
            f"PCs {rgb_pcs_ui.value} re-fit on "
            f"{int(_mask.sum())} patches{_note}",
            color=theme["fg"] if not _note else "#ff9f0a",
            fontsize=10,
        )

        for _a in _axes:
            _a.set_facecolor(theme["bg"])
            _a.set_xticks([])
            _a.set_yticks([])
            for _sp in _a.spines.values():
                _sp.set_visible(False)
        _fig.tight_layout()

        _blocks.append(mo.vstack([thresh_ui[_label], _fig], gap=0))

    mo.vstack(_blocks, gap=1)
    return


@app.cell
def _(FACETS, captured_label, layer_ui, np, plt, stage1, theme):
    _fig, _ax = plt.subplots(figsize=(11, 3.0))
    _fig.patch.set_facecolor(theme["bg"])
    _ax.set_facecolor(theme["bg"])
    for _label, _ in FACETS:
        _evr = stage1[_label]["evr"]
        _ax.plot(np.arange(1, len(_evr) + 1), np.cumsum(_evr), lw=2, label=_label)
    _ax.set_xlabel("Principal component", color=theme["fg"], fontsize=9)
    _ax.set_ylabel("Cumulative variance (%)", color=theme["fg"], fontsize=9)
    _ax.set_ylim(0, 105)
    _ax.tick_params(colors=theme["fg"], labelsize=8)
    for _s in _ax.spines.values():
        _s.set_color(theme["muted"])
    _leg = _ax.legend(fontsize=8, facecolor=theme["bg"], edgecolor=theme["muted"])
    for _t in _leg.get_texts():
        _t.set_color(theme["fg"])
    _ax.set_title(
        f"{captured_label} · how fast each facet concentrates variance "
        f"(layer {int(layer_ui.value)})",
        color=theme["fg"],
        fontsize=10,
    )
    _fig.tight_layout()
    _fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    **Reading this.** The paper's setting is object-centric photographs, where
    PC1 separates an object from its background. An ERA5 composite has no
    background — every patch is atmosphere — so the PC1 split reads as a
    saliency-style partition, not a segmentation. Treat it as "which patches are
    unlike the bulk", not "which patches are the object".

    Each row has its own threshold, in that facet's **own PC1 units** — the
    ranges genuinely differ between facets, so the same number does not mean the
    same thing in two rows. Compare using the **patches kept** count printed
    under each RGB panel, not the slider positions. Because the bounds come from
    the data, these sliders reset when you change the layer.

    The threshold and **RGB scaling** are unrelated. The first decides *which*
    patches survive into the re-fit; the second only decides how the resulting
    three components are stretched into colour. `min-max` is what the paper's
    `pca.py` does; `robust 2-98%` clips the extreme patches that otherwise
    flatten ERA5 frames.

    The RGB is re-fit on whatever survives, so it changes as you move a slider —
    expected, not instability. Colours are arbitrary up to rotation, so they
    carry no meaning across frames or between rows.

    Only **tokens at the last layer** correspond to the stored
    `patch_embeddings`. Every other facet and layer is exploratory and is not
    what retrieval uses. That correspondence requires the model's final
    `LayerNorm`, which is applied here to every block output — without it the
    last-layer tokens sit at cosine 0.79 to the stored vectors, not 1.0.
    """)
    return


if __name__ == "__main__":
    app.run()
