import marimo

__generated_with = "0.23.13"
app = marimo.App(layout_file="layouts/latent_exploration.grid.json")


@app.cell
def _():
    import base64

    import marimo as mo
    import IPython  # must be imported before lancedb to avoid circular import via tqdm→ipywidgets
    import lancedb
    import numpy as np

    from helpers.data import list_experiments, load_patch_matrix, open_experiment
    from wigglystuff import ColorPicker

    from helpers.patches import (
        RESAMPLING,
        crop_patch_with_buffer,
        fetch_image_blobs,
        format_latlon,
        frame_preview_uri,
        get_spatial_extent,
        open_source_table,
        patch_latlon,
        patch_grid,
        to_png_bytes,
    )

    return (
        ColorPicker,
        RESAMPLING,
        base64,
        crop_patch_with_buffer,
        fetch_image_blobs,
        format_latlon,
        frame_preview_uri,
        get_spatial_extent,
        list_experiments,
        load_patch_matrix,
        mo,
        np,
        open_experiment,
        open_source_table,
        patch_grid,
        patch_latlon,
        to_png_bytes,
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
def _(mo):
    sample_size = mo.ui.slider(
        start=10_000,
        stop=500_000,
        step=10_000,
        value=50_000,
        label="Patches to load",
        show_value=True,
    )
    sampling = mo.ui.dropdown(
        options=["random", "head"],
        value="random",
        label="Sampling",
    )
    mo.hstack([sample_size, sampling], justify="start")
    return sample_size, sampling


@app.cell
def _(load_patch_matrix, patch_emb_tbl, sample_size, sampling):
    if patch_emb_tbl is None:
        n_patches = None
        X, image_ids, patch_indices = None, None, None
    else:
        n_patches = patch_emb_tbl.count_rows()
        # "head" is the first N rows in ingest order, so it covers only the
        # earliest images — useful for a quick look, misleading for structure.
        X, image_ids, patch_indices = load_patch_matrix(
            patch_emb_tbl,
            limit=int(sample_size.value),
            random_sample=sampling.value == "random",
        )
    return X, image_ids, n_patches, patch_indices


@app.cell
def _(X, config, image_ids, mo, n_patches, patch_grid, patch_indices):
    if n_patches is None:
        status = mo.callout(
            mo.md("Enter a DB path and pick an experiment to load patches."),
            kind="info",
        )
    else:
        _h, _w = patch_grid(config)
        status = mo.md(
            f"**Model:** `{config.get('model_name', '?')}`  ·  "
            f"**Patches in table:** {n_patches:,}  ·  "
            f"**Loaded:** {X.shape[0]:,} × {X.shape[1]}  ·  "
            f"**Source images covered:** {len(set(image_ids)):,}  ·  "
            f"**Patch grid:** {_h}×{_w}  ·  "
            f"**Max patch_index:** {int(patch_indices.max())}"
        )
    status
    return


@app.cell
def _(config, mo, patch_grid):
    if config is None:
        _blurb = "Load an experiment to see patch geometry."
    else:
        _gh, _gw = patch_grid(config)
        _iw, _ih = int(config["image_w"]), int(config["image_h"])
        # Derived, not assumed: a different experiment or dataset changes every
        # one of these numbers.
        _pw, _ph = _iw / _gw, _ih / _gh
        _blurb = (
            f"Each patch is {_pw:g}×{_ph:g} px of a {_iw}×{_ih} image "
            f"({_gh}×{_gw} grid) — too small to read alone, so each tile shows "
            "the patch plus a ring of context, with the patch itself outlined."
        )
    mo.md(f"""
    ## Patch crops

    {_blurb}
    """)
    return


@app.cell
def _(
    config,
    embedding_db_path,
    get_spatial_extent,
    open_source_table,
    patch_emb_tbl,
):
    # The raw images live in a separate LanceDB; config records where. The
    # geographic extent is on that table's metadata, not in config.
    if patch_emb_tbl is None:
        src_img_tbl, spatial_extent = None, None
    else:
        src_img_tbl = open_source_table(embedding_db_path.value, config)
        spatial_extent = get_spatial_extent(src_img_tbl)
    return spatial_extent, src_img_tbl


@app.cell(hide_code=True)
def _(ColorPicker, RESAMPLING, mo):
    n_examples = mo.ui.slider(
        start=4, stop=200, step=4, value=12, label="Patches", show_value=True
    )
    n_columns = mo.ui.slider(
        start=1, stop=14, step=1, value=4, label="Columns", show_value=True
    )
    buffer_patches = mo.ui.slider(
        start=0, stop=6, step=1, value=2, label="Context (patches)", show_value=True
    )
    zoom = mo.ui.slider(start=1, stop=12, step=1, value=4, label="Zoom", show_value=True)
    # marimo has no native color input; wigglystuff's ColorPicker is a real
    # <input type="color"> and is already a project dependency.
    border_color = mo.ui.anywidget(ColorPicker(color="#00ff88"))
    border_width = mo.ui.slider(
        start=0, stop=10, step=1, value=4, label="Border px (0 = off)", show_value=True
    )
    resample = mo.ui.dropdown(
        options=list(RESAMPLING), value="nearest", label="Resampling"
    )

    mo.vstack(
        [
            mo.hstack([n_examples, n_columns, buffer_patches, zoom], justify="start"),
            mo.hstack(
                [
                    mo.vstack([mo.md("**Border color**"), border_color], gap=0.2),
                    border_width,
                    resample,
                ],
                justify="start",
                align="center",
            ),
        ],
        gap=0.5,
    )
    return (
        border_color,
        border_width,
        buffer_patches,
        n_columns,
        n_examples,
        resample,
        zoom,
    )


@app.cell(hide_code=True)
def _(
    base64,
    border_color,
    border_width,
    buffer_patches,
    config,
    crop_patch_with_buffer,
    fetch_image_blobs,
    format_latlon,
    frame_preview_uri,
    image_ids,
    mo,
    n_columns,
    n_examples,
    np,
    patch_grid,
    patch_indices,
    patch_latlon,
    resample,
    spatial_extent,
    src_img_tbl,
    to_png_bytes,
    zoom,
):
    if src_img_tbl is None or image_ids is None:
        gallery = mo.callout(
            mo.md("Load an experiment to see patch crops."), kind="info"
        )
    else:
        _rng = np.random.default_rng(0)
        _pick = _rng.choice(
            len(image_ids), min(int(n_examples.value), len(image_ids)), replace=False
        )
        # One scan covers every picked patch; patches sharing a parent image
        # cost nothing extra.
        _rows = fetch_image_blobs(
            src_img_tbl, [image_ids[i] for i in _pick], extra_cols=["dt", "max_wind_kts"]
        )
        _h, _w = patch_grid(config)

        # Hover previews are keyed by image_id: decoding the source PNG is the
        # most expensive step per tile, so tiles sharing a parent reuse it.
        _previews = {}
        # Preview width in px; the JPEG itself is full resolution.
        _preview_w = 448
        _tiles = []
        for _i in _pick:
            _row = _rows.get(image_ids[_i])
            if _row is None:
                continue
            if image_ids[_i] not in _previews:
                _previews[image_ids[_i]] = frame_preview_uri(_row["image_blob"])
            _crop = crop_patch_with_buffer(
                _row["image_blob"],
                patch_indices[_i],
                _h,
                _w,
                buffer_patches=int(buffer_patches.value),
                scale=int(zoom.value),
                outline=border_color.color,
                outline_width=int(border_width.value),
                resample=resample.value,
            )
            _r, _c = divmod(int(patch_indices[_i]), _w)
            # Falls back to grid coords when the source table has no extent.
            _geo = (
                format_latlon(*patch_latlon(patch_indices[_i], _h, _w, spatial_extent))
                if spatial_extent
                else None
            )
            # Frames with no storm carry NaN, not None, so `is not None` alone
            # would print "nan kts" on every calm frame.
            _wind = _row.get("max_wind_kts")
            _has_wind = _wind is not None and np.isfinite(_wind)
            # NaT, like NaN, compares unequal to itself.
            _dt = _row.get("dt")
            _stamp = (
                _dt.strftime("%Y-%m-%d %H:%M")
                if _dt is not None and _dt == _dt
                else "no date"
            )
            _uri = "data:image/png;base64," + base64.b64encode(
                to_png_bytes(_crop)
            ).decode()
            _label = (
                f"{_stamp}" + (f" · {_geo}" if _geo else "")
                + f"<br>patch {int(patch_indices[_i])} (r{_r}, c{_c})"
                + (f" · {_wind:.0f} kts" if _has_wind else "")
            )
            # The patch marker on the hover frame is a CSS box at percentage
            # coordinates, not drawn into the pixels: it costs no image work and
            # stays correct at any preview width.
            _mark = (
                f"left:{_c / _w * 100:.4f}%;top:{_r / _h * 100:.4f}%;"
                f"width:{100 / _w:.4f}%;height:{100 / _h:.4f}%;"
                f"border:2px solid {border_color.color}"
            )
            # max-width:none defeats the inherited img rule that would shrink
            # tiles to fit their cell.
            _tiles.append(
                f"<figure class='pc-tile' style='margin:0'>"
                f"<img src='{_uri}' style='display:block;width:{_crop.size[0]}px;"
                f"max-width:none' />"
                f"<figcaption style='font-size:0.75em;opacity:0.8;"
                f"text-align:center;margin-top:0.25rem'>{_label}</figcaption>"
                f"<span class='pc-full'>"
                f"<img src='{_previews[image_ids[_i]]}' "
                f"style='display:block;width:{_preview_w}px;max-width:none' />"
                f"<span class='pc-mark' style='{_mark}'></span>"
                f"</span>"
                f"</figure>"
            )

        # A grid with fixed-width columns, not flex rows: a short final row
        # leaves its cells empty instead of widening the remaining tiles.
        # Columns are sized in px so tile size never depends on how many tiles
        # share the row, or on how wide the notebook is.
        _per_row = int(n_columns.value)
        _tile_w = _tiles and _crop.size[0] or 0
        # Hover is pure CSS -- no JS, no round trip to the kernel. The preview
        # is positioned relative to the tile and lifted above its neighbours.
        _css = (
            "<style>"
            ".pc-tile{position:relative}"
            ".pc-full{display:none;position:absolute;left:0;top:0;z-index:30;"
            "box-shadow:0 6px 24px rgba(0,0,0,.55)}"
            ".pc-tile:hover .pc-full{display:block}"
            ".pc-mark{position:absolute;box-sizing:border-box;pointer-events:none}"
            "</style>"
        )
        gallery = mo.Html(
            _css
            + f"<div style='display:grid;"
            f"grid-template-columns:repeat({_per_row}, {_tile_w}px);"
            f"gap:1rem;justify-content:start;overflow-x:auto'>"
            + "".join(_tiles)
            + "</div>"
        )
    gallery
    return


if __name__ == "__main__":
    app.run()
