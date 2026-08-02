"""Presentation for the patch-embedding exploration notebook.

Builds HTML strings -- the notebook wraps them in `mo.Html` / `mo.md`. Keeping
marimo out of here means the same rendering can be exercised from a plain
script or a test.
"""

from helpers.geometry import format_latlon, patch_latlon, patch_rowcol

# Hover is pure CSS -- no JS, no round trip to the kernel. The preview is
# positioned relative to the tile and lifted above its neighbours.
GALLERY_CSS = (
    "<style>"
    ".pc-tile{position:relative}"
    ".pc-full{display:none;position:absolute;left:0;top:0;z-index:30;"
    "box-shadow:0 6px 24px rgba(0,0,0,.55)}"
    ".pc-tile:hover .pc-full{display:block}"
    ".pc-mark{position:absolute;box-sizing:border-box;pointer-events:none}"
    "</style>"
)


def experiment_summary(config, grid, n_patches) -> str:
    """One-line markdown summary of the experiment.

    Describes the table rather than a loaded slice of it: patches are now
    reached by selecting a region of the projection, so nothing is read up front
    and there is no sample to report.
    """
    spatial_h, spatial_w = grid
    return (
        f"**Model:** `{config.get('model_name', '?')}`  ·  "
        f"**Patches in table:** {n_patches:,}  ·  "
        f"**Patch grid:** {spatial_h}×{spatial_w}  ·  "
        f"**Embedding dim:** {config.get('embedding_dim', '?')}"
    )


def projection_summary(projection) -> str:
    """Markdown summary of a loaded projection.

    Cluster stats appear only when a `cluster` column exists -- a projection
    that was never clustered should say nothing about clusters rather than
    report zeros. A synthetic table announces itself loudly, so a made-up
    scatter is never mistaken for real projection output.
    """
    df = projection.df
    parts = [
        f"**Table:** `{projection.name}`",
        f"**Rows:** {len(projection):,}",
    ]
    if "cluster" in df.columns:
        labels = df["cluster"]
        n_clusters = int((labels[labels >= 0]).nunique())
        noise = float((labels < 0).mean())
        parts.append(f"**Clusters:** {n_clusters} (+{noise:.1%} noise)")
    parts.append(
        f"**Colour by:** {len(projection.categorical)} categorical, "
        f"{len(projection.continuous)} continuous"
    )
    line = "  ·  ".join(parts)

    if projection.is_synthetic:
        note = projection.metadata.get("note", "This table is synthetic.")
        line += f"\n\n> ⚠️ **Synthetic projection.** {note}"
    return line


def geometry_note(config, grid) -> str:
    """Sentence describing the patch geometry, derived from config.

    Every number here changes with the experiment or dataset, so none of them
    are written into the text.
    """
    spatial_h, spatial_w = grid
    img_w, img_h = int(config["image_w"]), int(config["image_h"])
    patch_w, patch_h = img_w / spatial_w, img_h / spatial_h
    return (
        f"Each patch is {patch_w:g}×{patch_h:g} px of a {img_w}×{img_h} image "
        f"({spatial_h}×{spatial_w} grid) — too small to read alone, so each tile "
        "shows the patch plus a ring of context, with the patch itself outlined. "
        "Hover a tile to see the whole frame."
    )


def patch_caption(row, patch_index, grid, extent) -> str:
    """Timestamp, position and storm intensity for one tile.

    Missing values are common and arrive as NaT/NaN rather than None, which
    compare unequal to themselves -- an `is not None` check alone would print
    "nan kts" on every calm frame.
    """
    spatial_h, spatial_w = grid
    row_i, col_i = patch_rowcol(patch_index, spatial_w)

    stamp = row.get("dt")
    stamp = (
        stamp.strftime("%Y-%m-%d %H:%M")
        if stamp is not None and stamp == stamp
        else "no date"
    )

    geo = (
        format_latlon(*patch_latlon(patch_index, spatial_h, spatial_w, extent))
        if extent
        else None
    )

    wind = row.get("max_wind_kts")
    has_wind = wind is not None and wind == wind

    return (
        stamp
        + (f" · {geo}" if geo else "")
        + f"<br>patch {int(patch_index)} (r{row_i}, c{col_i})"
        + (f" · {wind:.0f} kts" if has_wind else "")
    )


def tile_html(
    crop_uri,
    crop_width,
    caption,
    preview_uri=None,
    preview_width=448,
    patch_index=0,
    grid=(1, 1),
    mark_color="#00ff88",
    mark_width=2,
) -> str:
    """One gallery tile: the crop, its caption, and the hover preview.

    The patch marker on the hover frame is a CSS box at percentage coordinates
    rather than drawn into the pixels, so it costs no image work and stays
    correct at any preview width. `max-width:none` defeats the inherited img
    rule that would otherwise shrink tiles to fit their cell.
    """
    spatial_h, spatial_w = grid
    row_i, col_i = patch_rowcol(patch_index, spatial_w)

    hover = ""
    if preview_uri:
        mark = (
            f"left:{col_i / spatial_w * 100:.4f}%;top:{row_i / spatial_h * 100:.4f}%;"
            f"width:{100 / spatial_w:.4f}%;height:{100 / spatial_h:.4f}%;"
            f"border:{mark_width}px solid {mark_color}"
        )
        hover = (
            f"<span class='pc-full'>"
            f"<img src='{preview_uri}' "
            f"style='display:block;width:{preview_width}px;max-width:none' />"
            f"<span class='pc-mark' style='{mark}'></span>"
            f"</span>"
        )

    return (
        f"<figure class='pc-tile' style='margin:0'>"
        f"<img src='{crop_uri}' "
        f"style='display:block;width:{crop_width}px;max-width:none' />"
        f"<figcaption style='font-size:0.75em;opacity:0.8;text-align:center;"
        f"margin-top:0.25rem'>{caption}</figcaption>"
        f"{hover}"
        f"</figure>"
    )


def gallery_html(tiles, columns, tile_width) -> str:
    """Arrange tiles in a grid with fixed-width columns.

    A grid, not flex rows: a short final row leaves its cells empty instead of
    widening the remaining tiles. Tracks are sized in px so tile size never
    depends on how many tiles share the row, or on how wide the notebook is.

    columns : 0 (or None) fits as many columns as the container allows, via
              CSS `auto-fill`. The browser recomputes on resize with no
              re-render -- worth having, since rebuilding the gallery costs
              ~120 ms at 12 tiles and ~1.5 s at 200. Raising Zoom or Context
              widens each tile, so the count drops on its own.

              Note this measures the *cell* width, which marimo constrains,
              not the browser window.

              Deliberately not `minmax(tile_width, 1fr)`: elastic tracks are
              what made edge tiles look magnified before.
    """
    if not tiles:
        return "<em>No tiles to show.</em>"
    track = f"repeat(auto-fill, {tile_width}px)" if not columns else (
        f"repeat({columns}, {tile_width}px)"
    )
    # Centred so leftover width is split between both margins instead of
    # pooling on the right. `safe center` falls back to start-alignment when
    # the tracks overflow the container -- plain `center` would push the first
    # tile off the left edge where scrolling cannot reach it. The unprefixed
    # declaration first, so browsers without `safe` still centre.
    return (
        GALLERY_CSS
        + f"<div style='display:grid;grid-template-columns:{track};"
        f"gap:1rem;justify-content:center;justify-content:safe center;"
        f"overflow-x:auto'>"
        + "".join(tiles)
        + "</div>"
    )
