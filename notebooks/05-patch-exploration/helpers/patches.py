"""Patch-crop helpers for the patch-embedding exploration notebook.

A ViT patch is `patch_size` pixels square in model space. When the stored image
is close to the model input size, one patch is only a handful of pixels and is
unreadable on its own, so everything here is built around cropping a patch
*with surrounding context* and marking where the actual patch sits inside it.

Geometry is never assumed: the patch grid comes from the experiment config via
`patch_grid()`, and pixel dimensions are read off the decoded image itself.

PIL in / PIL out -- no marimo or plotting imports.
"""


def patch_grid(config: dict):
    """Return (spatial_h, spatial_w) -- the patch grid for this experiment.

    Prefers the attention grid the embedding run recorded. Falls back to
    deriving it from image dimensions and patch size for configs written before
    those keys existed. The grid is not assumed square: ERA5 runs are
    rectangular (e.g. 16x56).
    """
    def _as_int(key):
        try:
            return int(config.get(key))
        except (TypeError, ValueError):
            return None

    h, w = _as_int("attention_spatial_h"), _as_int("attention_spatial_w")
    if h and w:
        return h, w

    patch_size = _as_int("patch_size")
    img_w, img_h = _as_int("image_w"), _as_int("image_h")
    if patch_size and img_w and img_h:
        return img_h // patch_size, img_w // patch_size

    raise KeyError(
        "config has neither attention_spatial_h/w nor image_w/image_h/patch_size; "
        "cannot determine the patch grid"
    )


def resolve_source_path(experiments_db_path: str, source_path_from_config: str):
    """Resolve the config's source_path to an absolute path, or None.

    Walks up from the experiments DB until the relative source path resolves,
    so the same config works from different mount points.
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


def open_source_table(experiments_db_path: str, config: dict):
    """Open the raw image table this experiment was built from. None if absent."""
    import lancedb
    src_path = resolve_source_path(experiments_db_path, config.get("source_path", ""))
    if src_path is None:
        return None
    db = lancedb.connect(src_path)
    return db.open_table(config.get("raw_table", "images"))


def fetch_image_blobs(src_img_tbl, image_ids, extra_cols=None):
    """Batch-fetch {image_id: row_dict} for the given ids.

    Uses the Lance scanner with a pyarrow isin filter rather than
    .search().where(): .search() is the ANN vector API and can silently drop
    rows. One scan serves every patch that shares a parent image.
    """
    import pyarrow.compute as pc

    if src_img_tbl is None or not len(image_ids):
        return {}
    wanted = list(dict.fromkeys(image_ids))  # de-dupe, keep order
    cols = ["id", "image_blob"] + list(extra_cols or [])
    cols = [c for c in dict.fromkeys(cols) if c in src_img_tbl.schema.names]
    table = (
        src_img_tbl.to_lance()
        .scanner(columns=cols, filter=pc.field("id").isin(wanted))
        .to_table()
        .to_pandas()
    )
    return {row["id"]: row for _, row in table.iterrows()}


def get_spatial_extent(src_img_tbl):
    """Read lat/lon bounds from the source table's schema metadata, or None.

    The experiment config does not carry the extent -- only the raw image
    table's `dataset_info` blob does. Returns None rather than guessing when
    the metadata is absent, so callers can fall back to grid coordinates.
    """
    import json

    if src_img_tbl is None:
        return None
    raw = (src_img_tbl.schema.metadata or {}).get(b"dataset_info")
    if not raw:
        return None
    extent = (json.loads(raw) or {}).get("spatial_extent") or {}
    needed = ("lat_min", "lat_max", "lon_min", "lon_max")
    return extent if all(k in extent for k in needed) else None


def patch_latlon(patch_index: int, spatial_h: int, spatial_w: int, extent: dict):
    """Centre (lat, lon) of a patch, in the extent's own longitude convention.

    Assumes row 0 is north and column 0 is west. That is not recorded anywhere
    in the metadata, so it was checked against IBTrACS storm positions on this
    dataset: Hurricane Irma (2017-09-06 06:00, 155 kts, 17.7N 61.9W) maps to
    r13/c30, which lands on the vortex in the imagery.
    """
    row, col = divmod(int(patch_index), int(spatial_w))
    lat_span = extent["lat_max"] - extent["lat_min"]
    lon_span = extent["lon_max"] - extent["lon_min"]
    lat = extent["lat_max"] - (row + 0.5) * lat_span / spatial_h
    lon = extent["lon_min"] + (col + 0.5) * lon_span / spatial_w
    return lat, lon


def format_latlon(lat: float, lon: float) -> str:
    """'23.1°N, 95.6°W'. Accepts longitude as 0-360 or -180..180."""
    lon = ((lon + 180) % 360) - 180
    return (
        f"{abs(lat):.1f}°{'N' if lat >= 0 else 'S'}, "
        f"{abs(lon):.1f}°{'E' if lon >= 0 else 'W'}"
    )


def patch_box(patch_index: int, spatial_h: int, spatial_w: int, img_w: int, img_h: int):
    """Pixel box (left, top, right, bottom) of one patch in the stored image.

    patch_index is row-major over a spatial_h x spatial_w grid -- note the grid
    is rectangular here (16x56), so the divisor is spatial_w, not a square side.
    """
    row, col = divmod(int(patch_index), int(spatial_w))
    pw = img_w / spatial_w
    ph = img_h / spatial_h
    return (round(col * pw), round(row * ph), round((col + 1) * pw), round((row + 1) * ph))


RESAMPLING = {
    "nearest": "NEAREST",
    "bilinear": "BILINEAR",
    "bicubic": "BICUBIC",
    "lanczos": "LANCZOS",
}


def crop_patch_with_buffer(
    image_blob,
    patch_index: int,
    spatial_h: int,
    spatial_w: int,
    buffer_patches: int = 2,
    scale: int = 4,
    outline: str = "#00ff88",
    outline_width: int = 4,
    resample: str = "nearest",
    pad_color: str = "#1b1b1b",
):
    """Crop a patch plus `buffer_patches` of context and outline the patch.

    Every tile is exactly (2 * buffer_patches + 1) patches square, whatever the
    patch's position. Where the context window runs past the image the tile is
    padded with `pad_color` rather than being cropped short, so the patch stays
    centred and every tile shares one scale. Clamping instead would return
    smaller images for edge patches, which a flex layout then stretches back up
    -- making edge patches look zoomed in relative to interior ones. On a
    16x56 grid with buffer 2 that affects ~30% of patches, since the grid is
    only 16 rows tall.

    outline_width : border thickness in *display* pixels (after upscaling), so
                    the border keeps a constant visual weight as zoom changes.
                    0 draws no border.
    resample      : key of RESAMPLING. "nearest" keeps the patch grid crisp;
                    the smooth filters look nicer but blur the patch edges you
                    are trying to judge.
    pad_color     : fill for off-image area; marks where the image ends.
    """
    import io
    from PIL import Image, ImageDraw

    filt = getattr(Image.Resampling, RESAMPLING.get(resample, "NEAREST"))
    img = Image.open(io.BytesIO(image_blob)).convert("RGB")
    img_w, img_h = img.size
    left, top, right, bottom = patch_box(
        patch_index, spatial_h, spatial_w, img_w, img_h
    )

    # Size the window from the nominal patch size rather than from this patch's
    # rounded box: when img_w / spatial_w is not an integer the box can differ
    # by a pixel between columns, which would make tiles differ in size.
    patch_w, patch_h = round(img_w / spatial_w), round(img_h / spatial_h)
    bw = round(buffer_patches * img_w / spatial_w)
    bh = round(buffer_patches * img_h / spatial_h)
    win_l, win_t = left - bw, top - bh
    win_w, win_h = patch_w + 2 * bw, patch_h + 2 * bh

    # Paste whatever part of the window actually exists onto a filled canvas.
    canvas = Image.new("RGB", (win_w, win_h), pad_color)
    src_l, src_t = max(0, win_l), max(0, win_t)
    src_r, src_b = min(img_w, win_l + win_w), min(img_h, win_t + win_h)
    if src_r > src_l and src_b > src_t:
        canvas.paste(
            img.crop((src_l, src_t, src_r, src_b)), (src_l - win_l, src_t - win_t)
        )

    crop = canvas.resize((max(1, win_w * scale), max(1, win_h * scale)), filt)

    # Outline the true patch within the upscaled context.
    if outline_width > 0:
        draw = ImageDraw.Draw(crop)
        draw.rectangle(
            [
                (left - win_l) * scale,
                (top - win_t) * scale,
                (left - win_l + patch_w) * scale - 1,
                (top - win_t + patch_h) * scale - 1,
            ],
            outline=outline,
            width=int(outline_width),
        )
    return crop


def frame_preview_uri(image_blob, quality: int = 75) -> str:
    """Data URI of the whole frame as JPEG, for hover previews.

    The stored PNG is ~178 KB; the same 896x256 pixels as JPEG q75 are ~12 KB,
    because these fields are smooth. Lossy is fine for a context view -- the
    patch crop itself stays lossless, since that is the one being inspected.
    Cache the result per image_id: decoding the source PNG costs ~3 ms and is
    the most expensive step in building a tile.
    """
    import base64
    import io

    from PIL import Image

    img = Image.open(io.BytesIO(image_blob)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def to_png_bytes(pil_image) -> bytes:
    """Encode a PIL image as PNG bytes (what marimo's mo.image wants)."""
    import io
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return buf.getvalue()
