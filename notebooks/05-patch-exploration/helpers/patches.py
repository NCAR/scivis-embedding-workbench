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
):
    """Crop a patch plus `buffer_patches` of context and outline the patch.

    Returns a PIL RGB image upscaled by `scale`. The crop is clamped to the
    image, so patches on an edge return a smaller context window rather than
    being padded or shifted.

    outline_width : border thickness in *display* pixels (after upscaling), so
                    the border keeps a constant visual weight as zoom changes.
                    0 draws no border.
    resample      : key of RESAMPLING. "nearest" keeps the patch grid crisp;
                    the smooth filters look nicer but blur the patch edges you
                    are trying to judge.
    """
    import io
    from PIL import Image, ImageDraw

    filt = getattr(Image.Resampling, RESAMPLING.get(resample, "NEAREST"))
    img = Image.open(io.BytesIO(image_blob)).convert("RGB")
    img_w, img_h = img.size
    left, top, right, bottom = patch_box(
        patch_index, spatial_h, spatial_w, img_w, img_h
    )

    bw = buffer_patches * (img_w / spatial_w)
    bh = buffer_patches * (img_h / spatial_h)
    cl, ct = max(0, round(left - bw)), max(0, round(top - bh))
    cr, cb = min(img_w, round(right + bw)), min(img_h, round(bottom + bh))

    crop = img.crop((cl, ct, cr, cb))
    crop = crop.resize(
        (max(1, (cr - cl) * scale), max(1, (cb - ct) * scale)), filt
    )

    # Outline the true patch within the upscaled context.
    if outline_width > 0:
        draw = ImageDraw.Draw(crop)
        draw.rectangle(
            [
                (left - cl) * scale,
                (top - ct) * scale,
                (right - cl) * scale - 1,
                (bottom - ct) * scale - 1,
            ],
            outline=outline,
            width=int(outline_width),
        )
    return crop


def to_png_bytes(pil_image) -> bytes:
    """Encode a PIL image as PNG bytes (what marimo's mo.image wants)."""
    import io
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return buf.getvalue()
