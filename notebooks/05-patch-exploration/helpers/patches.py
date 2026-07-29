"""Patch imaging for the patch-embedding exploration notebook.

A ViT patch is `patch_size` pixels square in model space. When the stored image
is close to the model input size, one patch is only a handful of pixels and is
unreadable on its own, so everything here is built around cropping a patch
*with surrounding context* and marking where the actual patch sits inside it.

Geometry is never assumed: the patch grid is passed in (see `geometry.py`), and
pixel dimensions are read off the decoded image itself.

Bytes in / PIL out -- no marimo or plotting imports.
"""

from helpers.geometry import patch_box

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
    """Encode a PIL image as PNG bytes."""
    import io
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return buf.getvalue()


def to_png_uri(pil_image) -> str:
    """Data URI of a PIL image as lossless PNG -- used for the patch crop."""
    import base64
    return "data:image/png;base64," + base64.b64encode(to_png_bytes(pil_image)).decode()
