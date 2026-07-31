"""Patch-grid geometry for the patch-embedding exploration notebook.

Pure arithmetic over the ViT patch grid: which pixels a patch occupies, which
row/column it is, and where it falls on the globe. No I/O, no imaging.

Nothing here assumes a square grid -- ERA5 runs are 16x56 -- and nothing
hardcodes a size: the grid comes from the experiment config via `patch_grid()`.
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


def patch_rowcol(patch_index: int, spatial_w: int):
    """(row, col) of a patch. Row-major, so the divisor is the grid *width*."""
    return divmod(int(patch_index), int(spatial_w))


def patch_box(patch_index: int, spatial_h: int, spatial_w: int, img_w: int, img_h: int):
    """Pixel box (left, top, right, bottom) of one patch in the stored image.

    patch_index is row-major over a spatial_h x spatial_w grid -- note the grid
    is rectangular here (16x56), so the divisor is spatial_w, not a square side.
    """
    row, col = patch_rowcol(patch_index, spatial_w)
    pw = img_w / spatial_w
    ph = img_h / spatial_h
    return (round(col * pw), round(row * ph), round((col + 1) * pw), round((row + 1) * ph))


def patch_latlon(patch_index: int, spatial_h: int, spatial_w: int, extent: dict):
    """Centre (lat, lon) of a patch, in the extent's own longitude convention.

    Assumes row 0 is north and column 0 is west. That is not recorded anywhere
    in the metadata, so it was checked against IBTrACS storm positions on this
    dataset: Hurricane Irma (2017-09-06 06:00, 155 kts, 17.7N 61.9W) maps to
    r13/c30, which lands on the vortex in the imagery.
    """
    row, col = patch_rowcol(patch_index, spatial_w)
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
