"""Clickable thumbnail gallery as an anywidget.

Replaces the previous Plotly-based gallery, which encoded every thumbnail as a
raw `go.Image(z=...)` pixel array — ~4 MB of JSON for 25 thumbnails, re-sent in
full on every click just to move the selection rectangle. Here the thumbnails
are ordinary `<img>` tags carrying base64 JPEGs (browser-native decode), CSS
Grid owns the column layout, and a click syncs a single integer back to Python
while the red border is painted client-side on the next frame.
"""

from pathlib import Path
from typing import Any, Optional, Sequence

import anywidget
import traitlets

_STATIC = Path(__file__).parent / "static"

# The widget object is rebuilt whenever the thumbnails themselves change — a
# new search, but also a box-colour/thickness/theme tweak that re-annotates
# them — and a fresh widget starts at selected=-1. Stashing the last selection
# here lets the rebuilt gallery keep pointing at the same image, without
# routing it through marimo state (which would make the gallery cell depend on
# the selection and so re-run — and re-send every thumbnail — on every click,
# the exact cost this widget exists to remove).
#
# Keyed by the gallery's id list, so a genuinely new result set starts clean.
_LAST_SELECTION = {"key": None, "index": -1}


def remember_selection(gallery_ids, index):
    """Record the selected tile so a later rebuild of the same gallery keeps it."""
    _LAST_SELECTION["key"] = tuple(gallery_ids)
    _LAST_SELECTION["index"] = int(index)


def recall_selection(gallery_ids):
    """Selected index for this gallery, or -1 if it's a different result set."""
    if _LAST_SELECTION["key"] != tuple(gallery_ids):
        return -1
    idx = _LAST_SELECTION["index"]
    return idx if 0 <= idx < len(gallery_ids) else -1


class PatchGallery(anywidget.AnyWidget):
    """Grid of clickable thumbnails with captions and a selected-tile highlight.

    `thumbs` are complete image sources (``data:image/jpeg;base64,...``), so the
    caller controls annotation and resolution. `selected` is the index of the
    clicked tile, or -1 when nothing is selected; read it in marimo via
    ``mo.ui.anywidget(PatchGallery(...)).value["selected"]``.
    """

    _esm = _STATIC / "patch_gallery.js"
    _css = _STATIC / "patch_gallery.css"

    thumbs = traitlets.List(traitlets.Unicode(), default_value=[]).tag(sync=True)
    captions = traitlets.List(traitlets.Unicode(), default_value=[]).tag(sync=True)
    n_cols = traitlets.Int(3).tag(sync=True)
    thumb_w = traitlets.Int(192).tag(sync=True)
    thumb_h = traitlets.Int(192).tag(sync=True)
    max_h = traitlets.Int(650).tag(sync=True)
    theme = traitlets.Unicode("light").tag(sync=True)
    selected = traitlets.Int(-1).tag(sync=True)

    def __init__(
        self,
        *,
        thumbs: Optional[Sequence[str]] = None,
        captions: Optional[Sequence[str]] = None,
        n_cols: int = 3,
        thumb_w: int = 192,
        thumb_h: int = 192,
        max_h: int = 650,
        theme: str = "light",
        selected: int = -1,
        **kwargs: Any,
    ):
        super().__init__(
            thumbs=list(thumbs or []),
            captions=list(captions or []),
            n_cols=n_cols,
            thumb_w=thumb_w,
            thumb_h=thumb_h,
            max_h=max_h,
            theme=theme,
            selected=selected,
            **kwargs,
        )
