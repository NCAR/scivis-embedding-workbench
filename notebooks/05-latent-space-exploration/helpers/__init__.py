"""Helpers for the patch-embedding exploration notebook.

Layered so each module has one job:

    data.py       LanceDB: experiments, patch vectors, projections, source images
    geometry.py   patch grid arithmetic -- rows, columns, pixels, lat/lon
    patches.py    PIL: crop with context, frame previews, encoding
    viz.py        HTML: tiles, captions, gallery layout
    scatter.py    holoviews/datashader: the projection scatter
    controls.py   marimo widgets (the only module importing marimo)
    experiment.py PatchExperiment / PatchSample / Projection / DisplayOptions

The facades are the intended entry point:

    from helpers import PatchExperiment, DisplayOptions

    exp = PatchExperiment.open(db_path, "dinov3_12h")
    sample = exp.load_patches(limit=50_000)
    exp.gallery(sample, DisplayOptions(zoom=6))

Every underlying function stays importable on its own, e.g.
`from helpers.geometry import patch_latlon`.
"""

from helpers.experiment import (
    DisplayOptions,
    PatchExperiment,
    PatchSample,
    Projection,
)

__all__ = ["DisplayOptions", "PatchExperiment", "PatchSample", "Projection"]
