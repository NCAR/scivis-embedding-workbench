"""High-level API for exploring a patch-embedding experiment.

Wraps the plain functions in `data`, `geometry`, `patches` and `viz` so a
notebook can work in whole objects:

    exp = PatchExperiment.open(db_path, "dinov3_12h")
    sample = exp.load_patches(limit=50_000)
    exp.gallery(sample, DisplayOptions(zoom=6))

The underlying functions stay importable and usable on their own.
"""

from dataclasses import dataclass, field
from typing import Any

from helpers import data as _data
from helpers import patches as _patches
from helpers import viz as _viz
from helpers.geometry import patch_grid


@dataclass
class PatchSample:
    """A loaded slice of the patch table: vectors plus their identity."""

    X: Any
    image_ids: Any
    patch_indices: Any

    def __len__(self):
        return len(self.image_ids)

    def pick(self, n: int, seed: int = 0):
        """Row offsets of `n` randomly chosen patches from this sample."""
        import numpy as np

        rng = np.random.default_rng(seed)
        return rng.choice(len(self), min(int(n), len(self)), replace=False)


@dataclass
class DisplayOptions:
    """Gallery appearance. Field names match the keys of `display_form()`."""

    n_examples: int = 12
    # No widget: the gallery fits as many columns as the container allows.
    # Set a positive value programmatically to pin an exact count, e.g. for a
    # reproducible figure.
    columns: int = 0
    buffer_patches: int = 2
    zoom: int = 4
    border_color: str = "#00ff88"
    border_width: int = 4
    resample: str = "nearest"
    preview_width: int = 448
    show_preview: bool = True
    seed: int = 0


@dataclass
class PatchExperiment:
    """One experiment: its config, its patch table, and the images behind it."""

    db_path: str
    name: str
    config: dict
    patch_tbl: Any
    src_img_tbl: Any = None
    extent: dict = field(default=None)

    @classmethod
    def open(cls, db_path: str, name: str) -> "PatchExperiment":
        """Open an experiment and everything hanging off it.

        The source images live in a separate LanceDB recorded in config, and
        the geographic extent is on *that* table's metadata rather than in
        config. Both are optional: a missing source DB leaves the gallery
        unavailable but the embeddings still load.
        """
        config, patch_tbl = _data.open_experiment(db_path, name)
        src_img_tbl = _data.open_source_table(db_path, config)
        return cls(
            db_path=db_path,
            name=name,
            config=config,
            patch_tbl=patch_tbl,
            src_img_tbl=src_img_tbl,
            extent=_data.get_spatial_extent(src_img_tbl),
        )

    @property
    def grid(self):
        """(spatial_h, spatial_w) of the patch grid."""
        return patch_grid(self.config)

    @property
    def n_patches(self) -> int:
        return self.patch_tbl.count_rows()

    def load_patches(
        self, limit: int = None, random_sample: bool = True, seed: int = 42
    ) -> PatchSample:
        """Load patch vectors. See `data.load_patch_matrix` for the sampling note."""
        return PatchSample(
            *_data.load_patch_matrix(
                self.patch_tbl, limit=limit, random_sample=random_sample, seed=seed
            )
        )

    def summary(self, sample: PatchSample) -> str:
        return _viz.experiment_summary(self.config, self.grid, self.n_patches, sample)

    def geometry_note(self) -> str:
        return _viz.geometry_note(self.config, self.grid)

    def gallery(self, sample: PatchSample, opts: DisplayOptions = None) -> str:
        """HTML for a gallery of patch crops drawn from `sample`.

        One scan fetches every parent frame, and hover previews are cached per
        image_id, so tiles sharing a parent cost one decode rather than two.
        Cost scales with tile count, not table size: ~120 ms for 12 tiles,
        ~1.5 s for 200.
        """
        opts = opts or DisplayOptions()
        if self.src_img_tbl is None:
            return "<em>No source image table for this experiment.</em>"

        grid = self.grid
        picks = sample.pick(opts.n_examples, seed=opts.seed)
        rows = _data.fetch_image_blobs(
            self.src_img_tbl,
            [sample.image_ids[i] for i in picks],
            extra_cols=["dt", "max_wind_kts"],
        )

        previews, tiles, tile_width = {}, [], 0
        for i in picks:
            image_id = sample.image_ids[i]
            row = rows.get(image_id)
            if row is None:
                continue
            patch_index = sample.patch_indices[i]

            crop = _patches.crop_patch_with_buffer(
                row["image_blob"],
                patch_index,
                *grid,
                buffer_patches=opts.buffer_patches,
                scale=opts.zoom,
                outline=opts.border_color,
                outline_width=opts.border_width,
                resample=opts.resample,
            )
            tile_width = crop.size[0]

            if opts.show_preview and image_id not in previews:
                previews[image_id] = _patches.frame_preview_uri(row["image_blob"])

            tiles.append(
                _viz.tile_html(
                    _patches.to_png_uri(crop),
                    crop.size[0],
                    _viz.patch_caption(row, patch_index, grid, self.extent),
                    preview_uri=previews.get(image_id),
                    preview_width=opts.preview_width,
                    patch_index=patch_index,
                    grid=grid,
                    mark_color=opts.border_color,
                )
            )

        return _viz.gallery_html(tiles, opts.columns, tile_width)
