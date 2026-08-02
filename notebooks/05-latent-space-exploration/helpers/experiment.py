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
class Projection:
    """A 2-D projection table: coordinates plus whatever else was stored.

    `x` and `y` are the entire contract. Everything else -- clusters, times,
    geography, storm context -- is discovered, so a table holding only
    coordinates is a normal case rather than a degraded one, and a projection
    can be explored before it has ever been clustered.
    """

    name: str
    df: Any
    categorical: list = field(default_factory=list)
    continuous: list = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    # Kept so identity columns can be fetched for a handful of sampled rows.
    # `image_id` is a large_string across ~1M rows and is deliberately left out
    # of `df`; hover thumbnails need it for a few hundred of them.
    table: Any = None

    def __len__(self):
        return len(self.df)

    @property
    def is_synthetic(self) -> bool:
        return str(self.metadata.get("synthetic", "")).lower() == "true"

    def kind(self, column: str) -> str:
        """"density", "categorical" or "continuous" for a colour-by choice.

        Unknown columns fall back to density, so a stale widget value cannot
        raise -- it just draws the plot that always works.
        """
        if column in self.categorical:
            return "categorical"
        if column in self.continuous:
            return "continuous"
        return "density"

    def color_by_options(self) -> list:
        """Colour-by choices, density first because it needs only x/y."""
        return ["density", *self.categorical, *self.continuous]

    def summary(self) -> str:
        return _viz.projection_summary(self)


@dataclass
class DisplayOptions:
    """Gallery appearance. Field names match the keys of `display_form()`."""

    n_examples: int = 12
    # No widget: the gallery fits as many columns as the container allows.
    # Set a positive value programmatically to pin an exact count, e.g. for a
    # reproducible figure.
    columns: int = 0
    buffer_patches: int = 2
    zoom: int = 1
    border_color: str = "#00ff88"
    border_width: int = 1
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

    def summary(self) -> str:
        return _viz.experiment_summary(self.config, self.grid, self.n_patches)

    def list_projections(self, prefix: str = "umap_") -> list:
        """Projection tables sitting alongside the embeddings in this experiment."""
        return _data.list_projection_tables(self.db_path, self.name, prefix=prefix)

    def load_projection(self, name: str) -> "Projection":
        """Load a whole projection table and work out its colour-by columns.

        Roles come from the table's own schema metadata when the writer recorded
        them, and are inferred from dtypes otherwise. Either way they are pruned
        against the loaded frame, so a column that is constant in *this*
        experiment never reaches the dropdown.
        """
        tbl = _data.open_projection_table(self.db_path, self.name, name)
        df = _data.load_projection_frame(tbl)
        roles = _data.get_color_roles(tbl) or _data.infer_color_roles(df)
        roles = _data.usable_color_columns(df, roles)
        return Projection(
            name=name,
            df=df,
            categorical=roles["categorical"],
            continuous=roles["continuous"],
            metadata=_data.get_table_metadata(tbl),
            table=tbl,
        )

    def hover_frame(
        self,
        projection: "Projection",
        n: int = 300,
        seed: int = 0,
        thumbnails: bool = False,
        buffer_patches: int = 2,
        scale: int = 2,
        quality: int = 80,
        max_thumbnails: int = 300,
    ):
        """The scatter's hover sample, optionally with a patch crop per point.

        Returns a copy of `n` sampled rows of `projection.df`. With
        `thumbnails=True` it also carries `image_id` and a `thumb` column of
        JPEG data URIs -- the same crop-with-context the gallery renders.

        Cost is linear and dominated by cropping: measured at roughly 3 ms and
        3.7 KB per point, so 300 points is about 1 s and 1 MB while 2000 would
        be 6 s and 7 MB embedded in the document. `max_thumbnails` truncates the
        overlay -- not just the crops -- so every remaining glyph has an image
        and there are no half-loaded tooltips. Build this in its own cell so
        changing the colormap or the background does not pay the cost again.
        """
        df = projection.df
        n = min(int(n), len(df))
        if n <= 0:
            return df.iloc[:0].copy()

        sample = df.sample(n, random_state=seed)
        if not thumbnails:
            return sample

        if self.src_img_tbl is None or projection.table is None:
            return sample  # no source images: hover still works, without crops

        if len(sample) > max_thumbnails:
            sample = sample.iloc[:max_thumbnails]
        return self._add_crops(
            sample, projection, buffer_patches, scale, quality
        )

    def tile_frame(
        self,
        projection: "Projection",
        n: int = 24,
        seed: int = 0,
        buffer_patches: int = 3,
        scale: int = 3,
        quality: int = 82,
    ):
        """Rows for a few patch crops drawn *on* the scatter, spread across it.

        Selection is grid coverage rather than a random sample, so tiles reach
        the sparse arms of a projection instead of piling up in the dense core.
        See `data.representative_offsets`.

        Picked once over the full extent; zooming in does not repick them, so a
        tight zoom may hold few tiles or none.

        A wider context ring than the hover crops: these are read at a glance
        from across the plot rather than inspected up close, so more surrounding
        weather makes them easier to tell apart.
        """
        df = projection.df
        if projection.table is None or self.src_img_tbl is None or not len(df):
            return df.iloc[:0].copy()

        offsets = _data.representative_offsets(df, n=n)
        if not len(offsets):
            return df.iloc[:0].copy()
        return self._add_crops(
            df.iloc[offsets], projection, buffer_patches, scale, quality
        )

    def region_offsets(self, projection: "Projection", bounds, seed: int = 0):
        """Row offsets inside a box, shuffled once so paging can walk them.

        `bounds` is (x0, y0, x1, y1) from a box selection. The mask runs over the
        *whole* projection frame -- every row of the table, not the handful the
        hover overlay happens to draw -- which is the point of selecting a region
        rather than picking glyphs. Measured at 2.7 ms over 982k rows.

        The shuffle matters more than it looks. Table rows are ordered by
        image_id, so consecutive offsets are neighbouring patches of the *same*
        frame: a sequential page of 12 came from one parent image, where a
        shuffled page came from twelve. Twelve crops of one weather frame says
        nothing about a region. Shuffling once and holding the permutation means
        paging traverses the selection instead of redrawing it each time.
        """
        import numpy as np

        df = projection.df
        if bounds is None or not len(df):
            return np.empty(0, dtype=int)

        x0, y0, x1, y1 = bounds
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)

        xs = df["x"].to_numpy()
        ys = df["y"].to_numpy()
        inside = (xs >= x0) & (xs <= x1) & (ys >= y0) & (ys <= y1)
        offsets = np.flatnonzero(inside)
        return np.random.default_rng(seed).permutation(offsets)

    def patch_sample(self, projection: "Projection", offsets) -> PatchSample:
        """A `PatchSample` for the given row offsets of a projection.

        `image_id` is deliberately absent from the projection frame, so it is
        fetched for just these rows -- the same `take` trick `hover_frame` uses,
        ~2 ms for a page. `X` is left None: `gallery()` never reads it, so no
        embeddings are loaded to show patches.
        """
        import numpy as np

        offsets = np.asarray(offsets, dtype=int)
        if not len(offsets) or projection.table is None:
            return PatchSample(X=None, image_ids=np.array([]), patch_indices=np.array([]))

        ids = (
            projection.table.to_lance()
            .take(np.sort(offsets), columns=["image_id", "patch_index"])
            .to_pandas()
        )
        return PatchSample(
            X=None,
            image_ids=ids["image_id"].to_numpy(),
            patch_indices=ids["patch_index"].to_numpy(),
        )

    def _add_crops(self, sample, projection, buffer_patches, scale, quality):
        """Attach `image_id` and a `thumb` column of JPEG data URIs.

        `projection.df` was loaded whole and in table order, so a positional
        index is a row offset -- `take` then fetches the identity columns for
        just these rows, which is why `image_id` can stay out of the frame.
        """
        offsets = sample.index.to_numpy()
        ids = (
            projection.table.to_lance()
            .take(offsets, columns=["image_id", "patch_index"])
            .to_pandas()
        )
        sample = sample.copy()
        sample["image_id"] = ids["image_id"].to_numpy()

        spatial_h, spatial_w = self.grid
        blobs = _data.fetch_image_blobs(self.src_img_tbl, sample["image_id"].tolist())

        thumbs = []
        for image_id, patch_index in zip(sample["image_id"], sample["patch_index"]):
            row = blobs.get(image_id)
            if row is None:
                thumbs.append("")
                continue
            crop = _patches.crop_patch_with_buffer(
                row["image_blob"],
                int(patch_index),
                spatial_h,
                spatial_w,
                buffer_patches=buffer_patches,
                scale=scale,
            )
            thumbs.append(_patches.to_jpeg_uri(crop, quality=quality))
        sample["thumb"] = thumbs
        return sample

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
