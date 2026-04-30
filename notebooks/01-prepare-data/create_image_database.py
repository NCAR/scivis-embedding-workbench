import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 1. Data Preparation
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1a. Create a database for Images
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Why a database for images?
    Instead of treating images as loose files in folders, we store them in a database so they become a structured, queryable dataset. This allows fast filtering by time and metadata, consistent preprocessing, reproducible experiments, and direct integration with embedding and search pipelines. The database provides indexing, versioning, and efficient storage of both image content and derived representations, turning the image collection into a reliable data asset rather than an unmanaged directory of files.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    > **Note**
    > Before running this notebook, ensure you have the variable and limits determined. See notebook <>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Imports
    """)
    return


@app.cell
def _():
    import json
    from datetime import datetime, timezone
    from pathlib import Path

    import lancedb
    import pyarrow as pa

    # from helpers.ingest_images import ingest_images_to_table
    from helpers.parallel_ingest_images import ingest_images_to_table

    return Path, datetime, ingest_images_to_table, json, lancedb, pa, timezone


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### User Input
    """)
    return


@app.cell
def _(Path, lancedb):
    PROJECT_ROOT = Path("/glade/work/ncheruku/research/sample_data")

    # Folder holding the source images to ingest
    # image_dir = PROJECT_ROOT / "data" / "processed_rgb_rect"  # local Mac / rectangular daily
    image_dir = PROJECT_ROOT / "data" / "preprocessed_rgb_hourly"

    # ── Database ─────────────────────────────────────────────────────────────
    # Project name — used as the subfolder inside shared_source/ that holds
    # this dataset's LanceDB database. Rename this to separate datasets.
    SOURCE_PROJECT = "era5_hrly_2016_2018_images"

    # LanceDB storage directory — each project lives in its own subfolder
    db_dir = PROJECT_ROOT / "data" / "lancedb" / "shared_source" / SOURCE_PROJECT

    # Generic table name — the project folder already identifies the dataset,
    # so the table itself can have a stable, renameable name.
    IMG_RAW_TBL_NAME = "images"

    # ── Image dimensions ─────────────────────────────────────────────────────
    # Stored image width and height in pixels.
    # Both must be multiples of 16 for DINO patch compatibility.
    # For a 7:2 geographic aspect ratio (lon 70° × lat 20°) use WIDTH=896, HEIGHT=256.
    WIDTH  = 896
    HEIGHT = 256

    # Square thumbnail size stored alongside each image for quick previews
    THUMB_RESOLUTION = 64

    # JPEG compression quality for the thumbnail blob (1–95)
    JPEG_QUALITY = 90

    # ── Temporal extent ──────────────────────────────────────────────────────
    # Date range covered by this dataset (ISO-8601, inclusive)
    TEMPORAL_START = "2016-01-01"
    TEMPORAL_END   = "2018-12-31"

    # ── Filename format ───────────────────────────────────────────────────────
    # strptime pattern used to parse the image timestamp from the filename.
    # Must match the actual filenames in image_dir.
    # DT_FORMAT = "%Y%m%d_rgb.jpeg"      # daily: e.g. 20160101_rgb.jpeg
    DT_FORMAT = "%Y%m%d_%H_rgb.jpeg"     # hourly: e.g. 20171222_16_rgb.jpeg

    # ── Temporal subsampling ──────────────────────────────────────────────────
    # If the source folder contains finer-grained data than you need, set this
    # to a pandas freq string to ingest only the aligned subset.
    # Examples: "3h" (every 3 hours), "6h", "12h", "D" (daily noon), None (all)
    # INGEST_RESOLUTION = None           # ingest all files
    INGEST_RESOLUTION = None             # keep only timestamps aligned to 3-hour boundaries

    # ── Ingest performance ────────────────────────────────────────────────────
    # Number of parallel worker processes for image decoding/resizing
    # Set to (total_cores - 4) to leave headroom for OS and main process
    WORKERS = 28
    # Rows written to LanceDB per transaction (larger = fewer writes, more RAM)
    # 8192 is safe with 128GB RAM; reduces LanceDB commit overhead significantly
    BATCH_SIZE = 8192

    # Connect to DB
    db = lancedb.connect(str(db_dir))
    return (
        BATCH_SIZE,
        DT_FORMAT,
        HEIGHT,
        IMG_RAW_TBL_NAME,
        INGEST_RESOLUTION,
        JPEG_QUALITY,
        PROJECT_ROOT,
        TEMPORAL_END,
        TEMPORAL_START,
        THUMB_RESOLUTION,
        WIDTH,
        WORKERS,
        db,
        db_dir,
        image_dir,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Metadata
    """)
    return


@app.cell
def _(
    HEIGHT,
    JPEG_QUALITY,
    TEMPORAL_END,
    TEMPORAL_START,
    THUMB_RESOLUTION,
    WIDTH,
    datetime,
    json,
    timezone,
):
    # 1. Define the metadata structure

    metadata_dict = {
        # --- 1. CORE IDENTITY ---
        "dataset_name": "ERA5 Hurricane Training Data (RGB Composites)",
        "description": "Hourly weather composites at 3h resolution (MSL Anomaly, Wind, TCWV) for hurricane detection.",
        "author": "cherukuru",
        "generated_by_script": "e5_channels.ipynb",
        "created_at": datetime.now(timezone.utc).isoformat(),  # Dynamic Timestamp
        "row_count": 0,  # updated dynamically after ingestion
        # --- 2. SOURCE PROVENANCE (Sorted by Importance) ---
        "source_metadata": {
            "data_source": "ECMWF: https://cds.climate.copernicus.eu, Copernicus Climate Data Store",
            "conventions": "CF-1.6",
            "conversion_logic": "CISL RDA: Conversion from ECMWF GRIB1 data to netCDF4",
            "netcdf_version": "4.6.3",
            "conversion_platform": "Linux r8i6n32 4.12.14-94.41-default #1 SMP",
        },
        # --- 3. OUTPUT SPECIFICATIONS ---
        "image_specs": {
            "resolution": [WIDTH, HEIGHT],
            "thumb_resolution": [THUMB_RESOLUTION, THUMB_RESOLUTION],
            "format": "JPEG",
            "quality": JPEG_QUALITY,
            "resampling": "Bilinear",
            "projection": "Plate Carrée (Equirectangular)",
        },
        # --- 4. SPATIAL & TEMPORAL BOUNDS ---
        "spatial_extent": {
            "lat_min": 15.0,
            "lat_max": 35.0,
            "lon_min": 260.0,
            "lon_max": 330.0,
            "notes": "North Atlantic / Caribbean (approx 100W to 30W)",
        },
        "temporal_extent": {
            "start": TEMPORAL_START,
            "end": TEMPORAL_END,
            "interval": "3h",  # updated dynamically after ingestion
        },
        # --- 5. PHYSICS & CHANNELS ---
        "channels": {
            "red": {
                "variable": "MSL Pressure Anomaly",
                "range": [-20.0, 20.0],
                "unit": "hPa",
                "logic": "Inverted (Low Pressure = Bright Red)",
            },
            "green": {"variable": "10m Wind Speed (Hourly)", "range": [0.0, 35.0], "unit": "m/s", "logic": "Linear"},
            "blue": {
                "variable": "Total Column Water Vapor (Hourly)",
                "range": [20.0, 70.0],
                "unit": "kg/m^2",
                "logic": "Square Root Scaled",
            },
        },
        # --- 6. HURRICANE EVENT LABELS ---
        "hurricane_source": {
            "dataset": "IBTrACS v04r01",
            "full_name": "International Best Track Archive for Climate Stewardship",
            "url": "https://www.ncei.noaa.gov/products/international-best-track-archive",
            "basin": "North Atlantic (NA)",
            "wind_unit": "knots (WMO standard)",
            "coordinate_system": "degrees_north / degrees_east (negative for west)",
        },
        "hurricane_matching": {
            "logic": "TBD",  # updated dynamically after enrichment
            "columns": "hurricane_present, n_storms, max_wind_kts, max_category, storm_ids, storm_lats, storm_lons",
        },
        "saffir_simpson_scale": {
            "-1 (TD)": "< 34 kts",
            "0 (TS)": "34-63 kts",
            "1 (Cat1)": "64-82 kts",
            "2 (Cat2)": "83-95 kts",
            "3 (Cat3)": "96-112 kts",
            "4 (Cat4)": "113-136 kts",
            "5 (Cat5)": ">= 137 kts",
        },
    }

    # 2. Serialize to JSON Bytes
    arrow_metadata = {b"dataset_info": json.dumps(metadata_dict).encode("utf-8"), b"version": b"1.0"}
    return (arrow_metadata,)


@app.cell
def _(IMG_RAW_TBL_NAME, arrow_metadata, db, pa):
    # Image table
    schema = pa.schema(
        [
            pa.field("id", pa.string()),  # MD5 to generate a deterministic, content-based ID
            pa.field("filename", pa.string()),
            pa.field("dt", pa.timestamp("s")),
            pa.field("image_blob", pa.binary()),
            pa.field("thumb_blob", pa.binary()),
            # --- Hurricane event columns (populated after ingestion) ---
            pa.field("hurricane_present", pa.bool_()),
            pa.field("n_storms", pa.int32()),
            pa.field("max_wind_kts", pa.float32()),
            pa.field("max_category", pa.int32()),
            pa.field("storm_ids", pa.string()),
            pa.field("storm_lats", pa.string()),
            pa.field("storm_lons", pa.string()),
        ],
        metadata=arrow_metadata,
    )

    if IMG_RAW_TBL_NAME in db.list_tables():
        db.drop_table(IMG_RAW_TBL_NAME)

    table = db.create_table(IMG_RAW_TBL_NAME, schema=schema)

    print(f"Table {IMG_RAW_TBL_NAME} created.")
    return (table,)


@app.cell
def _():
    # # Config Table (Global Metadata)

    # config_data = [
    #     {"key": "db_name",          "value": DB_NAME},
    #     {"key": "created_at",       "value": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")},
    #     {"key": "author",           "value": "cherukuru"},
    #     {"key": "project",          "value": "era5 data"},
    #     {"key": "image_width",      "value": "224"},
    #     {"key": "image_height",     "value": "224"},
    #     {"key": "thumb_width",      "value": "64"},
    #     {"key": "height",           "value": "64"},
    #     {"key": "tbl_img_raw",      "value": IMG_RAW_TBL_NAME}
    # ]

    # # Create the config table
    # # We use overwrite mode to ensure we don't have stale configs if re-running
    # if METADATA_TBL_NAME in db.table_names():
    #     db.drop_table(METADATA_TBL_NAME)

    # # Note: We let LanceDB infer the simple schema (key: str, value: str) automatically
    # # by passing the list of dicts directly.
    # config_table = db.create_table(METADATA_TBL_NAME, data=config_data)
    # print(f"Table {METADATA_TBL_NAME} created with global metadata.")
    return


@app.cell
def _():
    # ## Serial workflow
    # ingest_images_to_table(
    #     table,
    #     image_dir=image_dir,
    #     width=WIDTH,
    #     height=HEIGHT,
    #     dt_format=DT_FORMAT,
    #     thumb_size=THUMB_RESOLUTION,
    #     batch_size=BATCH_SIZE
    # )
    return


@app.cell
def _(
    BATCH_SIZE,
    DT_FORMAT,
    HEIGHT,
    INGEST_RESOLUTION,
    THUMB_RESOLUTION,
    WIDTH,
    WORKERS,
    datetime,
    image_dir,
    ingest_images_to_table,
    table,
):
    from pathlib import Path as _Path

    import pandas as _pd

    from helpers.parallel_ingest_images import list_images_flat

    # Build file list, optionally filtered to a coarser temporal resolution.
    # For example, INGEST_RESOLUTION="3h" keeps only files whose timestamp
    # aligns to a 3-hour boundary (00, 03, 06, ... UTC), skipping the rest.
    _all_files = list_images_flat(image_dir)
    if INGEST_RESOLUTION is not None and DT_FORMAT is not None:
        _files = [
            p for p in _all_files
            if (lambda dt: dt == dt.floor(INGEST_RESOLUTION))(
                _pd.Timestamp(datetime.strptime(p.name, DT_FORMAT))
            )
        ]
        print(f"Resolution filter '{INGEST_RESOLUTION}': {len(_files)}/{len(_all_files)} files selected")
    else:
        _files = _all_files
        print(f"No resolution filter: ingesting all {len(_files)} files")

    ingest_images_to_table(
        table_obj=table,
        image_dir=image_dir,
        width=WIDTH,
        height=HEIGHT,
        dt_format=DT_FORMAT,
        thumb_size=THUMB_RESOLUTION,
        batch_size=BATCH_SIZE,
        workers=WORKERS,
        max_in_flight=WORKERS * 16,
        files=_files,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1b. Enrich with Hurricane Event Labels
    Load IBTrACS storm tracks, match to each image date, and fill
    the hurricane columns via `merge_insert`.
    """)
    return


@app.cell
def _(PROJECT_ROOT, TEMPORAL_END, TEMPORAL_START, table):
    from helpers.hurricane_metadata import (
        build_hurricane_lookup,
        enrich_image_rows,
        filter_to_domain,
        infer_temporal_resolution,
        load_ibtracs,
    )

    _ibtracs_path = PROJECT_ROOT / "data" / "ibtracs" / "ibtracs.NA.list.v04r01.csv"

    # Load IBTrACS North Atlantic, filtered to our temporal range
    _ibtracs_raw = load_ibtracs(
        _ibtracs_path,
        start_date=TEMPORAL_START,
        end_date=TEMPORAL_END,
    )

    # Keep only observations inside the ERA5 spatial domain
    _ibtracs_domain = filter_to_domain(
        _ibtracs_raw,
        lat_min=15.0,
        lat_max=35.0,
        lon_min=260.0,
        lon_max=330.0,
    )

    # Auto-detect temporal resolution from the image table
    _scanner = table.to_lance().scanner(columns=["id", "dt"])
    id_dt = _scanner.to_table().to_pandas()
    freq = infer_temporal_resolution(id_dt["dt"])

    # Build IBTrACS lookup at detected resolution
    hurricane_lookup = build_hurricane_lookup(_ibtracs_domain, freq=freq)

    print(
        f"IBTrACS: {len(_ibtracs_raw)} obs loaded, "
        f"{len(_ibtracs_domain)} in domain, "
        f"{len(hurricane_lookup)} unique {freq} buckets with storms"
    )
    return enrich_image_rows, freq, hurricane_lookup, id_dt


@app.cell
def _(enrich_image_rows, freq, hurricane_lookup, id_dt, json, table):
    import pyarrow as _pa

    # Compute hurricane columns for every image row at detected resolution
    _hurricane_df = enrich_image_rows(id_dt["dt"], hurricane_lookup, freq=freq)
    _hurricane_df["id"] = id_dt["id"].values

    # Build Arrow table with types matching the LanceDB table schema
    # (pandas defaults to int64/double/large_string, but schema has int32/float/string)
    _merge_fields = ["id", "hurricane_present", "n_storms", "max_wind_kts",
                     "max_category", "storm_ids", "storm_lats", "storm_lons"]
    _merge_schema = _pa.schema([table.schema.field(f) for f in _merge_fields])
    _merge_tbl = _pa.Table.from_pandas(_hurricane_df, schema=_merge_schema, preserve_index=False)

    # Merge hurricane columns into the table, matched on "id"
    table.merge_insert("id").when_matched_update_all().execute(_merge_tbl)

    # Update row_count, temporal interval, and matching logic in schema metadata
    _row_count = table.count_rows()
    _raw_meta = table.schema.metadata or {}
    _ds_info = json.loads(_raw_meta.get(b"dataset_info", b"{}"))
    _ds_info["row_count"] = _row_count
    _ds_info["temporal_extent"]["interval"] = freq
    _ds_info["hurricane_matching"]["logic"] = (
        f"Per image timestamp (freq={freq}), find nearest IBTrACS obs "
        f"in spatial domain; pick obs closest to bucket center per storm"
    )
    _new_meta = {"dataset_info": json.dumps(_ds_info), "version": "1.0"}
    table.to_lance().replace_schema_metadata(_new_meta)

    _n_with = int(_hurricane_df["hurricane_present"].sum())
    print(
        f"Hurricane enrichment complete ({freq}): "
        f"{_n_with}/{len(_hurricane_df)} timesteps with storms, "
        f"row_count metadata set to {_row_count}"
    )
    return


@app.cell
def _(table):
    table.schema
    return


@app.cell
def _(json, table):
    json.loads(table.schema.metadata[b"dataset_info"])
    return


@app.cell
def _(table):
    import io
    import os

    from PIL import Image

    table.search().select(["id"]).limit(10).to_pandas()

    row = table.search().limit(1000).to_pandas().iloc[999]
    img = Image.open(io.BytesIO(row["image_blob"]))

    print(img.size)
    img
    return os, row


@app.cell
def _(row):
    row["id"]
    return


@app.cell
def _(Path, db_dir, os):
    def dir_size_bytes(path: Path) -> int:
        total = 0
        for root, _, files in os.walk(path):
            for f in files:
                total += (Path(root) / f).stat().st_size
        return total


    table_path = db_dir / "era5_hrly_2016_2018_images.lance"


    # size_bytes = dir_size_bytes("/glade/work/ncheruku/research/bams-ai-data-exploration/data/lancedb/shared_source/era5/dinov3_image_embeddings.lance")

    size_bytes = dir_size_bytes(table_path)


    print(f"{size_bytes / 1024**2:.2f} MB")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
