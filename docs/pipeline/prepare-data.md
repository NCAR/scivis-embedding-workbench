# Image Database

**Script:** `notebooks/01-prepare-data/create_image_database.py`

Ingests JPEG composites into a LanceDB source table and enriches each row with IBTrACS hurricane labels matched at the appropriate temporal resolution.

---

## Database Layout

Each project lives in its own subfolder inside `shared_source/`, keeping the dataset identity in the folder and the table name generic and renameable:

```
lancedb/shared_source/
  era5_sample_images/     ← SOURCE_PROJECT (LanceDB database)
    images.lance          ← IMG_RAW_TBL_NAME = "images"
```

---

## Key Configuration

| Variable | Default | Description |
|---|---|---|
| `SOURCE_PROJECT` | `"era5_sample_images"` | Project folder name inside `shared_source/` |
| `IMG_RAW_TBL_NAME` | `"images"` | Table name (rename freely) |
| `DT_FORMAT` | `"%Y%m%d_%H_rgb.jpeg"` | strptime pattern for parsing timestamps from filenames |
| `INGEST_RESOLUTION` | `"3h"` | Subsample the source folder to this frequency (`None` = ingest all) |
| `TEMPORAL_START` / `TEMPORAL_END` | `"2016-01-01"` / `"2018-12-31"` | Date range for IBTrACS hurricane label loading |

---

## Temporal Subsampling

If the source folder contains finer-grained data than needed, `INGEST_RESOLUTION` filters the file list before any image is decoded — skipped files incur zero cost:

```python
INGEST_RESOLUTION = "3h"   # keep 00Z, 03Z, 06Z, ... from an hourly folder
INGEST_RESOLUTION = "6h"   # keep 00Z, 06Z, 12Z, 18Z
INGEST_RESOLUTION = None    # ingest everything
```

---

## Hurricane Enrichment

After ingestion, each image row is enriched with storm label columns sourced from IBTrACS v04r01 (North Atlantic basin):

| Column | Type | Description |
|---|---|---|
| `hurricane_present` | bool | Any storm in the spatial domain at this timestep |
| `n_storms` | int | Number of distinct storms |
| `max_wind_kts` | float | Maximum WMO wind speed (knots) |
| `max_category` | int | Saffir-Simpson code (−1=TD, 0=TS, 1–5=Cat) |
| `storm_ids` | string | Comma-separated IBTrACS SIDs |
| `storm_lats` / `storm_lons` | string | Positions of representative observations |

The temporal resolution is **auto-detected** from the `dt` column of the ingested table. For each image timestamp, the nearest IBTrACS observation (closest to the bucket center) is selected per storm.

!!! info "Data attribution"
    IBTrACS data is provided by NOAA/NCEI and is in the public domain. See the [Home](../index.md) page for full attribution.
