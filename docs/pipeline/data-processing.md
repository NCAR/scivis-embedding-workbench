# ERA5 Batch Processing

**Script:** `notebooks/00_data_processing/e5_batch.py`

Runs on NCAR Casper HPC. Reads ERA5 NetCDF files and produces a flat folder of JPEG composite images — one image per timestep — ready for ingestion into LanceDB.

---

## What It Produces

Each output JPEG encodes three atmospheric fields as RGB channels:

| Channel | Variable | Range | Encoding |
|---|---|---|---|
| Red | MSL pressure anomaly | −20 to +20 hPa | Inverted (low pressure = bright red) |
| Green | 10m wind speed | 0 to 35 m/s | Linear |
| Blue | Total column water vapor | 20 to 70 kg/m² | Square-root scaled |

Output images are 896×256 px (7:2 aspect ratio, Plate Carrée projection) covering the North Atlantic domain (15–35°N, 100–30°W).

---

## Key CLI Arguments

```bash
uv run python e5_batch.py \
  --data-dir /path/to/era5/netcdf \
  --out-root /path/outside/repo/era5_jpegs \
  --start 2016-01-01 \
  --end 2018-12-31 \
  --workers 8
```

| Argument | Description |
|---|---|
| `--data-dir` | Directory containing ERA5 NetCDF files |
| `--out-root` | Output folder for JPEGs (defaults to one level above repo root) |
| `--start` / `--end` | Date range to process |
| `--workers` | Number of parallel JPEG encode workers |

!!! note "Output location"
    The default `--out-root` resolves to one level above the repo root at runtime, so generated images never land inside the repository.

---

## Processing Pipeline

1. Load ERA5 variables via Xarray + Dask (lazy, chunked)
2. Compute the three composite channels per timestep
3. Batch-resize frames on GPU using PyTorch (one kernel call per batch)
4. Encode and write JPEGs in parallel via a `ProcessPoolExecutor`
