# SciVis Embedding Workbench

A research toolkit for generating, exploring, and analyzing patch-level embeddings from scientific imagery — built around ERA5 weather composites and hurricane detection.

---

## What It Does

The workbench provides an end-to-end pipeline that takes raw ERA5 atmospheric reanalysis data and produces searchable, analyzable embedding representations:

```
ERA5 NetCDF files
      ↓  e5_batch.py
  JPEG composites  (R=pressure anomaly · G=wind speed · B=water vapor)
      ↓  create_image_database.py
  LanceDB source table  (images + IBTrACS hurricane labels)
      ↓  generate_dinov3_embeddings.py
  Image & patch embeddings  (DINOv2 / OpenCLIP)
      ↓  app.py
  Interactive dashboard  (map · timeline · similarity search · UMAP)
```

---

## Key Features

- **Flexible temporal resolution** — ingest daily, 3-hourly, 6-hourly or hourly images; temporal resolution is auto-detected and propagated through the entire pipeline
- **Hurricane labels** — IBTrACS storm tracks matched to each image at the ingested temporal resolution (nearest-observation logic)
- **Multiple embedding models** — DINOv2 (square and rectangular), OpenCLIP; easily extensible via the model registry
- **Patch-level analysis** — attention maps and per-patch embeddings alongside image-level vectors
- **HPC-ready** — batch processing scripts designed for NCAR Casper; Marimo notebooks accessible via SSH tunnel

---

## Data Sources

This project uses the following openly licensed datasets:

| Dataset | Provider | License |
|---|---|---|
| ERA5 atmospheric reanalysis | ECMWF / Copernicus Climate Change Service | [Copernicus License](https://cds.climate.copernicus.eu/api/v2/terms/static/licence-to-use-copernicus-products.pdf) |
| IBTrACS storm tracks v04r01 | NOAA / NCEI | Public domain (US Government) |

!!! note "Attribution"
    ERA5 data: "Contains modified Copernicus Climate Change Service information 2016–2018. Neither the European Commission nor ECMWF is responsible for any use that may be made of the Copernicus information or data it contains."

    IBTrACS data: Knapp, K.R. et al. (2010). The International Best Track Archive for Climate Stewardship (IBTrACS). *Bulletin of the American Meteorological Society*, 91, 363–376.

---

## Models Used

| Model | Source | License |
|---|---|---|
| DINOv2 | Meta AI | Apache 2.0 |
| OpenCLIP | LAION / ML Foundations | MIT / BSD |
