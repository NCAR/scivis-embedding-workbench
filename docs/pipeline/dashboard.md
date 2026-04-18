# Dashboard

**Script:** `notebooks/03-dashboard-app/app.py`

An interactive Marimo application for exploring embedding experiments. Loads an experiment from the experiments database, joins embeddings back to the source image table, and provides several analysis views.

---

## Setup

Point the dashboard at your experiments database path and select an experiment from the dropdown. The app reads all configuration (source table path, model info, dimensions) from the experiment's config table automatically.

---

## Views

### Geographic Extent Map
A Cartopy map showing the ERA5 spatial domain (North Atlantic, 15–35°N, 100–30°W) with the patch grid overlaid. Illustrates the spatial resolution at which patch embeddings were computed.

### Hurricane Timeline
A Plotly time-series chart showing `max_wind_kts` and `hurricane_present` over the dataset date range. Sourced directly from the IBTrACS labels stored in the source image table.

### Patch Similarity Search
Select any image and patch to retrieve the most similar patches across the full dataset using vector search against `patch_embeddings`. Results are displayed as a grid of thumbnails.

### UMAP / PCA Scatter
Dimensionality reduction of image embeddings to 2D, colored by hurricane category or date. GPU-accelerated via cuML when available.

---

## Theme

The app follows the system dark/light mode preference by default (controlled via `.marimo.toml` at the repo root). The map theme toggle in the header independently controls plot and map color schemes.

---

## Running Locally

```bash
uv run marimo edit notebooks/03-dashboard-app/app.py
```

For remote access on NCAR Casper, see [Getting Started](../getting-started.md#running-on-ncar-casper).
