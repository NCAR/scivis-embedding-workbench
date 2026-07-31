"""Build a synthetic projection table for developing the latent-space explorer.

The real `x`/`y`/`cluster` columns will come from UMAP + HDBSCAN later. Everything
else in the schema — identities, timestamps, patch geometry, storm context — is
read from the actual experiment, so the table has the real row count, the real
join cardinality and real value distributions. Swapping in genuine UMAP output
later changes four columns and nothing else.

The synthetic coordinates are not noise: clusters are derived from latitude band,
season and storm presence, so colouring the scatter by those columns shows
structure. That makes the table a useful fixture for the viewer's colour-by paths
rather than a uniform blob that hides bugs.

    uv run python notebooks/05-latent-space-exploration/helpers/make_synthetic_projection.py

Writes a new table alongside `config`, `image_embeddings` and `patch_embeddings`
in the experiment directory. Only that one table is created or replaced; the
existing tables are opened read-only and never modified.

The table name carries `synthetic` so it cannot be mistaken for real projection
output sitting in the same directory, and the schema metadata says so too. When
real UMAP output lands, write it as `umap_patch_001` next to this one.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import lancedb
import numpy as np
import pandas as pd
import pyarrow as pa

SRC_EXPERIMENT = Path(
    "/Users/ncheruku/Documents/Work/sample_data/data/lancedb/experiments/era5/dinov3_24h"
)
SRC_IMAGES = Path(
    "/Users/ncheruku/Documents/Work/sample_data/data/lancedb/shared_source/"
    "era5_hrly_2016_2018_images"
)
# Same directory as the embeddings: the projection is part of the experiment.
OUT_EXPERIMENT = SRC_EXPERIMENT

RUN_ID = "synthetic"
TABLE = f"umap_patch_{RUN_ID}"
SEED = 42

# Patch grid and extent, from the source config / dataset_info. Kept as constants
# rather than re-read so the geometry used here is visible in one place.
GRID_H, GRID_W = 16, 56
LAT_MIN, LAT_MAX = 15.0, 35.0
LON_MIN, LON_MAX = 260.0, 330.0

N_CLUSTERS = 24
NOISE_FRACTION = 0.08


def load_identities(db) -> pd.DataFrame:
    """Read only the id columns from patch_embeddings — never the 768-d vectors."""
    tbl = db.open_table("patch_embeddings")
    return (
        tbl.search()
        .select(["image_id", "patch_index"])
        .limit(tbl.count_rows())
        .to_pandas()
        .sort_values(["image_id", "patch_index"], ignore_index=True)
    )


def load_image_metadata() -> pd.DataFrame:
    """Timestamp and storm context, one row per image. No blobs are touched."""
    tbl = lancedb.connect(SRC_IMAGES).open_table("images")
    df = (
        tbl.search()
        .select(
            [
                "id",
                "dt",
                "hurricane_present",
                "n_storms",
                "max_wind_kts",
                "max_category",
            ]
        )
        .limit(tbl.count_rows())
        .to_pandas()
    )
    return df.rename(columns={"id": "image_id"})


def add_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """Patch index -> grid cell -> centre lat/lon.

    patch_index is row-major from the top-left: index 0..55 is the top row of the
    16x56 grid, 56..111 the next row down. Confirmed against the embedding script,
    which reshapes the flat 896-token attention map with `attn.reshape(16, 56)` —
    numpy's default C order — so token order and this mapping agree.
    """
    idx = df["patch_index"].to_numpy()
    row, col = idx // GRID_W, idx % GRID_W
    df["patch_row"] = row.astype(np.int8)
    df["patch_col"] = col.astype(np.int8)
    df["lat"] = (LAT_MAX - (row + 0.5) * (LAT_MAX - LAT_MIN) / GRID_H).astype(np.float32)
    lon = LON_MIN + (col + 0.5) * (LON_MAX - LON_MIN) / GRID_W
    df["lon"] = (lon - 360.0).astype(np.float32)  # 260..330 -> -100..-30
    return df


def add_time_parts(df: pd.DataFrame) -> pd.DataFrame:
    dt = pd.DatetimeIndex(df["dt"])
    df["year"] = dt.year.astype(np.int16)
    df["dayofyear"] = dt.dayofyear.astype(np.int16)
    df["month"] = dt.month.astype(np.int8)
    df["day"] = dt.day.astype(np.int8)
    df["hour"] = dt.hour.astype(np.int8)
    return df


def synth_clusters(df: pd.DataFrame, rng: np.random.Generator) -> np.ndarray:
    """Assign clusters from real covariates so colour-by shows real structure.

    Latitude band x season x storm presence, then a slice re-labelled as noise.
    """
    lat_band = np.digitize(df["lat"].to_numpy(), np.linspace(LAT_MIN, LAT_MAX, 5)[1:-1])
    season = (df["month"].to_numpy() % 12) // 3
    storm = df["hurricane_present"].to_numpy().astype(int)

    cluster = (lat_band * 6 + season * 1 + storm * 3) % N_CLUSTERS
    # Blur the boundaries so clusters are not perfectly separable.
    flip = rng.random(len(df)) < 0.15
    cluster[flip] = rng.integers(0, N_CLUSTERS, flip.sum())

    noise = rng.random(len(df)) < NOISE_FRACTION
    cluster = cluster.astype(np.int16)
    cluster[noise] = -1
    return cluster


def synth_coordinates(
    cluster: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gaussian blobs on a ring, plus a diffuse background for the noise label.

    Shaped to look like a UMAP result: tight cores, uneven cluster sizes, and
    noise points scattered between the blobs rather than in a neat halo.
    """
    n = len(cluster)
    angles = rng.permutation(N_CLUSTERS) * (2 * np.pi / N_CLUSTERS)
    radii = 6.0 + rng.random(N_CLUSTERS) * 4.0
    centers = np.stack([radii * np.cos(angles), radii * np.sin(angles)], axis=1)
    spreads = 0.35 + rng.random(N_CLUSTERS) * 0.9

    x = np.empty(n, dtype=np.float32)
    y = np.empty(n, dtype=np.float32)
    prob = np.zeros(n, dtype=np.float32)

    for k in range(N_CLUSTERS):
        m = cluster == k
        if not m.any():
            continue
        pts = rng.normal(centers[k], spreads[k], size=(int(m.sum()), 2))
        x[m], y[m] = pts[:, 0], pts[:, 1]
        # Membership falls off with distance from the cluster centre.
        d = np.linalg.norm(pts - centers[k], axis=1)
        prob[m] = np.clip(1.0 - d / (4.0 * spreads[k]), 0.02, 1.0)

    m = cluster == -1
    if m.any():
        pts = rng.normal(0.0, 7.0, size=(int(m.sum()), 2))
        x[m], y[m] = pts[:, 0], pts[:, 1]
        prob[m] = 0.0

    return x, y, prob


def main() -> None:
    rng = np.random.default_rng(SEED)

    print(f"reading identities from {SRC_EXPERIMENT.name} ...")
    df = load_identities(lancedb.connect(SRC_EXPERIMENT))
    print(f"  {len(df):,} patches")

    meta = load_image_metadata()
    n_before = len(df)
    df = df.merge(meta, on="image_id", how="left", validate="many_to_one")
    assert len(df) == n_before, "merge changed the row count"
    missing = int(df["dt"].isna().sum())
    assert missing == 0, f"{missing:,} patches have no matching image row"
    print(f"  joined {df['image_id'].nunique():,} images, no unmatched rows")

    df = add_geometry(df)
    df = add_time_parts(df)

    df["cluster"] = synth_clusters(df, rng)
    df["x"], df["y"], df["cluster_prob"] = synth_coordinates(df["cluster"].to_numpy(), rng)

    df["n_storms"] = df["n_storms"].astype(np.int8)
    df["max_category"] = df["max_category"].astype(np.int8)
    df["max_wind_kts"] = df["max_wind_kts"].astype(np.float32)
    df["patch_index"] = df["patch_index"].astype(np.int32)

    df = df[
        [
            "image_id",
            "patch_index",
            "x",
            "y",
            "cluster",
            "cluster_prob",
            "dt",
            "year",
            "dayofyear",
            "month",
            "day",
            "hour",
            "lat",
            "lon",
            "patch_row",
            "patch_col",
            "hurricane_present",
            "n_storms",
            "max_wind_kts",
            "max_category",
        ]
    ].sort_values(["image_id", "patch_index"], ignore_index=True)

    OUT_EXPERIMENT.mkdir(parents=True, exist_ok=True)
    out_db = lancedb.connect(OUT_EXPERIMENT)
    # Rebuild from scratch each run: this is a build artifact, not a store. The
    # guard matters because this now writes into the real experiment directory —
    # a careless edit to TABLE would otherwise delete source data.
    protected = {"config", "image_embeddings", "patch_embeddings"}
    assert TABLE not in protected, f"refusing to overwrite source table {TABLE!r}"
    shutil.rmtree(OUT_EXPERIMENT / f"{TABLE}.lance", ignore_errors=True)

    # Column roles travel with the table so the viewer can build its colour-by
    # dropdown from the data instead of a hardcoded list.
    meta_json = {
        "run_id": RUN_ID,
        "synthetic": "true",
        "note": "x/y/cluster/cluster_prob are SYNTHETIC. All other columns are real.",
        "created_at": datetime.now(UTC).isoformat(),
        "source_experiment": str(SRC_EXPERIMENT),
        "source_table": "patch_embeddings",
        "color_by": json.dumps(
            {
                "categorical": [
                    "cluster",
                    "max_category",
                    "hurricane_present",
                    "month",
                    "year",
                    "hour",
                ],
                "continuous": [
                    "cluster_prob",
                    "lat",
                    "lon",
                    "max_wind_kts",
                    "dayofyear",
                ],
            }
        ),
        "umap": json.dumps({"backend": "synthetic", "n_components": 2}),
        "hdbscan": json.dumps(
            {"backend": "synthetic", "n_clusters": N_CLUSTERS,
             "noise_fraction": NOISE_FRACTION}
        ),
    }

    # Metadata has to be attached to the Arrow schema before the table is
    # created — LanceTable has no setter for it afterwards.
    arrow_tbl = pa.Table.from_pandas(df, preserve_index=False)
    arrow_tbl = arrow_tbl.replace_schema_metadata(
        {k: str(v) for k, v in meta_json.items()}
    )
    out_db.create_table(TABLE, arrow_tbl)

    size_mb = sum(
        f.stat().st_size for f in (OUT_EXPERIMENT / f"{TABLE}.lance").rglob("*") if f.is_file()
    ) / 1e6
    print(f"\nwrote {OUT_EXPERIMENT / (TABLE + '.lance')}")
    print(f"  {len(df):,} rows x {len(df.columns)} cols, {size_mb:.1f} MB on disk")
    print(f"  clusters: {df['cluster'].nunique() - 1} + noise "
          f"({(df['cluster'] == -1).mean():.1%})")


if __name__ == "__main__":
    main()
