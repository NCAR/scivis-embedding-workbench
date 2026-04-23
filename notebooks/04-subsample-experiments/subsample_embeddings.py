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
    # Subsample Embeddings by Temporal Resolution

    Given a **1-hour base experiment** (image + patch embeddings for every hourly
    timestep), this notebook produces one new experiment per target resolution by
    filtering the existing embeddings — **no model re-run required**.

    Each output experiment is a self-contained LanceDB subfolder with:
    - `config` table (metadata + provenance linking back to the 1hr base)
    - `image_embeddings` table (one row per retained image)
    - `patch_embeddings` table (all patches for retained images)
    - IVF-PQ vector index on `patch_embeddings.embedding`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Imports")
    return


@app.cell
def _():
    import math
    import sys
    from datetime import UTC, datetime
    from pathlib import Path

    import lancedb
    import pandas as pd
    import pyarrow as pa
    import pyarrow.compute as pc
    from rich.console import Console
    from rich.table import Table

    console = Console()
    return UTC, Console, Table, console, datetime, lancedb, math, pa, pc, pd, sys, Path


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Configuration

    Set `BASE_EXPERIMENT` to the name of the 1-hour experiment subfolder inside
    `DB_URI` (e.g. `"dinov3_rect_1h"`). Set `RESOLUTIONS` to the cadences you
    want to generate.
    """)
    return


@app.cell
def _(Path):
    # ── Paths ─────────────────────────────────────────────────────────────────
    # Local Mac (for reference):
    # PROJECT_ROOT = Path("/Users/ncheruku/Documents/Work/sample_data")

    # NCAR Casper
    PROJECT_ROOT = Path("/glade/work/ncheruku/research/sample_data")

    # Root folder that holds all experiment subfolders (one per model/run)
    DB_URI = PROJECT_ROOT / "data" / "lancedb" / "experiments" / "era5"

    # Name of the 1-hour base experiment subfolder inside DB_URI
    BASE_EXPERIMENT = "dinov3_rect_1h"

    # ── Resolutions to generate ───────────────────────────────────────────────
    # Each entry is a (pandas_freq_string, output_project_name) pair.
    # pandas freq strings: "1h", "3h", "6h", "12h", "24h"
    # "24h" aligns to midnight UTC (00:00); adjust DAILY_HOUR below if needed.
    RESOLUTIONS = [
        ("3h",  "dinov3_rect_3h"),
        ("6h",  "dinov3_rect_6h"),
        ("12h", "dinov3_rect_12h"),
        ("24h", "dinov3_rect_24h"),
    ]

    # For 24h subsampling: which UTC hour to retain (0 = midnight, 12 = noon).
    DAILY_HOUR = 0

    # ── Index parameters ──────────────────────────────────────────────────────
    # num_sub_vectors must divide the embedding dimension (768).
    # 64 sub-vectors → 768/64 = 12 bytes per vector (same as base experiment).
    NUM_SUB_VECTORS = 64

    # IVF-PQ num_partitions auto-scaled per resolution (see cell below).
    # Override here if you want a fixed value for all resolutions.
    NUM_PARTITIONS_OVERRIDE = None   # e.g. 512, or None to auto-scale

    return (
        BASE_EXPERIMENT,
        DAILY_HOUR,
        DB_URI,
        NUM_PARTITIONS_OVERRIDE,
        NUM_SUB_VECTORS,
        PROJECT_ROOT,
        RESOLUTIONS,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Load base experiment")
    return


@app.cell
def _(BASE_EXPERIMENT, DB_URI, console, lancedb):
    _base_uri = str(DB_URI / BASE_EXPERIMENT)
    _base_db  = lancedb.connect(_base_uri)

    # Load config dict from base experiment
    _cfg_df = _base_db.open_table("config").to_pandas()
    base_config = dict(zip(_cfg_df["key"], _cfg_df["value"]))

    console.print(f"[green]Base experiment:[/green] {_base_uri}")
    console.print(f"  model      : {base_config.get('model_name', '?')}")
    console.print(f"  resolution : {base_config.get('temporal_resolution', '1h')}")
    console.print(f"  images     : {base_config.get('processed_image_count', '?')}")

    base_img_emb_tbl   = _base_db.open_table("image_embeddings")
    base_patch_emb_tbl = _base_db.open_table("patch_embeddings")
    return base_config, base_img_emb_tbl, base_patch_emb_tbl


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Load source table — build id → dt map")
    return


@app.cell
def _(base_config, console, lancedb, pc, pd, DB_URI):
    # Resolve source DB path: try stored source_path relative to DB_URI first,
    # then fall back to raw_db_uri (absolute GLADE path).
    from pathlib import Path as _Path

    _source_path = base_config.get("source_path", "")
    _raw_uri     = base_config.get("raw_db_uri", "")
    _source_tbl  = base_config.get("source", "images")

    _resolved = None
    if _source_path:
        # Walk up from DB_URI to find where source_path exists
        _candidate = _Path(DB_URI)
        for _ in range(10):
            _candidate = _candidate.parent
            _try = _candidate / _source_path
            if _try.exists():
                _resolved = str(_try)
                break
    if _resolved is None and _raw_uri and _Path(_raw_uri).exists():
        _resolved = _raw_uri

    if _resolved is None:
        raise FileNotFoundError(
            f"Cannot find source DB. source_path={_source_path!r}, "
            f"raw_db_uri={_raw_uri!r}"
        )

    _src_db  = lancedb.connect(_resolved)
    _src_tbl = _src_db.open_table(_source_tbl)

    # Fetch id + dt for every row (no blobs — lightweight)
    _dt_df = (
        _src_tbl.to_lance()
        .scanner(columns=["id", "dt"])
        .to_table()
        .to_pandas()
    )
    _dt_df["dt"] = pd.to_datetime(_dt_df["dt"], utc=True)

    id_to_dt = dict(zip(_dt_df["id"], _dt_df["dt"]))

    console.print(f"\n[green]Source table:[/green] {_resolved} → {_source_tbl}")
    console.print(f"  rows : {len(_dt_df):,}")
    console.print(f"  dt range : {_dt_df['dt'].min()} → {_dt_df['dt'].max()}")
    return id_to_dt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Helper — compute aligned image ids for a given resolution

    A timestep is *aligned* to a cadence if flooring its datetime to that cadence
    leaves it unchanged:  `dt.floor(freq) == dt`.

    For `24h` we additionally filter to the configured `DAILY_HOUR` so that
    "daily" unambiguously means one specific UTC hour (default midnight).
    """)
    return


@app.cell
def _(DAILY_HOUR, pd):
    def aligned_ids(id_to_dt: dict, freq: str) -> list[str]:
        """Return image ids whose timestamp aligns to *freq* boundaries."""
        result = []
        for img_id, dt in id_to_dt.items():
            if dt.floor(freq) == dt:
                if freq in ("24h", "D"):
                    if dt.hour != DAILY_HOUR:
                        continue
                result.append(img_id)
        return result
    return (aligned_ids,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Helper — choose IVF-PQ num_partitions")
    return


@app.cell
def _(NUM_PARTITIONS_OVERRIDE, math):
    def choose_num_partitions(n_patches: int) -> int:
        """Scale num_partitions ≈ sqrt(n_patches), clamped to [64, 2048]."""
        if NUM_PARTITIONS_OVERRIDE is not None:
            return NUM_PARTITIONS_OVERRIDE
        return max(64, min(2048, int(2 ** round(math.log2(math.sqrt(n_patches))))))
    return (choose_num_partitions,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Generate subsampled experiments

    For each target resolution:
    1. Compute the aligned image ids
    2. Filter `image_embeddings` and `patch_embeddings` via Lance scanner
    3. Write to a new experiment subfolder
    4. Copy + extend config (preserves full provenance chain)
    5. Build IVF-PQ index on `patch_embeddings`
    """)
    return


@app.cell
def _(
    BASE_EXPERIMENT,
    DB_URI,
    NUM_SUB_VECTORS,
    RESOLUTIONS,
    UTC,
    aligned_ids,
    base_config,
    base_img_emb_tbl,
    base_patch_emb_tbl,
    choose_num_partitions,
    console,
    datetime,
    id_to_dt,
    lancedb,
    pa,
    pc,
    Table,
):
    summary_rows = []

    for _freq, _project in RESOLUTIONS:
        console.rule(f"[bold cyan]{_project}[/bold cyan]  ({_freq})")

        # ── 1. Aligned image ids ───────────────────────────────────────────
        _img_ids = aligned_ids(id_to_dt, _freq)
        _img_ids_set = set(_img_ids)
        console.print(f"  aligned images : {len(_img_ids):,}")

        # ── 2. Filter image_embeddings ─────────────────────────────────────
        console.print("  filtering image_embeddings …")
        _img_arrow = (
            base_img_emb_tbl.to_lance()
            .scanner(filter=pc.field("image_id").isin(_img_ids))
            .to_table()
        )
        n_images = _img_arrow.num_rows
        console.print(f"    → {n_images:,} rows")

        # ── 3. Filter patch_embeddings ─────────────────────────────────────
        console.print("  filtering patch_embeddings …")
        _patch_arrow = (
            base_patch_emb_tbl.to_lance()
            .scanner(filter=pc.field("image_id").isin(_img_ids))
            .to_table()
        )
        n_patches = _patch_arrow.num_rows
        console.print(f"    → {n_patches:,} rows  ({n_patches // max(n_images, 1)} patches/image)")

        # ── 4. Write to new experiment DB ──────────────────────────────────
        _out_uri = str(DB_URI / _project)
        _out_db  = lancedb.connect(_out_uri)
        console.print(f"  writing to {_out_uri} …")

        _out_db.create_table("image_embeddings", data=_img_arrow,   mode="overwrite")
        _out_db.create_table("patch_embeddings", data=_patch_arrow, mode="overwrite")

        # ── 5. Config table ────────────────────────────────────────────────
        # Start from the base config and layer subsampling provenance on top.
        _skip_keys = {"created_at", "processed_image_count", "elapsed_seconds",
                      "throughput_img_per_sec", "temporal_resolution"}
        _config_rows = [
            {"key": k, "value": v}
            for k, v in base_config.items()
            if k not in _skip_keys
        ]
        _config_rows += [
            {"key": "created_at",            "value": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")},
            {"key": "temporal_resolution",   "value": _freq},
            {"key": "processed_image_count", "value": str(n_images)},
            {"key": "processed_patch_count", "value": str(n_patches)},
            {"key": "subsampled_from",       "value": BASE_EXPERIMENT},
            {"key": "subsample_script",      "value": "notebooks/04-subsample-experiments/subsample_embeddings.py"},
        ]
        _out_db.create_table("config", data=_config_rows, mode="overwrite")

        # ── 6. IVF-PQ index ────────────────────────────────────────────────
        _n_parts = choose_num_partitions(n_patches)
        console.print(f"  building IVF-PQ index  (partitions={_n_parts}, sub_vectors={NUM_SUB_VECTORS}) …")
        _patch_tbl = _out_db.open_table("patch_embeddings")
        _patch_tbl.create_index(
            metric="cosine",
            index_type="IVF_PQ",
            num_partitions=_n_parts,
            num_sub_vectors=NUM_SUB_VECTORS,
            vector_column_name="embedding",
        )
        console.print(f"  [green]done[/green]")

        summary_rows.append({
            "resolution": _freq,
            "project":    _project,
            "images":     n_images,
            "patches":    n_patches,
            "partitions": _n_parts,
        })

    # ── Summary table ──────────────────────────────────────────────────────────
    console.rule("[bold]Summary[/bold]")
    _tbl = Table(show_header=True)
    _tbl.add_column("Resolution")
    _tbl.add_column("Project")
    _tbl.add_column("Images",   justify="right")
    _tbl.add_column("Patches",  justify="right")
    _tbl.add_column("IVF parts", justify="right")
    for _row in summary_rows:
        _tbl.add_row(
            _row["resolution"],
            _row["project"],
            f"{_row['images']:,}",
            f"{_row['patches']:,}",
            str(_row["partitions"]),
        )
    console.print(_tbl)
    return (summary_rows,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Verify

    Spot-check each output experiment: open the config and confirm row counts
    match expectations.
    """)
    return


@app.cell
def _(DB_URI, RESOLUTIONS, console, lancedb, Table):
    _tbl = Table(title="Verification", show_header=True)
    _tbl.add_column("Project")
    _tbl.add_column("Config temporal_resolution")
    _tbl.add_column("image_embeddings rows", justify="right")
    _tbl.add_column("patch_embeddings rows",  justify="right")

    for _freq, _project in RESOLUTIONS:
        _uri = str(DB_URI / _project)
        try:
            _db  = lancedb.connect(_uri)
            _cfg = dict(zip(*[_db.open_table("config").to_pandas()[c] for c in ["key", "value"]]))
            _n_img   = _db.open_table("image_embeddings").count_rows()
            _n_patch = _db.open_table("patch_embeddings").count_rows()
            _tbl.add_row(_project, _cfg.get("temporal_resolution", "?"),
                         f"{_n_img:,}", f"{_n_patch:,}")
        except Exception as _e:
            _tbl.add_row(_project, "ERROR", str(_e), "")

    console.print(_tbl)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
