import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Subsample Embeddings

    Generate **3 h / 6 h / 12 h / 24 h** experiments from an existing 1 h base
    experiment by filtering embedding tables to temporally-aligned timesteps.

    No model re-run is required — image and patch embeddings are copied directly
    from the base LanceDB tables. An IVF-PQ vector index is built on each output
    `patch_embeddings` table using partition counts scaled to the dataset size.
    """)
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
    return Path, Table, UTC, console, datetime, lancedb, math, pa, pc, pd, sys


@app.cell
def _(Path, sys):
    # ── Environment paths ─────────────────────────────────────────────────────
    # NCAR Casper
    PROJECT_ROOT = Path("/glade/work/ncheruku/research/sample_data")
    # Local Mac (uncomment to switch):
    # PROJECT_ROOT = Path("/Users/ncheruku/Documents/Work/sample_data")

    # Name of the 1 h base experiment subfolder inside DB_URI
    BASE_EXPERIMENT = "dinov3_1h"

    # LanceDB root holding all experiments
    DB_URI = PROJECT_ROOT / "data" / "lancedb" / "experiments" / "era5"

    # Temporal resolutions to generate: list of (pandas_freq_str, output_name)
    RESOLUTIONS = [
        ("3h",  "dinov3_3h"),
        ("6h",  "dinov3_6h"),
        ("12h", "dinov3_12h"),
        ("24h", "dinov3_24h"),
    ]

    # Expose the embedding_experiment helpers without a package install
    _helpers_dir = str(
        Path(__file__).parent.parent / "02-generate-embeddings" / "helpers"
    )
    if _helpers_dir not in sys.path:
        sys.path.insert(0, _helpers_dir)
    return BASE_EXPERIMENT, DB_URI, RESOLUTIONS


@app.cell
def _(BASE_EXPERIMENT, DB_URI, lancedb):
    from embedding_experiment import load_config

    _base_db_uri = str(DB_URI / BASE_EXPERIMENT)
    _base_db = lancedb.connect(_base_db_uri)

    base_config = load_config(_base_db_uri, "config")
    base_img_tbl = _base_db.open_table("image_embeddings")
    base_patch_tbl = _base_db.open_table("patch_embeddings")

    print(f"Base experiment  : {BASE_EXPERIMENT}")
    print(f"  image_embeddings : {base_img_tbl.count_rows():,} rows")
    print(f"  patch_embeddings : {base_patch_tbl.count_rows():,} rows")
    print(f"  config keys      : {len(base_config)}")
    return base_config, base_img_tbl, base_patch_tbl, load_config


@app.cell
def _(DB_URI, base_config, console, lancedb):
    def _resolve_source_path(db_path_str, source_path_str):
        """Walk up ancestor directories to resolve a relative (or absolute) source path."""
        from pathlib import Path as _Path
        source = _Path(source_path_str)
        if source.is_absolute():
            return str(source) if source.exists() else None
        candidate = _Path(db_path_str)
        for _ in range(12):
            resolved = candidate / source
            if resolved.exists():
                return str(resolved)
            parent = candidate.parent
            if parent == candidate:
                break
            candidate = parent
        return None

    _source_path = _resolve_source_path(
        str(DB_URI),
        base_config.get("source_path", ""),
    )
    if _source_path is None:
        raise FileNotFoundError(
            f"Cannot locate source table. "
            f"source_path in config = {base_config.get('source_path')!r}. "
            "Check that the path is reachable from DB_URI ancestors."
        )

    _src_db = lancedb.connect(_source_path)
    _src_tbl = _src_db.open_table(base_config.get("source", "images"))

    # Lightweight scan — id + dt only (no blobs)
    _dt_arrow = (
        _src_tbl.to_lance()
        .scanner(columns=["id", "dt"])
        .to_table()
    )
    _dt_df = _dt_arrow.to_pandas().set_index("id")
    id_to_dt = _dt_df["dt"].to_dict()

    console.print(f"Source table    : {_source_path}")
    console.print(f"Total timesteps : {len(id_to_dt):,}")
    return (id_to_dt,)


@app.cell
def _(id_to_dt, pd):
    def aligned_ids(id_to_dt, freq):
        """Return image_ids whose timestamp aligns exactly to the given cadence.

        A timestep is retained when flooring it to `freq` leaves it unchanged:
            pd.Timestamp(dt).floor(freq) == dt

        Parameters
        ----------
        id_to_dt : dict[str, Any]
            Mapping from image_id to raw datetime value (pandas Timestamp or str).
        freq : str
            Pandas offset alias, e.g. "3h", "6h", "12h", "24h".

        Returns
        -------
        list[str]
        """
        retained = []
        for img_id, dt in id_to_dt.items():
            try:
                ts = pd.Timestamp(dt)
                if ts == ts.floor(freq):
                    retained.append(img_id)
            except Exception:
                pass  # skip unparseable timestamps
        return retained

    # Quick sanity preview
    for _freq, _name in [("3h", None), ("6h", None), ("12h", None), ("24h", None)]:
        _n = len(aligned_ids(id_to_dt, _freq))
        print(f"  {_freq:>4s} → {_n:,} images  "
              f"(~1/{len(id_to_dt) // max(_n, 1)} of base)")
    return (aligned_ids,)


@app.cell
def _(
    BASE_EXPERIMENT,
    DB_URI,
    RESOLUTIONS,
    UTC,
    aligned_ids,
    base_config,
    base_img_tbl,
    base_patch_tbl,
    console,
    datetime,
    id_to_dt,
    lancedb,
    math,
    pa,
    pc,
):
    try:
        import torch as _torch
        _cuda_available = _torch.cuda.is_available()
    except ImportError:
        _cuda_available = False

    for _freq, _project in RESOLUTIONS:
        _out_path = DB_URI / _project

        # ── Guard: refuse to overwrite ───────────────────────────────────────
        if _out_path.exists():
            raise FileExistsError(
                f"Output experiment already exists: {_out_path}\n"
                "Delete the folder manually before re-generating."
            )

        console.rule(f"[bold cyan]{_project}[/bold cyan]  ({_freq})")

        # ── Temporal filter ──────────────────────────────────────────────────
        _img_ids = aligned_ids(id_to_dt, _freq)
        console.print(f"  Retained images : {len(_img_ids):,}")

        # ── Scan base tables (full pass, no ANN API) ─────────────────────────
        _img_arrow = (
            base_img_tbl.to_lance()
            .scanner(filter=pc.field("image_id").isin(_img_ids))
            .to_table()
        )
        _patch_arrow = (
            base_patch_tbl.to_lance()
            .scanner(filter=pc.field("image_id").isin(_img_ids))
            .to_table()
        )

        _n_images  = _img_arrow.num_rows
        _n_patches = _patch_arrow.num_rows
        console.print(f"  image_embeddings : {_n_images:,} rows")
        console.print(f"  patch_embeddings : {_n_patches:,} rows")

        # ── Create output tables ─────────────────────────────────────────────
        _out_db = lancedb.connect(str(_out_path))
        _out_db.create_table("image_embeddings", data=_img_arrow)
        _patch_out_tbl = _out_db.create_table("patch_embeddings", data=_patch_arrow)

        # ── Config: overwrite / add subsample-specific keys first, then copy
        #    remaining base keys verbatim ─────────────────────────────────────
        _overrides = {
            "created_at":            datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "processed_image_count": str(_n_images),
            "processed_patch_count": str(_n_patches),
            "temporal_resolution":   _freq,
            "subsampled_from":       BASE_EXPERIMENT,
            "subsample_script":      "notebooks/04-benchmarking/subsample_embeddings.py",
        }
        _config_rows = [{"key": _k, "value": _v} for _k, _v in _overrides.items()]
        _override_keys = set(_overrides)
        for _k, _v in base_config.items():
            if _k not in _override_keys:
                _config_rows.append({"key": _k, "value": str(_v)})

        _out_db.create_table("config", data=pa.Table.from_pylist(_config_rows))

        # ── IVF-PQ index on patch_embeddings ─────────────────────────────────
        # num_partitions: nearest power-of-2 to N/4096 (minimum 1)
        # num_sub_vectors: hardcoded to 96  (768 / 96 = 8 bytes per vector)
        _num_partitions  = max(1, 2 ** round(math.log2(_n_patches / 4096)))
        _num_sub_vectors = 96

        _index_kwargs = dict(
            metric="cosine",
            index_type="IVF_PQ",
            num_partitions=_num_partitions,
            num_sub_vectors=_num_sub_vectors,
            vector_column_name="embedding",
        )
        if _cuda_available:
            _index_kwargs["accelerator"] = "cuda"

        console.print(
            f"  Building IVF-PQ : partitions={_num_partitions}, "
            f"sub_vectors={_num_sub_vectors}, "
            f"accelerator={'cuda' if _cuda_available else 'cpu (no CUDA)'}"
        )
        _patch_out_tbl.create_index(**_index_kwargs)
        console.print(f"  [green]✓ {_project} complete[/green]")

    console.print("\n[bold green]All experiments generated.[/bold green]")
    return


@app.cell
def _(DB_URI, RESOLUTIONS, Table, console, lancedb, load_config):
    """Re-open each output experiment and print a summary Rich table."""
    _vtbl = Table(title="Subsampled Experiments — Verification")
    _vtbl.add_column("Experiment",   style="cyan",  no_wrap=True)
    _vtbl.add_column("Resolution",   style="yellow")
    _vtbl.add_column("Images",       justify="right")
    _vtbl.add_column("Patches",      justify="right")
    _vtbl.add_column("Config keys",  justify="right")
    _vtbl.add_column("Index",        justify="center")

    for _freq, _project in RESOLUTIONS:
        _out_path = DB_URI / _project
        if not _out_path.exists():
            _vtbl.add_row(_project, _freq, "—", "—", "—", "[red]missing[/red]")
            continue
        try:
            _db       = lancedb.connect(str(_out_path))
            _img_n    = _db.open_table("image_embeddings").count_rows()
            _ptbl     = _db.open_table("patch_embeddings")
            _patch_n  = _ptbl.count_rows()
            _has_idx  = "[green]✓[/green]" if _ptbl.list_indices() else "[red]✗[/red]"
            _cfg      = load_config(str(_out_path), "config")
            _cfg_n    = len(_cfg)
        except Exception as _e:
            _vtbl.add_row(_project, _freq, "ERR", "ERR", "ERR", str(_e))
            continue
        _vtbl.add_row(
            _project, _freq,
            f"{_img_n:,}", f"{_patch_n:,}", str(_cfg_n),
            _has_idx,
        )

    console.print(_vtbl)
    return


if __name__ == "__main__":
    app.run()
