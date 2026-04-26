import marimo

__generated_with = "0.20.4"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Generate Embeddings

    End-to-end workflow for running image and patch embedding experiments against a LanceDB database.
    Choose a model family, configure your data paths, then either run inline or generate a PBS/HPC job command.
    """)
    return


@app.cell
def _():
    import marimo as mo
    import lancedb
    import matplotlib.pyplot as plt
    import numpy as np
    from pathlib import Path

    from helpers.embedding_experiment import (
        build_cli_command,
        get_model_info,
        list_models,
        load_config,
        print_table_sizes,
        run_experiment,
        setup_experiment,
    )

    return (
        Path,
        build_cli_command,
        get_model_info,
        lancedb,
        list_models,
        load_config,
        mo,
        np,
        plt,
        print_table_sizes,
        setup_experiment,
    )


@app.cell
def _(mo):
    mo.md("""
    ## Available Models
    """)
    return


@app.cell
def _(list_models):
    list_models()
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Configuration

    Set paths below. Update `PROJECT_ROOT` to switch between environments.
    """)
    return


@app.cell
def _(Path):
    # Local Mac (for reference):
    # PROJECT_ROOT = Path("/Users/ncheruku/Documents/Work/sample_data")

    # NCAR Casper
    PROJECT_ROOT = Path("/glade/work/ncheruku/research/sample_data")

    # Project folder name — must match the SOURCE_PROJECT used during ingest.
    # This is the subfolder inside shared_source/ that holds the source LanceDB.
    SOURCE_PROJECT = "era5_hrly_2016_2018_images"

    SOURCE_URI = PROJECT_ROOT / "data" / "lancedb" / "shared_source" / SOURCE_PROJECT
    IMG_RAW_TBL_NAME = "images"
    DB_URI = PROJECT_ROOT / "data" / "lancedb" / "experiments" / "era5"
    AUTHOR = "Cherukuru. N. W"

    # Target image dimensions passed to the embedding script.
    # Both must be multiples of the model's patch size (16 for ViT-base-patch16).
    # For the ERA5 7:2 geographic aspect ratio use IMAGE_H=256, IMAGE_W=896.
    IMAGE_H = 256
    IMAGE_W = 896
    return (
        AUTHOR,
        DB_URI,
        IMAGE_H,
        IMAGE_W,
        IMG_RAW_TBL_NAME,
        PROJECT_ROOT,
        SOURCE_URI,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Model Selection

    Change `PROJECT_NAME` to switch between model families (`dinov3` or `openclip`).
    Defaults are loaded from `helpers/model_registry.json`.
    """)
    return


@app.cell
def _(get_model_info):
    # Model-specific config — change this for a different model family
    PROJECT_NAME = "dinov3_1h"
    model_info = get_model_info("dinov3_rect")

    MODEL = model_info["default_model"]
    SCRIPT = model_info["script_path"]
    BATCH = 128        # A100: override registry default (64)
    WORKERS = 4       # A100: override registry default (5); half of 32 cores
    SCAN_BATCH = 8192  # A100: override registry default (1000); 128 GB RAM

    print(f"Family: {PROJECT_NAME}")
    print(f"Script: {model_info['script']}")
    print(f"Model:  {MODEL}")
    return BATCH, MODEL, PROJECT_NAME, SCAN_BATCH, SCRIPT, WORKERS


@app.cell
def _(mo):
    mo.md(r"""
    ## Setup Experiment

    Creates a config entry in LanceDB that tracks metadata (author, paths, model, timestamps)
    and returns the table names for image and patch embeddings.
    """)
    return


@app.cell
def _(
    AUTHOR,
    DB_URI,
    IMG_RAW_TBL_NAME,
    PROJECT_NAME,
    PROJECT_ROOT,
    SOURCE_URI,
    setup_experiment,
):
    experiment = setup_experiment(
        PROJECT_NAME, AUTHOR, SOURCE_URI, IMG_RAW_TBL_NAME,
        DB_URI, project_root=PROJECT_ROOT,
    )
    return (experiment,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Run

    **Case 1** — run inline; blocks until complete, streaming logs to stdout.
    **Case 2** — print a ready-to-paste shell command for PBS/HPC job submission.
    Comment out whichever case you don't need.
    """)
    return


@app.cell
def _():
    # # Case 1: Run inline (interactive notebook workflow)
    # run_experiment(
    #     SCRIPT, SOURCE_URI, IMG_RAW_TBL_NAME,
    #     experiment["exp_db_uri"], experiment["config_name"],
    #     MODEL, batch=BATCH, scan_batch=SCAN_BATCH, workers=WORKERS,
    #     image_h=IMAGE_H, image_w=IMAGE_W,
    # )
    return


@app.cell
def _(
    BATCH,
    IMAGE_H,
    IMAGE_W,
    IMG_RAW_TBL_NAME,
    MODEL,
    SCAN_BATCH,
    SCRIPT,
    SOURCE_URI,
    WORKERS,
    build_cli_command,
    experiment,
):
    # Case 2: Build CLI command for PBS / external job submission
    cmd = build_cli_command(
        SCRIPT, SOURCE_URI, IMG_RAW_TBL_NAME,
        experiment["exp_db_uri"], experiment["config_name"],
        MODEL, batch=BATCH, scan_batch=SCAN_BATCH, workers=WORKERS,
        image_h=IMAGE_H, image_w=IMAGE_W,
    )
    print(cmd)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Inspect Results

    Load the config record and check table sizes on disk.
    Works for both Case 1 (inline) and Case 2 (after the job finishes).
    """)
    return


@app.cell
def _(experiment, load_config):
    # Inspect config after run completes (works for both Case 1 and Case 2)
    config = load_config(experiment["exp_db_uri"], experiment["config_name"])
    config
    return (config,)


@app.cell
def _(experiment, print_table_sizes):
    print_table_sizes(
        experiment["exp_db_uri"],
        experiment["config_name"],
        experiment["img_emb_name"],
        experiment["patch_emb_name"],
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Explore Embeddings

    Open the result tables and visualise a sample image embedding (histogram + attention map).
    """)
    return


@app.cell
def _(experiment, lancedb):
    db = lancedb.connect(experiment["exp_db_uri"])
    patch_tbl = db.open_table(experiment["patch_emb_name"])
    return db, patch_tbl


@app.cell
def _(db, experiment):
    img_tbl = db.open_table(experiment["img_emb_name"])
    return (img_tbl,)


@app.cell
def _(np, plt):
    def preview_image_embedding(tbl, idx: int = 0, spatial_h: int = None, spatial_w: int = None) -> None:
        """Preview a single row from the image embeddings table by index.

        Prints a column summary and renders two plots:
          - Left:  histogram of the embedding vector values
          - Right: attention map reshaped to its spatial grid (e.g. 16×56)

        Parameters
        ----------
        tbl : LanceDB Table
            The image embeddings table (must have columns: image_id, embedding, attention_map).
        idx : int
            Row index to inspect.
        spatial_h : int
            Number of patch rows (image_h // patch_size). Read from config["attention_spatial_h"].
        spatial_w : int
            Number of patch columns (image_w // patch_size). Read from config["attention_spatial_w"].
        """
        row = tbl.to_lance().take([idx]).to_pydict()

        image_id = row["image_id"][0]
        emb = np.array(row["embedding"][0], dtype=np.float32)
        attn = np.array(row["attention_map"][0], dtype=np.float32)

        attn_2d = attn.reshape(spatial_h, spatial_w)

        # ── Column summary ──────────────────────────────────────────────────
        print(f"idx         : {idx}")
        print(f"image_id    : {image_id}")
        print(f"embedding   : dim={len(emb)}, norm={np.linalg.norm(emb):.6f}, "
              f"min={emb.min():.4f}, max={emb.max():.4f}, mean={emb.mean():.4f}")
        print(f"attention   : patches={len(attn)} ({spatial_h}×{spatial_w}), "
              f"min={attn.min():.4f}, max={attn.max():.4f}, sum={attn.sum():.4f}")

        # ── Plots ────────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        fig.suptitle(f"idx={idx}  |  {image_id}", fontsize=10)

        # Embedding histogram
        axes[0].hist(emb, bins=64, color="steelblue", edgecolor="none")
        axes[0].axvline(emb.mean(), color="tomato", linestyle="--",
                        label=f"mean = {emb.mean():.3f}")
        axes[0].set_title(f"Embedding  (dim={len(emb)}, ‖v‖={np.linalg.norm(emb):.4f})")
        axes[0].set_xlabel("value")
        axes[0].set_ylabel("count")
        axes[0].legend(fontsize=8)

        # Attention map heatmap
        im = axes[1].imshow(attn_2d, cmap="inferno", interpolation="nearest")
        axes[1].set_title(f"Attention map  ({spatial_h}×{spatial_w})")
        axes[1].axis("off")
        plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

        plt.tight_layout()
        plt.show()

    return (preview_image_embedding,)


@app.cell
def _(config, img_tbl, preview_image_embedding):
    _spatial_h = int(config["attention_spatial_h"])
    _spatial_w = int(config["attention_spatial_w"])
    preview_image_embedding(img_tbl, idx=789, spatial_h=_spatial_h, spatial_w=_spatial_w)
    return


@app.cell
def _(patch_tbl):
    patch_tbl.schema
    return


@app.cell
def _(patch_tbl):
    row = patch_tbl.search().limit(1).to_pandas().iloc[0]
    row["image_id"]
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## GPU / PyTorch Check

    Confirm that CUDA is available before running GPU-accelerated indexing below.
    """)
    return


@app.cell
def _():
    import torch

    print(f"PyTorch Version: {torch.__version__}")
    print(f"Is CUDA available? {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Create Vector Index

    Builds an IVF-PQ ANN index on patch embeddings for fast approximate nearest-neighbour search.
    Adjust `num_partitions` and `num_sub_vectors` to trade recall for speed.
    """)
    return


@app.cell
def _(patch_tbl):
    patch_tbl.create_index(metric="cosine", index_type="IVF_PQ", num_partitions=4096, num_sub_vectors=96, accelerator="cuda", vector_column_name="embedding")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
