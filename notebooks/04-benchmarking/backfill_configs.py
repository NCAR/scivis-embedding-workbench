#!/usr/bin/env python3
"""
backfill_configs.py — Upsert missing v5 metadata into LanceDB experiment config tables.

One-off recovery for experiments whose config was wiped by a re-run of
setup_experiment() (mode="overwrite") after the PBS embedding job had written its
metadata.  Uses the exact same helper functions as v5_dino_embeddings_lancedb.py so
the key names and values match what a normal run would have produced.

The model is loaded WITHOUT pretrained weights (architecture only) to avoid
downloading a checkpoint just for geometry introspection.

All writes use upsert_config() — existing correct values are never overwritten.
Safe to re-run.

Usage
-----
    uv run python notebooks/04-benchmarking/backfill_configs.py
"""

from __future__ import annotations

import platform
import sys
import uuid
from datetime import UTC, datetime
from pathlib import Path

# ── Path setup ─────────────────────────────────────────────────────────────────
_repo_root   = Path(__file__).parent.parent.parent
_helpers_dir = str(_repo_root / "notebooks" / "02-generate-embeddings" / "helpers")
if _helpers_dir not in sys.path:
    sys.path.insert(0, _helpers_dir)

import lancedb
import timm
import torch
from embedding_experiment import load_config, upsert_config
from rich.console import Console
from rich.table import Table

# Import the exact helper functions v5 uses so values are identical
from v5_dino_embeddings_lancedb import (
    get_git_info,
    get_pkg_version,
    get_script_sha256,
    resolve_model_data_config,
)

console = Console()

# ── Experiment config (matches generate_dinov3_embeddings.py) ──────────────────
PROJECT_ROOT = Path("/glade/work/ncheruku/research/sample_data")
DB_URI       = PROJECT_ROOT / "data" / "lancedb" / "experiments" / "era5"
SOURCE_URI   = PROJECT_ROOT / "data" / "lancedb" / "shared_source" / "era5_hrly_2016_2018_images"

MODEL_NAME  = "vit_base_patch16_dinov3"
IMAGE_H     = 256
IMAGE_W     = 896
BATCH_SIZE  = 128
SCAN_BATCH  = 8192
WORKERS     = 4

EXPERIMENTS = [
    ("1h",  "dinov3_1h"),
    ("3h",  "dinov3_3h"),
    ("6h",  "dinov3_6h"),
    ("12h", "dinov3_12h"),
    ("24h", "dinov3_24h"),
]

# ── Derive geometry from model architecture (no pretrained weights needed) ──────
console.print("[bold]Loading model architecture (pretrained=False) …[/bold]")
_model = timm.create_model(
    MODEL_NAME,
    pretrained=False,
    num_classes=0,
    global_pool="avg",
    dynamic_img_size=True,
)
_model.eval()

with torch.no_grad():
    _dummy = torch.zeros(1, 3, IMAGE_H, IMAGE_W)
    _tok   = _model.forward_features(_dummy)

patch_size_used  = int(_model.patch_embed.patch_size[0])
spatial_h        = IMAGE_H // patch_size_used
spatial_w        = IMAGE_W // patch_size_used
num_patch_tokens = spatial_h * spatial_w
num_tokens_total = int(_tok.shape[1])
num_extra_tokens = num_tokens_total - num_patch_tokens
embedding_dim    = int(_tok.shape[2])
attn_layer_idx   = len(_model.blocks) - 1
attn_num_heads   = int(_model.blocks[attn_layer_idx].attn.num_heads)

console.print(f"  patch_size       = {patch_size_used}")
console.print(f"  spatial grid     = {spatial_h} × {spatial_w}")
console.print(f"  num_patch_tokens = {num_patch_tokens}")
console.print(f"  num_extra_tokens = {num_extra_tokens}  (CLS + registers)")
console.print(f"  num_tokens_total = {num_tokens_total}")
console.print(f"  embedding_dim    = {embedding_dim}")
console.print(f"  attn_layer_index = {attn_layer_idx}  ({attn_num_heads} heads)")

del _model, _tok, _dummy

# ── Preprocessing config (matches v5's resolve_model_data_config call) ──────────
_data_cfg    = resolve_model_data_config(MODEL_NAME)
mean_val     = str(_data_cfg["mean"])
std_val      = str(_data_cfg["std"])
interpolation = str(_data_cfg.get("interpolation", "bicubic"))

# ── Environment / reproducibility info ────────────────────────────────────────
_git       = get_git_info()
_sha256 = get_script_sha256()  # hashes v5_dino_embeddings_lancedb.py (function uses its own __file__)

try:
    import timm as _timm_mod
    _timm_ver = get_pkg_version(_timm_mod)
except Exception:
    _timm_ver = "unknown"

_torch_ver  = get_pkg_version(torch)
_python_ver = sys.version.replace("\n", " ")
_platform   = platform.platform()

# ── Base config dict (keys and values match v5_dino_embeddings_lancedb.py) ──────
BASE_CONFIG: dict[str, str] = {
    # Script identity
    "script_name":                "v5_dino_embeddings_lancedb.py",
    "script_sha256":              _sha256,
    "backfill_note":              (
        "Config repopulated by backfill_configs.py. "
        "git_commit/branch/dirty reflect the backfill run, not the original embedding run. "
        "run_id, elapsed_seconds, throughput_img_per_sec could not be recovered."
    ),
    # Git (current repo state, not original run)
    "git_commit":                 _git["git_commit"],
    "git_branch":                 _git["git_branch"],
    "git_dirty":                  _git["git_dirty"],
    # Unrecoverable runtime fields
    "run_id":                     "N/A (backfilled)",
    "elapsed_seconds":            "N/A (backfilled)",
    "throughput_img_per_sec":     "N/A (backfilled)",
    # Source data (from notebook config)
    "raw_db_uri":                 str(SOURCE_URI),
    "raw_table":                  "images",
    "raw_img_id_field":           "id",
    "raw_img_blob_field":         "image_blob",
    # Model
    "model_name":                 MODEL_NAME,
    "pretrained":                 "true",
    "image_h":                    str(IMAGE_H),
    "image_w":                    str(IMAGE_W),
    "image_mode":                 "rectangular",
    "patch_size":                 str(patch_size_used),
    "embedding_dim":              str(embedding_dim),
    "num_patch_tokens":           str(num_patch_tokens),
    "num_extra_tokens":           str(num_extra_tokens),
    "num_tokens_total":           str(num_tokens_total),
    # Preprocessing
    "interpolation":              interpolation,
    "mean":                       mean_val,
    "std":                        std_val,
    "crop_pct":                   "none — v5 does not center-crop",
    # Device / precision (GPU used for actual run; we don't record that here)
    "dtype_requested":            "fp16",
    "dtype_used":                 "fp16",
    # Pipeline params (from notebook)
    "batch_size":                 str(BATCH_SIZE),
    "scan_batch":                 str(SCAN_BATCH),
    "workers":                    str(WORKERS),
    "limit":                      "0",
    "mp_start_method":            "spawn",
    "img_emb_table_current":      "image_embeddings",
    "patch_emb_table_current":    "patch_embeddings",
    # Attention map metadata
    "attention_layer_index":      str(attn_layer_idx),
    "attention_num_heads_averaged": str(attn_num_heads),
    "attention_spatial_h":        str(spatial_h),
    "attention_spatial_w":        str(spatial_w),
    "attention_num_patch_tokens": str(num_patch_tokens),
    "attention_dtype":            "float32",
    "attention_storage":          "pa.list_(pa.float32(), num_patch_tokens)",
    # IVF-PQ index params
    "ivf_metric":                 "cosine",
    "ivf_num_sub_vectors":        "96",
    # Environment
    "torch_version":              _torch_ver,
    "timm_version":               _timm_ver,
    "python_version":             _python_ver,
    "platform":                   _platform,
}

# ── Upsert into each experiment ────────────────────────────────────────────────
console.print()
summary = Table(title="Config Backfill — Results")
summary.add_column("Experiment",   style="cyan", no_wrap=True)
summary.add_column("Images",       justify="right")
summary.add_column("Patches",      justify="right")
summary.add_column("Keys before",  justify="right")
summary.add_column("Keys after",   justify="right")
summary.add_column("Status",       justify="center")

for freq, exp_name in EXPERIMENTS:
    exp_path = DB_URI / exp_name
    if not exp_path.exists():
        summary.add_row(exp_name, "—", "—", "—", "—", "[red]missing[/red]")
        continue

    try:
        db        = lancedb.connect(str(exp_path))
        img_tbl   = db.open_table("image_embeddings")
        patch_tbl = db.open_table("patch_embeddings")

        n_images  = img_tbl.count_rows()
        n_patches = patch_tbl.count_rows()

        # Sanity-check embedding_dim against the live Arrow schema
        schema_dim = patch_tbl.to_lance().schema.field("embedding").type.list_size
        if schema_dim and schema_dim != embedding_dim:
            console.print(
                f"[yellow]  {exp_name}: schema embedding_dim={schema_dim} "
                f"overrides derived value {embedding_dim}[/yellow]"
            )

        cfg_before = load_config(str(exp_path), "config")
        n_before   = len(cfg_before)

        updates = dict(BASE_CONFIG)
        updates["temporal_resolution"]   = freq
        updates["processed_image_count"] = str(n_images)
        updates["processed_patch_count"] = str(n_patches)

        upsert_config(str(exp_path), "config", updates)

        cfg_after = load_config(str(exp_path), "config")
        n_after   = len(cfg_after)

        summary.add_row(
            exp_name, f"{n_images:,}", f"{n_patches:,}",
            str(n_before), str(n_after),
            "[green]✓[/green]",
        )

    except Exception as exc:
        summary.add_row(exp_name, "ERR", "ERR", "—", "—", f"[red]{exc}[/red]")

console.print(summary)
