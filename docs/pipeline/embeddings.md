# Embeddings

**Orchestrator:** `notebooks/02-generate-embeddings/generate_dinov3_embeddings.py`
**Engine:** `notebooks/02-generate-embeddings/helpers/v5_dino_embeddings_lancedb.py`

Connects to the source image table, configures an experiment, and runs a high-throughput embedding pipeline that produces image-level and patch-level representations.

---

## Supported Models

| Family | Model | Output dim |
|---|---|---|
| `dinov3_rect` | DINOv2 ViT-B/14 (rectangular) | 768 |
| `dinov3` | DINOv2 ViT-B/14 (square) | 768 |
| `openclip` | OpenCLIP ViT-B/32 | 512 |

Models are registered in `helpers/model_registry.json` and loaded via `timm`.

---

## Experiment Output Layout

Each experiment writes to its own subfolder under `experiments/era5/`:

```
lancedb/experiments/era5/
  <experiment_name>/
    <experiment_name>_config.lance   ← ~35 key/value metadata pairs
    image_embeddings.lance           ← one row per image
    patch_embeddings.lance           ← one row per patch per image
```

### `image_embeddings` columns

| Column | Description |
|---|---|
| `image_id` | Foreign key back to source `images` table |
| `embedding` | L2-normalized image vector (mean-pooled patches) |
| `attention_map` | Flat CLS-to-patch attention map (`spatial_h × spatial_w`) |

### `patch_embeddings` columns

| Column | Description |
|---|---|
| `patch_id` | Unique patch identifier |
| `image_id` | Foreign key back to source `images` table |
| `patch_index` | Position within the image grid |
| `embedding` | L2-normalized patch vector |

---

## Inference Pipeline

The engine runs three concurrent components to maximize throughput:

1. **Worker pool** — decodes JPEG blobs and normalizes tensors in parallel (`mp.Pool`)
2. **Batch collector** — accumulates preprocessed tensors until the batch is full, then flushes to GPU
3. **Async writer** — background thread writing embedding rows to LanceDB while the GPU processes the next batch

For rectangular images, `dynamic_img_size=True` is used so positional embeddings adapt to the non-square grid without retraining.

---

## Running on HPC

The orchestrator notebook can generate a ready-to-submit PBS job script, or run the embedding script directly via subprocess for interactive sessions.

!!! note "Model licenses"
    DINOv2 is released by Meta AI under Apache 2.0. OpenCLIP is released by LAION under MIT/BSD. Model weights are downloaded automatically at runtime via `timm` / `open_clip`.
