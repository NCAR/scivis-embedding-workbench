# Embeddings

**Orchestrator:** `notebooks/02-generate-embeddings/generate_dinov3_embeddings.py`
**Engine:** `notebooks/02-generate-embeddings/helpers/v5_dino_embeddings_lancedb.py`

Connects to the source image table, configures an experiment, and runs a high-throughput embedding pipeline that produces image-level and patch-level representations.

---

## Supported Models

| Family | Model | Output dim |
|---|---|---|
| `dinov3_rect` | DINOv3 ViT-B/16 (rectangular) | 768 |
| `dinov3` | DINOv3 ViT-B/16 (square) | 768 |
| `dinov3_sat_rect` | DINOv3 ViT-L/16 SAT-493M (rectangular, satellite) | 1024 |
| `openclip` | OpenCLIP ViT-B/32 | 512 |

Models are registered in `helpers/model_registry.json` and loaded via `timm`.

### Satellite weights

`dinov3_sat_rect` uses `vit_large_patch16_dinov3.sat493m`, pretrained on SAT-493M — 493M 512×512 crops sampled from Maxar RGB ortho-rectified imagery at 0.6 m resolution. It is **RGB only**: three channels, no NIR/SWIR/SAR and no band-adaptive input stem. Multispectral sources would need a different model family and a new engine script.

The `.sat493m` tag is required. Dropping it loads the LVD-1689M web weights *and* ImageNet normalization instead of the satellite mean/std of `(0.430, 0.411, 0.296)` / `(0.213, 0.156, 0.143)`, which `resolve_model_data_config()` picks up automatically from the tag.

Switching to this family raises the embedding dim from 768 to 1024, which grows the `patch_embeddings` table by roughly a third. The orchestrator derives the IVF-PQ `num_sub_vectors` from `embedding_dim` via `pick_num_sub_vectors()`, since LanceDB requires it to divide the vector dimension evenly.

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
    DINOv2 is released by Meta AI under Apache 2.0. DINOv3 weights are released by Meta AI under the DINOv3 License and are **gated on HuggingFace** — accept the license on the model page, then authenticate with `hf auth login` before the first run. OpenCLIP is released by LAION under MIT/BSD. Model weights are downloaded automatically at runtime via `timm` / `open_clip`.
