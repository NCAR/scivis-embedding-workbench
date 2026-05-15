"""Direct ANN search over the patch embeddings table — no image pre-filter."""
import pandas as pd

from ..base import SearchContext, SearchStrategy


class PatchFirst(SearchStrategy):
    name = "Patch first"
    description = (
        "Vector-search patch_emb_tbl directly against the query patch. "
        "Metadata + spatial filters are applied as a WHERE clause; no image "
        "pre-filter is performed."
    )

    def search(self, ctx: SearchContext) -> pd.DataFrame:
        parts = []
        if ctx.allowed_image_ids:
            id_clause = ", ".join(f"'{i}'" for i in ctx.allowed_image_ids)
            parts.append(f"image_id IN ({id_clause})")
        if ctx.spatial_patch_indices:
            patch_clause = ", ".join(str(i) for i in ctx.spatial_patch_indices)
            parts.append(f"patch_index IN ({patch_clause})")
        where = " AND ".join(parts) if parts else None

        q = (
            ctx.patch_emb_tbl.search(
                ctx.query_patch_embedding, vector_column_name="embedding"
            )
            .metric("cosine")
            .refine_factor(int(ctx.refine_factor))
            .select(["image_id", "patch_index"])
            .limit(ctx.n_similar_patches)
        )
        if where:
            q = q.where(where)
        return q.to_pandas()
