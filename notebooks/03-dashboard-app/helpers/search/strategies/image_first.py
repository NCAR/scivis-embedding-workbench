"""Hierarchical search: pick top-N similar images first, then search patches within them."""
import pandas as pd

from ..base import SearchContext, SearchStrategy


class ImageFirst(SearchStrategy):
    name = "Image first"
    description = (
        "Find the top-N similar images (optionally restricted to ones with "
        "patches in the spatial region), then ANN-search patches within those "
        "images against the query patch."
    )

    def search(self, ctx: SearchContext) -> pd.DataFrame:
        empty = pd.DataFrame(columns=["image_id", "patch_index", "_distance"])

        img_q = (
            ctx.img_emb_tbl.search()
            .where(f"image_id = '{ctx.selected_img_id}'")
            .select(["embedding"])
            .limit(1)
            .to_pandas()
            .iloc[0]
        )

        patch_clause = (
            f"patch_index IN ({', '.join(str(i) for i in ctx.spatial_patch_indices)})"
            if ctx.spatial_patch_indices else ""
        )
        id_clause = (
            ", ".join(f"'{i}'" for i in ctx.allowed_image_ids)
            if ctx.allowed_image_ids else None
        )

        if ctx.spatial_patch_indices:
            spatial_img_ids = (
                ctx.patch_emb_tbl.search()
                .where(patch_clause)
                .select(["image_id"])
                .limit(100_000)
                .to_pandas()["image_id"]
                .unique()
                .tolist()
            )
            if ctx.allowed_image_ids:
                allowed_set = set(ctx.allowed_image_ids)
                spatial_img_ids = [i for i in spatial_img_ids if i in allowed_set]
            if not spatial_img_ids:
                return empty
            img_where = f"image_id IN ({', '.join(repr(i) for i in spatial_img_ids)})"
        elif id_clause:
            img_where = f"image_id IN ({id_clause})"
        else:
            img_where = None

        img_search = (
            ctx.img_emb_tbl.search(img_q["embedding"], vector_column_name="embedding")
            .metric("cosine")
            .select(["image_id"])
        )
        if img_where:
            img_search = img_search.where(img_where)
        sim_ims = img_search.limit(ctx.n_similar_images).to_pandas()
        if sim_ims.empty:
            return empty

        img_filter = ", ".join(f"'{i}'" for i in sim_ims["image_id"].tolist())
        where = f"image_id IN ({img_filter})"
        if patch_clause:
            where += f" AND {patch_clause}"

        return (
            ctx.patch_emb_tbl.search(
                ctx.query_patch_embedding, vector_column_name="embedding"
            )
            .where(where)
            .metric("cosine")
            .refine_factor(int(ctx.refine_factor))
            .select(["image_id", "patch_index"])
            .limit(ctx.n_similar_patches)
            .to_pandas()
        )
