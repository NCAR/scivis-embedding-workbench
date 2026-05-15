"""Tests for the pluggable search-strategy registry."""
import importlib.util
import sys
from dataclasses import fields
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# The repo has two `helpers` packages (one per notebook). The conftest exposes
# the embedding-experiments one for other tests. We load the dashboard
# `helpers.search` package directly from its file path under a unique module
# name so it doesn't collide with the cached `helpers` module.
_PKG_ROOT = Path(__file__).parent.parent / "notebooks" / "03-dashboard-app" / "helpers" / "search"


def _load(modname: str, path: Path):
    spec = importlib.util.spec_from_file_location(modname, path, submodule_search_locations=[str(path.parent)] if path.name == "__init__.py" else None)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod


_search_pkg = _load("dashboard_search", _PKG_ROOT / "__init__.py")
SearchContext = _search_pkg.SearchContext
SearchStrategy = _search_pkg.SearchStrategy
discover_strategies = _search_pkg.discover_strategies


def test_registry_loads_builtins():
    strategies = discover_strategies()
    assert "Image first" in strategies
    assert "Patch first" in strategies
    for inst in strategies.values():
        assert isinstance(inst, SearchStrategy)
        assert callable(inst.search)
        assert inst.name
        assert inst.description


def test_missing_name_rejected():
    with pytest.raises(TypeError, match="must define a non-empty `name`"):
        class _Nameless(SearchStrategy):
            def search(self, ctx):  # pragma: no cover
                return pd.DataFrame()


def test_duplicate_name_rejected():
    class _Unique(SearchStrategy):
        name = "__test_unique_strategy__"
        def search(self, ctx):  # pragma: no cover
            return pd.DataFrame()

    try:
        with pytest.raises(ValueError, match="Duplicate search-strategy name"):
            class _Dup(SearchStrategy):
                name = "__test_unique_strategy__"
                def search(self, ctx):  # pragma: no cover
                    return pd.DataFrame()
    finally:
        # Clean up so subsequent test runs in the same process don't see the registration
        SearchStrategy._registry.pop("__test_unique_strategy__", None)


def test_search_context_has_all_fields():
    ctx = SearchContext(
        img_emb_tbl="img",
        patch_emb_tbl="patch",
        src_img_tbl="src",
        selected_img_id="abc",
        patch_idx=7,
        query_patch_embedding=np.zeros(4, dtype=np.float32),
        n_similar_images=10,
        n_similar_patches=100,
        refine_factor=20,
        allowed_image_ids=["x", "y"],
        spatial_patch_indices=[1, 2, 3],
    )
    names = {f.name for f in fields(SearchContext)}
    expected = {
        "img_emb_tbl", "patch_emb_tbl", "src_img_tbl",
        "selected_img_id", "patch_idx", "query_patch_embedding",
        "n_similar_images", "n_similar_patches", "refine_factor",
        "allowed_image_ids", "spatial_patch_indices",
    }
    assert names == expected
    assert ctx.patch_idx == 7
    assert ctx.spatial_patch_indices == [1, 2, 3]
