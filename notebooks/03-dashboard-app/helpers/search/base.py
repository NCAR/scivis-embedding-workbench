"""Pluggable search strategy framework for the dashboard.

A strategy is any subclass of `SearchStrategy` that lives in a `.py` file under
`helpers/search/strategies/`. Subclasses auto-register at import time via
`__init_subclass__`; `discover_strategies()` imports every module in that
directory and returns `{name: instance}` for the dropdown to consume.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
import pandas as pd


@dataclass
class SearchContext:
    """All inputs a strategy may need. Passed to `SearchStrategy.search()`.

    Mirrors the values gathered in app.py before the search dispatch — tables,
    the query (image + patch), result-size knobs, and active filters.
    """
    img_emb_tbl: Any
    patch_emb_tbl: Any
    src_img_tbl: Any
    selected_img_id: str
    patch_idx: int
    query_patch_embedding: np.ndarray
    n_similar_images: int
    n_similar_patches: int
    refine_factor: int
    allowed_image_ids: Optional[list]
    spatial_patch_indices: Optional[list]


class SearchStrategy(ABC):
    """Base class for a search strategy. Subclass and define `name` + `search()`.

    Subclasses auto-register in `SearchStrategy._registry` keyed by `name`.
    """
    name: str = ""
    description: str = ""

    _registry: dict = {}

    def __init_subclass__(cls, **kw):
        super().__init_subclass__(**kw)
        if not getattr(cls, "name", ""):
            raise TypeError(
                f"{cls.__name__} must define a non-empty `name` class attribute"
            )
        if cls.name in SearchStrategy._registry:
            existing = SearchStrategy._registry[cls.name].__module__
            raise ValueError(
                f"Duplicate search-strategy name {cls.name!r} "
                f"(already registered by {existing})"
            )
        SearchStrategy._registry[cls.name] = cls

    @abstractmethod
    def search(self, ctx: SearchContext) -> pd.DataFrame:
        """Run the search and return a DataFrame with columns
        `image_id`, `patch_index`, `_distance`."""


def discover_strategies() -> dict:
    """Import every `.py` in `helpers/search/strategies/` and return
    `{name: instance}` for every registered subclass."""
    import importlib
    import pkgutil

    from . import strategies as _pkg

    for _, modname, _ in pkgutil.iter_modules(_pkg.__path__):
        importlib.import_module(f"{_pkg.__name__}.{modname}")
    return {name: cls() for name, cls in SearchStrategy._registry.items()}
