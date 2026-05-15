# Search strategies

Each `.py` file in this directory that defines a `SearchStrategy` subclass becomes a selectable option in the Spatial Search tab's "Search mode" dropdown. Files ending in `.example` are ignored.

## Add your own strategy

1. Copy `template.py.example` to a new file (e.g. `my_strategy.py`).
2. Edit `name`, `description`, and the body of `search()`. See [`../base.py`](../base.py) for the `SearchContext` dataclass — it carries all the inputs your strategy can use (LanceDB tables, query embedding, filters, result-size knobs).
3. Restart marimo. The strategy appears in the dropdown automatically — no registration list to update.

The strategy must return a `pandas.DataFrame` with columns `image_id`, `patch_index`, `_distance`. Lower `_distance` = more similar (LanceDB's standard cosine-distance convention).

## Contribute upstream

Strategies are self-contained — one file, no central registry. To share yours with everyone, open a PR with just your new file. Reviewers see exactly the new strategy; no merge conflicts with parallel contributions.

Keep strategies focused: one strategy per file, no cross-imports between strategies. If two strategies share helpers, put the helpers in their own module under `helpers/search/`.

## Built-ins for reference

- [`patch_first.py`](patch_first.py) — direct ANN over `patch_emb_tbl` with WHERE filters.
- [`image_first.py`](image_first.py) — hierarchical: top-N similar images, then patches within them.
