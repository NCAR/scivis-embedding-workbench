"""Tests for pure @app.function helpers extracted from app.py."""
import re
from pathlib import Path

import numpy as np
import pytest

# Extract each function from app.py source using regex (same pattern as test_theme.py).
# This avoids importing the full Marimo app, which has heavy side-effects (lancedb,
# IPython, wigglystuff, cartopy, etc.).
_APP_SRC = (Path(__file__).parent.parent / "notebooks" / "03-dashboard-app" / "app.py").read_text()


def _extract(fn_name: str):
    """Pull a top-level function definition out of app.py and exec it."""
    m = re.search(
        rf"(^def {fn_name}\b.*?)(?=^@app\.|^def |\Z)",
        _APP_SRC,
        re.MULTILINE | re.DOTALL,
    )
    assert m, f"{fn_name} not found in app.py"
    ns: dict = {}
    exec(m.group(1), ns)  # noqa: S102
    return ns[fn_name]


list_experiments       = _extract("list_experiments")
resolve_source_path    = _extract("resolve_source_path")
compute_thumb_dimensions = _extract("compute_thumb_dimensions")
apply_brush_filter     = _extract("apply_brush_filter")


# ── list_experiments ──────────────────────────────────────────────────────────

def test_list_experiments_empty_dir(tmp_path):
    assert list_experiments(str(tmp_path)) == []


def test_list_experiments_missing_path():
    assert list_experiments("/nonexistent/path/xyz_abc") == []


def test_list_experiments_finds_valid_subfolder(tmp_path):
    # LanceDB tables are directories, so config.lance is a directory
    (tmp_path / "dinov3" / "config.lance").mkdir(parents=True)
    assert list_experiments(str(tmp_path)) == ["dinov3"]


def test_list_experiments_ignores_flat_table_layout(tmp_path):
    # Old flat layout: dinov3_config.lance at the top level — should NOT be discovered
    (tmp_path / "dinov3_config.lance").mkdir()
    assert list_experiments(str(tmp_path)) == []


def test_list_experiments_requires_config_lance_dir(tmp_path):
    # Subfolder exists but has no config.lance inside — should be ignored
    (tmp_path / "dinov3").mkdir()
    assert list_experiments(str(tmp_path)) == []


def test_list_experiments_sorted(tmp_path):
    for name in ["zoo", "alpha", "beta"]:
        (tmp_path / name / "config.lance").mkdir(parents=True)
    assert list_experiments(str(tmp_path)) == ["alpha", "beta", "zoo"]


def test_list_experiments_multiple(tmp_path):
    for name in ["dinov3", "openclip"]:
        (tmp_path / name / "config.lance").mkdir(parents=True)
    assert list_experiments(str(tmp_path)) == ["dinov3", "openclip"]


# ── resolve_source_path ───────────────────────────────────────────────────────

def test_resolve_absolute_existing(tmp_path):
    source = tmp_path / "data" / "source"
    source.mkdir(parents=True)
    assert resolve_source_path(str(tmp_path), str(source)) == str(source)


def test_resolve_absolute_missing(tmp_path):
    result = resolve_source_path(str(tmp_path), str(tmp_path / "nope"))
    assert result is None


def test_resolve_relative_found_at_ancestor(tmp_path):
    # DB is deep: tmp/experiments/era5/dinov3
    # Source is at: tmp/data/source  (relative path: data/source)
    db_path = tmp_path / "experiments" / "era5" / "dinov3"
    db_path.mkdir(parents=True)
    source = tmp_path / "data" / "source"
    source.mkdir(parents=True)
    result = resolve_source_path(str(db_path), "data/source")
    assert result == str(source)


def test_resolve_relative_not_found(tmp_path):
    db_path = tmp_path / "experiments" / "dinov3"
    db_path.mkdir(parents=True)
    assert resolve_source_path(str(db_path), "nonexistent/path") is None


# ── compute_thumb_dimensions ──────────────────────────────────────────────────

def test_square_domain_at_equator():
    # lat 0–10, lon 0–10, mean_lat=5 → cos(5°) ≈ 0.996 ≈ 1 → nearly square
    ext = {"lat_min": 0.0, "lat_max": 10.0, "lon_min": 0.0, "lon_max": 10.0}
    w, h = compute_thumb_dimensions(ext, base_size=200)
    # effective_lon ≈ 10 * 0.996 ≈ 9.96, aspect ≈ 0.996 < 1 → h=base_size
    assert h == 200
    assert w <= 200


def test_wide_domain_width_capped(tmp_path):
    # ERA5 domain: lat 15–35 (range=20), lon 260–330 (range=70), mean_lat=25
    ext = {"lat_min": 15.0, "lat_max": 35.0, "lon_min": 260.0, "lon_max": 330.0}
    w, h = compute_thumb_dimensions(ext, base_size=192)
    assert w == 192   # width capped at base_size
    assert h < 192    # height reduced


def test_zero_lat_range_fallback(tmp_path):
    ext = {"lat_min": 10.0, "lat_max": 10.0, "lon_min": 0.0, "lon_max": 20.0}
    w, h = compute_thumb_dimensions(ext, base_size=100)
    assert w == 100 and h == 100


def test_zero_lon_range_fallback(tmp_path):
    ext = {"lat_min": 0.0, "lat_max": 20.0, "lon_min": 10.0, "lon_max": 10.0}
    w, h = compute_thumb_dimensions(ext, base_size=100)
    assert w == 100 and h == 100


def test_returns_integers():
    ext = {"lat_min": 15.0, "lat_max": 35.0, "lon_min": 260.0, "lon_max": 330.0}
    w, h = compute_thumb_dimensions(ext, base_size=192)
    assert isinstance(w, int) and isinstance(h, int)


def test_tall_domain_height_capped():
    # lat range >> lon range → aspect < 1 → height == base_size
    ext = {"lat_min": 0.0, "lat_max": 60.0, "lon_min": 0.0, "lon_max": 5.0}
    w, h = compute_thumb_dimensions(ext, base_size=192)
    assert h == 192
    assert w < 192


# ── apply_brush_filter ────────────────────────────────────────────────────────

def test_no_brush_returns_none():
    data = {"x": np.array([1.0, 2.0, 3.0])}
    assert apply_brush_filter(data, {}) is None


def test_numeric_range_filter():
    data = {"x": np.array([1.0, 2.0, 3.0, 4.0])}
    result = apply_brush_filter(data, {"x": {"range": [2.0, 3.0]}})
    assert result == [1, 2]


def test_numeric_range_inclusive_bounds():
    data = {"x": np.array([1.0, 2.0, 3.0])}
    result = apply_brush_filter(data, {"x": {"range": [1.0, 3.0]}})
    assert result == [0, 1, 2]


def test_categorical_string_filter():
    data = {"label": np.array(["a", "b", "a", "c"])}
    result = apply_brush_filter(data, {"label": {"values": ["a"]}})
    assert result == [0, 2]


def test_categorical_multi_value_filter():
    data = {"label": np.array(["a", "b", "c", "a"])}
    result = apply_brush_filter(data, {"label": {"values": ["a", "b"]}})
    assert result == [0, 1, 3]


def test_unknown_axis_ignored():
    data = {"x": np.array([1.0, 2.0, 3.0])}
    # "y" not in data_cols → no filter applied → all rows pass
    result = apply_brush_filter(data, {"y": {"range": [0.0, 5.0]}})
    assert result == [0, 1, 2]


def test_and_logic_two_numeric_axes():
    data = {
        "x": np.array([1.0, 2.0, 3.0]),
        "y": np.array([10.0, 20.0, 30.0]),
    }
    result = apply_brush_filter(data, {
        "x": {"range": [1.0, 2.0]},
        "y": {"range": [15.0, 30.0]},
    })
    # x in [1,2] → indices 0,1; y in [15,30] → indices 1,2; intersection → 1
    assert result == [1]


def test_no_rows_match_returns_empty():
    data = {"x": np.array([1.0, 2.0, 3.0])}
    result = apply_brush_filter(data, {"x": {"range": [10.0, 20.0]}})
    assert result == []
