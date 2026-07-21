"""Tests for pure helpers extracted into notebooks/03-dashboard-app/helpers/."""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless CI: no display available

import numpy as np
import pytest

# Extract each function from helpers/data.py or helpers/viz.py source using regex.
# This avoids importing the full Marimo app, which has heavy side-effects (lancedb,
# IPython, wigglystuff, cartopy, etc.).
_HELPERS_DIR = Path(__file__).parent.parent / "notebooks" / "03-dashboard-app" / "helpers"
_DATA_SRC = (_HELPERS_DIR / "data.py").read_text()
_VIZ_SRC = (_HELPERS_DIR / "viz.py").read_text()


def _extract(fn_name: str, src: str):
    """Pull a top-level function definition out of a helper module and exec it."""
    m = re.search(
        rf"(^def {fn_name}\b.*?)(?=^def |\Z)",
        src,
        re.MULTILINE | re.DOTALL,
    )
    assert m, f"{fn_name} not found"
    ns: dict = {}
    exec(m.group(1), ns)  # noqa: S102
    return ns[fn_name]


def _extract_group(src: str, *fn_names: str) -> dict:
    """Extract multiple functions into a shared namespace so they can call each
    other (e.g. render_thumbnail_gallery calls get_theme_colors)."""
    ns: dict = {}
    for fn_name in fn_names:
        m = re.search(
            rf"(^def {fn_name}\b.*?)(?=^def |\Z)",
            src,
            re.MULTILINE | re.DOTALL,
        )
        assert m, f"{fn_name} not found"
        exec(m.group(1), ns)  # noqa: S102
    return ns


list_experiments       = _extract("list_experiments",       _DATA_SRC)
resolve_source_path    = _extract("resolve_source_path",    _DATA_SRC)
compute_thumb_dimensions = _extract("compute_thumb_dimensions", _DATA_SRC)
apply_brush_filter     = _extract("apply_brush_filter",     _DATA_SRC)

_gallery_ns = _extract_group(_VIZ_SRC, "get_theme_colors", "render_thumbnail_gallery")
render_thumbnail_gallery = _gallery_ns["render_thumbnail_gallery"]

_map_ns = _extract_group(_VIZ_SRC, "get_theme_colors", "make_extent_map")
make_extent_map = _map_ns["make_extent_map"]
render_basemap = _extract("render_basemap", _VIZ_SRC)
build_coastline_traces = _extract("build_coastline_traces", _VIZ_SRC)


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


# ── render_thumbnail_gallery (click-to-zoom lightbox) ────────────────────────

class _StubDt:
    """Minimal stand-in for a pandas Timestamp (only strftime is used)."""
    def strftime(self, fmt):
        return "2017-09-06 14:00"


def _make_thumbs(n: int = 2):
    # 1 byte of "jpeg" payload is enough — the function only base64-encodes it.
    return [(f"img_{i}.jpg", b"\xff\xd8\xff", _StubDt()) for i in range(n)]


def test_render_no_full_blobs_has_no_lightbox():
    """Default behavior (no full_blobs): no checkbox / overlay markup is emitted."""
    _count, html = render_thumbnail_gallery(
        _make_thumbs(2), n_filtered=2, max_display=10,
    )
    assert 'type="checkbox"' not in html
    assert "<style>" not in html
    assert ":checked" not in html
    # Base thumbnails still render
    assert html.count("<img ") == 2


def test_render_with_full_blobs_emits_css_lightbox():
    """When full_blobs is supplied, each thumb gets a hidden checkbox + overlay label."""
    thumbs = _make_thumbs(2)
    full_blobs = [b"\xff\xd8\xff\xe0\xaa", b"\xff\xd8\xff\xe0\xbb"]
    _count, html = render_thumbnail_gallery(
        thumbs, n_filtered=2, max_display=10, full_blobs=full_blobs,
    )
    # Exactly one hidden checkbox per zoomable thumb
    assert html.count('type="checkbox"') == 2
    # CSS sibling selector drives the open state (no JS at all)
    assert ":checked ~" in html
    assert "cursor: zoom-in" in html
    assert "cursor: zoom-out" in html
    # Style block is emitted exactly once, not per-thumb
    assert html.count("<style>") == 1
    # No inline JS handlers (marimo strips these anyway)
    assert "onclick" not in html
    assert "showModal" not in html
    # Two <img> per slot (thumb + full-res inside overlay) = 4 total
    assert html.count("<img ") == 4


def test_render_lightbox_ids_are_unique_per_render():
    """Each thumb's checkbox gets a distinct id so labels toggle the right slot."""
    thumbs = _make_thumbs(3)
    full_blobs = [b"\x01", b"\x02", b"\x03"]
    _count, html = render_thumbnail_gallery(
        thumbs, n_filtered=3, max_display=10, full_blobs=full_blobs,
    )
    ids = re.findall(r'id="lb-([^"]+)"', html)
    assert len(ids) == 3
    assert len(set(ids)) == 3, f"lightbox ids should be unique: {ids}"


def test_render_mixed_full_blobs_some_none():
    """If full_blobs has a None entry, that thumb falls back to non-zoomable."""
    thumbs = _make_thumbs(2)
    full_blobs = [b"\x01", None]   # first zoomable, second not
    _count, html = render_thumbnail_gallery(
        thumbs, n_filtered=2, max_display=10, full_blobs=full_blobs,
    )
    assert html.count('type="checkbox"') == 1
    # Style block still emitted (at least one slot is zoomable)
    assert html.count("<style>") == 1


def test_render_all_full_blobs_none_emits_no_style():
    """If full_blobs is provided but every entry is None, no style block / CSS emitted."""
    thumbs = _make_thumbs(2)
    _count, html = render_thumbnail_gallery(
        thumbs, n_filtered=2, max_display=10, full_blobs=[None, None],
    )
    assert "<style>" not in html
    assert 'type="checkbox"' not in html


# ── render_thumbnail_gallery (dt formatting) ──────────────────────────────────

def test_render_shows_em_dash_for_none_dt():
    """dt=None should render an em-dash, not the literal 'None'."""
    thumbs = [("img_0.jpg", b"\xff\xd8\xff", None)]
    _count, html = render_thumbnail_gallery(thumbs, n_filtered=1, max_display=10)
    assert "—" in html
    assert ">None<" not in html


def test_render_shows_em_dash_for_nat_dt():
    """dt=pd.NaT should render an em-dash, not the literal 'NaT'."""
    import pandas as pd
    thumbs = [("img_0.jpg", b"\xff\xd8\xff", pd.NaT)]
    _count, html = render_thumbnail_gallery(thumbs, n_filtered=1, max_display=10)
    assert "—" in html
    assert "NaT" not in html


def test_render_formats_normal_datetime():
    """A real datetime should render in YYYY-MM-DD HH:MM format."""
    from datetime import datetime
    thumbs = [("img_0.jpg", b"\xff\xd8\xff", datetime(2017, 9, 6, 14, 0))]
    _count, html = render_thumbnail_gallery(thumbs, n_filtered=1, max_display=10)
    assert "2017-09-06 14:00" in html


# ── make_extent_map / render_basemap / build_coastline_traces ────────────────
# Regression tests for two longitude-convention bugs:
#  1. Collapse: cartopy's PlateCarree normalizes lon 0 and lon 360 to the same
#     point, so set_extent([0, 360, ...]) collapses to a zero-width view
#     instead of showing the full globe.
#  2. Orientation: even once uncollapsed, a default-centered (central_longitude=0)
#     projection always displays the Atlantic-centered -180..180 view regardless
#     of the requested lon_min/lon_max, so a 0-360 dataset (column 0 = lon 0,
#     e.g. Greenwich) renders with Greenwich in the *middle* instead of at the
#     left edge — misaligned with the source image's own pixel convention.

def test_make_extent_map_span_matches_requested_extent():
    """The displayed lon span (in axes-native x-units) must equal the
    requested lon_max - lon_min, regardless of a collapsed 0-360 span or of
    which central longitude the axes ends up using to display it."""
    cases = [
        (0.0, 360.0, 359.9),      # DYAMOND-style global; clamped to avoid collapse
        (-180.0, 180.0, 359.9),   # equivalent global convention
        (200.0, 210.0, 10.0),     # small region above 180
        (260.0, 330.0, 70.0),     # ERA5-style region
    ]
    for lon_min, lon_max, expected_span in cases:
        fig = make_extent_map(
            lat_min=-90, lat_max=90, lon_min=lon_min, lon_max=lon_max,
            spatial_h=4, spatial_w=4,
        )
        xlim = fig.axes[0].get_xlim()
        assert xlim[1] - xlim[0] == pytest.approx(expected_span, abs=1.0), (lon_min, lon_max)


def test_make_extent_map_0_360_places_greenwich_at_left_edge():
    """For a 0-360 dataset, lon=0 (Greenwich) must land on the left edge and
    lon=360 on the right edge — matching the source image's pixel convention
    (column 0 = lon 0) — not the middle, which is what a default
    central_longitude=0 projection would (incorrectly) show."""
    import cartopy.crs as ccrs
    fig = make_extent_map(
        lat_min=-90, lat_max=90, lon_min=0.0, lon_max=360.0,
        spatial_h=4, spatial_w=4,
    )
    ax = fig.axes[0]
    data_crs = ccrs.PlateCarree()
    xlim = ax.get_xlim()
    x_at_lon0, _ = ax.projection.transform_point(0.0, 0.0, data_crs)
    assert x_at_lon0 == pytest.approx(xlim[0], abs=1.0)


def test_make_extent_map_regional_extent_places_lon_min_at_left_edge():
    """Same orientation guarantee for a non-global regional extent."""
    import cartopy.crs as ccrs
    fig = make_extent_map(
        lat_min=10, lat_max=20, lon_min=200.0, lon_max=210.0,
        spatial_h=4, spatial_w=4,
    )
    ax = fig.axes[0]
    data_crs = ccrs.PlateCarree()
    xlim = ax.get_xlim()
    x_at_lon_min, _ = ax.projection.transform_point(200.0, 15.0, data_crs)
    assert x_at_lon_min == pytest.approx(xlim[0], abs=1.0)


def test_render_basemap_0_360_extent_not_blank():
    """A collapsed extent would render an (almost) uniform-color raster."""
    arr = render_basemap(
        lat_min=-90, lat_max=90, lon_min=0.0, lon_max=360.0, target_w=64,
    )
    # Blank/collapsed output is a single flat color (std ~ 0); a real globe
    # render has clear ocean/land/coastline contrast.
    assert arr.std() > 10


def test_build_coastline_traces_0_360_covers_both_hemispheres():
    """Western-hemisphere coastlines (native negative longitude, e.g. the
    Americas around lon -170 to -30) must wrap into the 190-330 stretch of a
    0-360 display instead of being dropped. A single stray near-180 point
    from floating-point rounding at the antimeridian doesn't count — require
    a real cluster of wrapped points, not just one."""
    traces = build_coastline_traces(
        lat_min=-90, lat_max=90, lon_min=0.0, lon_max=360.0, n_rows=14, n_cols=14,
    )
    xs = np.concatenate([np.array(t.x) for t in traces])
    assert (xs < 180).sum() > 100    # eastern hemisphere present
    assert (xs > 200).sum() > 100    # western hemisphere wrapped into view


def test_build_coastline_traces_regional_extent_pulls_fewer_lines():
    """A small regional extent should pull in far fewer coastline lines than
    a full-globe extent (some individual lines, e.g. Antarctica, span nearly
    the whole longitude range and are kept whole rather than clipped)."""
    regional = build_coastline_traces(
        lat_min=10, lat_max=20, lon_min=20.0, lon_max=30.0, n_rows=4, n_cols=4,
    )
    full_globe = build_coastline_traces(
        lat_min=-90, lat_max=90, lon_min=0.0, lon_max=360.0, n_rows=14, n_cols=14,
    )
    assert len(regional) < len(full_globe)
