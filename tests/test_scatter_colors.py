"""Regression tests for colour correctness in the latent-space scatter.

Two bugs, one shape: the key described something other than the picture.

Categorical -- `hd.shade` built the legend's CategoricalColorMapper from
holoviews' own default palette (ColorBrewer Set1) and ignored the `color_key` it
was handed, so the swatches named colours appearing nowhere in the raster. On a
boolean column the image drew False #d60000 / True #018700 while the legend
claimed #e41a1c / #377eb8 -- red and *blue* for a plot containing red and green.

Continuous -- three different ranges were in play at once. `shade` normalised to
the current zoom window, so colours changed as you panned; the colorbar came
from the few hundred hover-sample points; and neither was the column's own
range. Measured on max_wind_kts: column 20..155, hover sample 25..145, colorbar
25..145, raster whatever was on screen.
"""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from bokeh.models import CategoricalColorMapper

REPO_ROOT = Path(__file__).parent.parent
SCATTER = (
    REPO_ROOT / "notebooks" / "05-latent-space-exploration" / "helpers" / "scatter.py"
)


def _load_scatter():
    """Load scatter.py by path.

    Two notebook directories ship a package called `helpers` and only one can
    be live per interpreter, so this never imports the package. scatter.py has
    no module-level third-party imports, which makes that cheap.
    """
    spec = importlib.util.spec_from_file_location("_scatter_under_test", SCATTER)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


scatter = _load_scatter()


class _FakeState:
    def __init__(self, mappers):
        self._mappers = mappers

    def select(self, spec):
        return self._mappers


class _FakePlot:
    """Stands in for a holoviews plot. Only `.state.select()` is exercised."""

    def __init__(self, *mappers):
        self.state = _FakeState(list(mappers))


def _run(color_key, mapper):
    scatter._legend_color_hook(color_key)(_FakePlot(mapper), element=None)
    return dict(zip(mapper.factors, mapper.palette))


def test_boolean_legend_matches_the_raster():
    """The reported case: legend said red/blue, raster drew red/green."""
    mapper = CategoricalColorMapper(
        factors=["False", "True"], palette=["#e41a1c", "#377eb8"]
    )
    got = _run({False: "#d60000", True: "#018700"}, mapper)
    assert got == {"False": "#d60000", "True": "#018700"}


def test_integer_factors_are_matched_by_string():
    """Mapper factors are strings whatever the column dtype -- year is int16."""
    mapper = CategoricalColorMapper(
        factors=["2016", "2017", "2018"], palette=["#e41a1c", "#377eb8", "#4daf4a"]
    )
    got = _run({2016: "#d60000", 2017: "#018700", 2018: "#b500ff"}, mapper)
    assert got == {"2016": "#d60000", "2017": "#018700", "2018": "#b500ff"}


def test_noise_keeps_its_reserved_colour():
    """HDBSCAN's -1 is grey by design, not a palette entry."""
    mapper = CategoricalColorMapper(
        factors=["-1", "0", "1"], palette=["#e41a1c", "#377eb8", "#4daf4a"]
    )
    got = _run(
        {-1: scatter.NOISE_COLOR, 0: "#d60000", 1: "#018700"}, mapper
    )
    assert got["-1"] == scatter.NOISE_COLOR


def test_unknown_factors_are_left_alone():
    """Better a stale swatch than a confidently wrong one."""
    mapper = CategoricalColorMapper(
        factors=["a", "b"], palette=["#111111", "#222222"]
    )
    got = _run({"a": "#d60000"}, mapper)
    assert got == {"a": "#d60000", "b": "#222222"}


@pytest.mark.parametrize("color_key", [None, {}])
def test_no_key_is_a_no_op(color_key):
    """Density and continuous columns pass no key and must not be touched."""
    mapper = CategoricalColorMapper(
        factors=["a", "b"], palette=["#111111", "#222222"]
    )
    assert _run(color_key, mapper) == {"a": "#111111", "b": "#222222"}


def test_hook_is_wired_into_the_scatter():
    """The hook existing is not enough -- it has to be attached for categoricals."""
    source = SCATTER.read_text()
    assert "_legend_color_hook(color_key)" in source, \
        "the legend hook is defined but never appended to the plot's hooks"


# ── continuous: one span for the raster and the bar ──────────────────────────

def test_span_clips_outliers_out_of_the_ramp():
    """A long tail would otherwise spend the whole ramp on a handful of points.

    max_wind_kts is the real case: mostly low values with a thin tail. Scaling
    to the extremes leaves the bulk rendered flat.
    """
    rng = np.random.default_rng(0)
    values = pd.Series(np.concatenate([rng.uniform(0.0, 100.0, 1000), [1e6]]))
    low, high = scatter._column_span(values)

    assert high < 1_000.0, "the outlier must not set the top of the ramp"
    assert 0.0 <= low < high <= 100.0


def test_span_covers_the_bulk_of_a_normal_column():
    rng = np.random.default_rng(0)
    values = pd.Series(rng.normal(50, 10, 10_000))
    low, high = scatter._column_span(values)
    inside = values.between(low, high).mean()
    assert 0.93 <= inside <= 0.99, "p2..p98 should hold ~96% of the column"


@pytest.mark.parametrize("values", [
    pytest.param([7.0, 7.0, 7.0], id="constant"),
    pytest.param([], id="empty"),
    pytest.param([float("nan")] * 3, id="all-nan"),
    pytest.param(["a", "b"], id="non-numeric"),
])
def test_span_is_none_when_there_is_no_range(values):
    """None leaves datashader on its own handling rather than a degenerate span."""
    assert scatter._column_span(pd.Series(values)) is None


def test_span_falls_back_to_extremes_when_the_middle_is_flat():
    """Flat across the middle 96% but not overall still needs a usable range."""
    values = pd.Series([5.0] * 1000 + [0.0, 10.0])
    span = scatter._column_span(values)
    assert span is not None and span[0] < span[1]


def test_same_value_keeps_its_colour_when_the_window_changes():
    """The reported bug: colours shifted as you zoomed.

    `shade` normalises to the aggregate it is given, and that aggregate is
    rebuilt for the visible window on every zoom. The value 1.5 sits in both a
    full view (0..10) and a zoomed one (0..2); unpinned it drew dark purple in
    one and teal in the other.
    """
    import xarray as xr
    from datashader import transfer_functions as tf

    cmap = ["#440154", "#21918c", "#fde725"]

    def colour_of_middle(values, span):
        agg = xr.DataArray(
            np.array([values], dtype="float64"),
            dims=["y", "x"],
            coords={"y": [0], "x": list(range(len(values)))},
        )
        return int(np.asarray(tf.shade(agg, cmap=cmap, how="linear", span=span).data)[0][1])

    full, zoomed = [0.0, 1.5, 10.0], [0.0, 1.5, 2.0]

    # Unpinned, the same value renders differently -- this is the bug, asserted
    # so the test cannot quietly stop exercising anything.
    assert colour_of_middle(full, None) != colour_of_middle(zoomed, None)

    span = (0.0, 10.0)
    assert colour_of_middle(full, span) == colour_of_middle(zoomed, span)


def test_span_is_wired_into_both_the_raster_and_the_bar():
    """Pinning only one of them leaves the bar describing a different scale."""
    source = SCATTER.read_text()
    assert "span=_column_span" in source or "span = _column_span" in source, \
        "the continuous branch never computes a span"
    assert "span=span" in source, "shade is not given the span"
    assert '"clim": span' in source, "the colorbar is not given the same span"
