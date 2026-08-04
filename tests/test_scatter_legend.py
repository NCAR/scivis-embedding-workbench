"""Regression tests for the categorical legend in the latent-space scatter.

`hd.shade` builds the legend's CategoricalColorMapper from holoviews' own
default palette (ColorBrewer Set1) and ignores the `color_key` it is handed, so
the swatches name colours that appear nowhere in the raster. Observed on a
boolean column: the image drew False #d60000 / True #018700 while the legend
claimed #e41a1c / #377eb8 -- red and *blue* for a plot containing red and green.

`_legend_color_hook` repaints the mapper from the shared key. These tests pin
that behaviour; the hook is pure enough to check without rendering.
"""

import importlib.util
import sys
from pathlib import Path

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
