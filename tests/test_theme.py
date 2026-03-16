"""Tests for get_theme_colors() centralized theme helper in app.py."""
import re
from pathlib import Path

# Extract just the get_theme_colors function from app.py without importing the
# full Marimo app (which has side-effects: anywidget, lancedb, etc.)
_APP_PATH = Path(__file__).parent.parent / "notebooks" / "03-dashboard-app" / "app.py"
_src = _APP_PATH.read_text()

# Pull the function source: starts at "def get_theme_colors" and ends at the
# next top-level definition (@app.function / @app.cell / def at col 0)
_match = re.search(r"(^def get_theme_colors\b.*?)(?=^@app\.|^def |\Z)", _src, re.MULTILINE | re.DOTALL)
assert _match, "get_theme_colors not found in app.py"

_ns: dict = {}
exec(_match.group(1), _ns)
get_theme_colors = _ns["get_theme_colors"]

_REQUIRED_KEYS = {
    "text", "border", "bg", "ocean", "land", "coast", "grid",
    "gallery_bg", "bar_color", "line_color", "plotly_template",
}


def test_light_theme_has_all_keys():
    assert _REQUIRED_KEYS <= set(get_theme_colors("light").keys())


def test_dark_theme_has_all_keys():
    assert _REQUIRED_KEYS <= set(get_theme_colors("dark").keys())


def test_light_dark_bg_differ():
    assert get_theme_colors("light")["bg"] != get_theme_colors("dark")["bg"]


def test_light_dark_plotly_template_differ():
    assert get_theme_colors("light")["plotly_template"] == "plotly_white"
    assert get_theme_colors("dark")["plotly_template"] == "plotly_dark"


def test_unknown_theme_defaults_to_light_colors():
    # Anything other than "dark" falls through is_dark=False
    c = get_theme_colors("system")
    assert c["plotly_template"] == "plotly_white"
    assert c["bg"] == "#ffffff"


def test_exact_coast_values():
    # Regression: coast was accidentally set to #555555 for dark in early drafts
    assert get_theme_colors("dark")["coast"] == "#aaaaaa"
    assert get_theme_colors("light")["coast"] == "#555555"


def test_gallery_bg_is_translucent_in_dark():
    # Gallery dark background is rgba, not opaque hex
    assert get_theme_colors("dark")["gallery_bg"].startswith("rgba(")
    assert get_theme_colors("light")["gallery_bg"] == "#ffffff"
