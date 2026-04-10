"""Tests for pure helper functions in parallel_ingest_images.py.

Functions are extracted via regex from the source file (same technique as test_theme.py)
to avoid the relative import `from .image_utils import ...` which would fail when the
helpers package from notebooks/02-generate-embeddings is already on sys.path.
"""
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pytest

_SRC_PATH = (
    Path(__file__).parent.parent
    / "notebooks" / "01-prepare-data" / "helpers" / "parallel_ingest_images.py"
)
_src = _SRC_PATH.read_text()

# Shared namespace for all exec'd fragments
_NS = {
    "Path": Path,
    "List": List,
    "Optional": Optional,
    "datetime": datetime,
}

# Extract IMAGE_EXTENSIONS constant
_ext_match = re.search(r"^IMAGE_EXTENSIONS\s*=\s*\[.*?\]", _src, re.MULTILINE)
assert _ext_match, "IMAGE_EXTENSIONS not found"
exec(_ext_match.group(0), _NS)


def _extract_fn(name: str):
    m = re.search(
        rf"(^def {name}\b.*?)(?=^def |\Z)",
        _src,
        re.MULTILINE | re.DOTALL,
    )
    assert m, f"{name} not found in source"
    exec(m.group(1), _NS)
    return _NS[name]


is_image_file         = _extract_fn("is_image_file")
list_images_flat      = _extract_fn("list_images_flat")
parse_dt_from_filename = _extract_fn("parse_dt_from_filename")


# ── is_image_file ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("name,expected", [
    ("photo.jpg",    True),
    ("photo.JPEG",   True),
    ("image.png",    True),
    ("image.PNG",    True),
    ("image.tif",    True),
    ("image.tiff",   True),
    ("image.bmp",    True),
    ("image.webp",   True),
    ("doc.pdf",      False),
    ("data.csv",     False),
    ("no_extension", False),
    (".hiddenfile",  False),
])
def test_is_image_file(name, expected):
    assert is_image_file(Path(name)) == expected


# ── list_images_flat ──────────────────────────────────────────────────────────

def test_list_images_flat_returns_sorted(tmp_path):
    (tmp_path / "b.jpg").touch()
    (tmp_path / "a.png").touch()
    (tmp_path / "data.csv").touch()
    result = list_images_flat(tmp_path)
    assert [p.name for p in result] == ["a.png", "b.jpg"]


def test_list_images_flat_empty_dir(tmp_path):
    assert list_images_flat(tmp_path) == []


def test_list_images_flat_no_images(tmp_path):
    (tmp_path / "readme.txt").touch()
    (tmp_path / "data.csv").touch()
    assert list_images_flat(tmp_path) == []


def test_list_images_flat_ignores_subdirs(tmp_path):
    """list_images_flat is not recursive — files in subdirs are ignored."""
    sub = tmp_path / "subdir"
    sub.mkdir()
    (sub / "nested.jpg").touch()    # inside subdir — should be ignored
    (tmp_path / "top.jpg").touch()  # top-level — should be found
    result = list_images_flat(tmp_path)
    assert len(result) == 1
    assert result[0].name == "top.jpg"


def test_list_images_flat_multiple_extensions(tmp_path):
    for name in ["a.png", "b.tif", "c.jpeg", "d.bmp", "e.webp"]:
        (tmp_path / name).touch()
    result = list_images_flat(tmp_path)
    assert len(result) == 5


# ── parse_dt_from_filename ────────────────────────────────────────────────────

def test_parse_dt_known_format():
    dt = parse_dt_from_filename("20170906_rgb.jpeg", "%Y%m%d_rgb.jpeg")
    assert dt.year == 2017
    assert dt.month == 9
    assert dt.day == 6


def test_parse_dt_wrong_format_raises():
    with pytest.raises(ValueError, match="Failed to parse"):
        parse_dt_from_filename("notadate.jpg", "%Y%m%d_rgb.jpeg")


def test_parse_dt_with_time_component():
    dt = parse_dt_from_filename("20170906_1200_rgb.jpeg", "%Y%m%d_%H%M_rgb.jpeg")
    assert dt.hour == 12
    assert dt.minute == 0


def test_parse_dt_different_date():
    dt = parse_dt_from_filename("20161231_rgb.jpeg", "%Y%m%d_rgb.jpeg")
    assert dt.year == 2016
    assert dt.month == 12
    assert dt.day == 31
