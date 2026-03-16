"""Tests for the hurricane_metadata helper module."""

import importlib.util
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Direct import to avoid collision with tests/helpers/ package
_MOD_PATH = Path(__file__).parent.parent / "notebooks" / "01-prepare-data" / "helpers" / "hurricane_metadata.py"
_spec = importlib.util.spec_from_file_location("hurricane_metadata", _MOD_PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

wind_to_saffir_simpson = _mod.wind_to_saffir_simpson
filter_to_domain = _mod.filter_to_domain
build_daily_hurricane_lookup = _mod.build_daily_hurricane_lookup
enrich_image_rows = _mod.enrich_image_rows
load_ibtracs = _mod.load_ibtracs

# ── Saffir-Simpson classification ────────────────────────────────────────────

SAFFIR_SIMPSON_CASES = [
    (float("nan"), -1),  # unknown → TD
    (0, -1),             # calm → TD
    (33, -1),            # just below TS threshold
    (34, 0),             # TS lower bound
    (63, 0),             # TS upper bound
    (64, 1),             # Cat 1 lower bound
    (82, 1),             # Cat 1 upper bound
    (83, 2),             # Cat 2
    (95, 2),
    (96, 3),             # Cat 3
    (112, 3),
    (113, 4),            # Cat 4
    (136, 4),
    (137, 5),            # Cat 5 lower bound
    (155, 5),            # Cat 5
    (200, 5),            # extreme Cat 5
]


@pytest.mark.parametrize("wind_kts,expected", SAFFIR_SIMPSON_CASES)
def test_wind_to_saffir_simpson(wind_kts, expected):
    assert wind_to_saffir_simpson(wind_kts) == expected


# ── Spatial filter ───────────────────────────────────────────────────────────

def _make_ibtracs_rows(lats, lons):
    """Build a minimal IBTrACS-like DataFrame for filter testing."""
    return pd.DataFrame({
        "SID": [f"STORM{i}" for i in range(len(lats))],
        "ISO_TIME": pd.Timestamp("2017-09-06 12:00"),
        "LAT": lats,
        "LON": lons,
        "WMO_WIND": 100.0,
        "WMO_PRES": 950.0,
    })


def test_filter_to_domain_basic():
    df = _make_ibtracs_rows(
        lats=[25.0, 10.0, 25.0, 50.0],   # in, below, in, above
        lons=[-70.0, -70.0, -120.0, -70.0],  # in, in, outside west, in
    )
    result = filter_to_domain(df, lat_min=15, lat_max=35, lon_min=260, lon_max=330)
    # Only first row (lat=25, lon=-70) should pass
    assert len(result) == 1
    assert result.iloc[0]["SID"] == "STORM0"


def test_filter_to_domain_boundary_inclusive():
    """Points exactly on the boundary should be included."""
    df = _make_ibtracs_rows(
        lats=[15.0, 35.0],
        lons=[-100.0, -30.0],
    )
    result = filter_to_domain(df, lat_min=15, lat_max=35, lon_min=260, lon_max=330)
    assert len(result) == 2


# ── Daily lookup builder ────────────────────────────────────────────────────

def test_build_lookup_picks_closest_to_noon():
    """When a storm has multiple obs in a day, pick the one closest to 12Z."""
    df = pd.DataFrame({
        "SID": ["S1", "S1", "S1"],
        "ISO_TIME": pd.to_datetime([
            "2017-09-06 00:00", "2017-09-06 12:00", "2017-09-06 18:00"
        ]),
        "LAT": [20.0, 21.0, 22.0],
        "LON": [-65.0, -66.0, -67.0],
        "WMO_WIND": [100.0, 120.0, 110.0],
        "WMO_PRES": [960.0, 940.0, 950.0],
    })
    lookup = build_daily_hurricane_lookup(df)
    entry = lookup["20170906"]
    assert entry["n_storms"] == 1
    # Should have picked the 12Z observation (lat=21, wind=120)
    assert entry["max_wind_kts"] == 120.0
    assert entry["storm_lats"] == "21.0"


def test_build_lookup_prefers_obs_with_wind():
    """Among equidistant obs, prefer one with valid WMO_WIND."""
    df = pd.DataFrame({
        "SID": ["S1", "S1"],
        "ISO_TIME": pd.to_datetime([
            "2017-09-06 06:00", "2017-09-06 18:00"
        ]),
        "LAT": [20.0, 22.0],
        "LON": [-65.0, -67.0],
        "WMO_WIND": [float("nan"), 80.0],  # 06Z has no wind, 18Z has wind
        "WMO_PRES": [960.0, 950.0],
    })
    lookup = build_daily_hurricane_lookup(df)
    entry = lookup["20170906"]
    # Both are 6 hrs from noon; should prefer the one with wind data (18Z)
    assert entry["max_wind_kts"] == 80.0
    assert entry["storm_lats"] == "22.0"


def test_build_lookup_multi_storm():
    """Multiple storms on the same day produce correct aggregation."""
    df = pd.DataFrame({
        "SID": ["S1", "S2"],
        "ISO_TIME": pd.to_datetime(["2017-09-06 12:00", "2017-09-06 12:00"]),
        "LAT": [20.0, 25.0],
        "LON": [-65.0, -70.0],
        "WMO_WIND": [155.0, 80.0],
        "WMO_PRES": [920.0, 970.0],
    })
    lookup = build_daily_hurricane_lookup(df)
    entry = lookup["20170906"]
    assert entry["n_storms"] == 2
    assert entry["max_wind_kts"] == 155.0
    assert entry["max_category"] == 5
    assert "S1" in entry["storm_ids"]
    assert "S2" in entry["storm_ids"]


def test_build_lookup_empty():
    df = pd.DataFrame(columns=["SID", "ISO_TIME", "LAT", "LON", "WMO_WIND", "WMO_PRES"])
    lookup = build_daily_hurricane_lookup(df)
    assert lookup == {}


# ── Row enrichment ──────────────────────────────────────────────────────────

def test_enrich_image_rows_defaults():
    """Days not in the lookup get safe defaults."""
    lookup = {}
    dates = pd.Series([pd.Timestamp("2016-02-15"), pd.Timestamp("2016-03-01")])
    result = enrich_image_rows(dates, lookup)
    assert len(result) == 2
    assert result["hurricane_present"].tolist() == [False, False]
    assert result["n_storms"].tolist() == [0, 0]
    assert result["max_category"].tolist() == [-1, -1]
    assert result["storm_ids"].tolist() == ["", ""]


def test_enrich_image_rows_mixed():
    """Mix of storm and non-storm days."""
    lookup = {
        "20170906": {
            "hurricane_present": True,
            "n_storms": 1,
            "max_wind_kts": 155.0,
            "max_category": 5,
            "storm_ids": "S1",
            "storm_lats": "20.0",
            "storm_lons": "-65.0",
        }
    }
    dates = pd.Series([pd.Timestamp("2017-09-06"), pd.Timestamp("2017-01-01")])
    result = enrich_image_rows(dates, lookup)
    assert result.iloc[0]["hurricane_present"] == True
    assert result.iloc[0]["max_category"] == 5
    assert result.iloc[1]["hurricane_present"] == False
    assert result.iloc[1]["n_storms"] == 0


# ── IBTrACS CSV loading (integration, requires data file) ───────────────────

IBTRACS_CSV = Path("/Users/ncheruku/Documents/Work/sample_data/data/ibtracs/ibtracs.NA.list.v04r01.csv")


@pytest.mark.skipif(not IBTRACS_CSV.exists(), reason="IBTrACS CSV not available")
class TestWithIBTrACSData:
    """Integration tests that use the real IBTrACS dataset."""

    def test_load_ibtracs_shape(self):
        df = load_ibtracs(IBTRACS_CSV, "2016-01-01", "2018-12-31")
        assert len(df) > 3000
        assert df["SID"].nunique() > 40
        assert set(df.columns) >= {"SID", "ISO_TIME", "LAT", "LON", "WMO_WIND"}

    def test_full_pipeline_irma(self):
        """Irma on 2017-09-06 should show as Cat 5."""
        raw = load_ibtracs(IBTRACS_CSV, "2017-09-01", "2017-09-15")
        dom = filter_to_domain(raw, 15, 35, 260, 330)
        lookup = build_daily_hurricane_lookup(dom)
        entry = lookup.get("20170906", {})
        assert entry.get("hurricane_present") is True
        assert entry.get("max_wind_kts", 0) >= 150
        assert entry.get("max_category") == 5

    def test_full_pipeline_quiet_day(self):
        """Mid-February 2016 should have no storms."""
        raw = load_ibtracs(IBTRACS_CSV, "2016-02-01", "2016-02-28")
        dom = filter_to_domain(raw, 15, 35, 260, 330)
        lookup = build_daily_hurricane_lookup(dom)
        assert "20160215" not in lookup
