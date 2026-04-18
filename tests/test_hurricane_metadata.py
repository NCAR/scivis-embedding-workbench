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

wind_to_saffir_simpson    = _mod.wind_to_saffir_simpson
filter_to_domain          = _mod.filter_to_domain
build_hurricane_lookup    = _mod.build_hurricane_lookup
infer_temporal_resolution = _mod.infer_temporal_resolution
enrich_image_rows         = _mod.enrich_image_rows
load_ibtracs              = _mod.load_ibtracs
_bucket_key               = _mod._bucket_key


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

def _make_ibtracs_rows(lats, lons, times=None, winds=None):
    """Build a minimal IBTrACS-like DataFrame for testing."""
    n = len(lats)
    return pd.DataFrame({
        "SID": [f"STORM{i}" for i in range(n)],
        "ISO_TIME": times if times is not None else [pd.Timestamp("2017-09-06 12:00")] * n,
        "LAT": lats,
        "LON": lons,
        "WMO_WIND": winds if winds is not None else [100.0] * n,
        "WMO_PRES": [950.0] * n,
    })


def test_filter_to_domain_basic():
    df = _make_ibtracs_rows(
        lats=[25.0, 10.0, 25.0, 50.0],
        lons=[-70.0, -70.0, -120.0, -70.0],
    )
    result = filter_to_domain(df, lat_min=15, lat_max=35, lon_min=260, lon_max=330)
    assert len(result) == 1
    assert result.iloc[0]["SID"] == "STORM0"


def test_filter_to_domain_boundary_inclusive():
    """Points exactly on the boundary should be included."""
    df = _make_ibtracs_rows(lats=[15.0, 35.0], lons=[-100.0, -30.0])
    result = filter_to_domain(df, lat_min=15, lat_max=35, lon_min=260, lon_max=330)
    assert len(result) == 2


# ── _bucket_key ──────────────────────────────────────────────────────────────

@pytest.mark.parametrize("ts_str,freq,expected", [
    ("2017-09-06 14:00", "D",  "20170906"),
    ("2017-09-06 14:00", "3h", "2017090612"),  # floors to 12Z
    ("2017-09-06 14:30", "h",  "2017090614"),  # floors to 14Z
    ("2017-09-06 00:00", "6h", "2017090600"),
    ("2017-09-06 07:00", "6h", "2017090606"),  # floors to 06Z
])
def test_bucket_key(ts_str, freq, expected):
    ts = pd.Timestamp(ts_str)
    assert _bucket_key(ts, freq) == expected


# ── infer_temporal_resolution ────────────────────────────────────────────────

@pytest.mark.parametrize("freq_str,expected", [
    ("h",   "h"),
    ("3h",  "3h"),
    ("6h",  "6h"),
    ("12h", "12h"),
    ("D",   "D"),
])
def test_infer_temporal_resolution(freq_str, expected):
    ts = pd.Series(pd.date_range("2017-01-01", periods=10, freq=freq_str))
    assert infer_temporal_resolution(ts) == expected


def test_infer_temporal_resolution_single_entry():
    """A single timestamp cannot determine a gap — defaults to daily."""
    ts = pd.Series([pd.Timestamp("2017-01-01")])
    assert infer_temporal_resolution(ts) == "D"


def test_infer_temporal_resolution_unsorted():
    """Resolution is computed correctly even on unsorted input."""
    ts = pd.Series([
        pd.Timestamp("2017-01-01 06:00"),
        pd.Timestamp("2017-01-01 00:00"),
        pd.Timestamp("2017-01-01 12:00"),
        pd.Timestamp("2017-01-01 03:00"),
        pd.Timestamp("2017-01-01 09:00"),
    ])
    assert infer_temporal_resolution(ts) == "3h"


# ── build_hurricane_lookup ───────────────────────────────────────────────────

def test_build_lookup_daily_key_format():
    """freq='D' produces %Y%m%d keys."""
    df = _make_ibtracs_rows(lats=[20.0], lons=[-65.0])
    lookup = build_hurricane_lookup(df, freq="D")
    assert "20170906" in lookup


def test_build_lookup_subdaily_key_format():
    """freq='3h' produces %Y%m%d%H keys."""
    df = _make_ibtracs_rows(lats=[20.0], lons=[-65.0])
    lookup = build_hurricane_lookup(df, freq="3h")
    # 12Z obs floors to 12Z bucket → key "2017090612"
    assert "2017090612" in lookup
    assert "20170906" not in lookup


def test_build_lookup_picks_closest_to_bucket_center():
    """For daily freq, bucket center is 12Z — obs at 12Z should be chosen."""
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
    lookup = build_hurricane_lookup(df, freq="D")
    entry = lookup["20170906"]
    assert entry["n_storms"] == 1
    assert entry["max_wind_kts"] == 120.0
    assert entry["storm_lats"] == "21.0"


def test_build_lookup_subdaily_picks_closest_to_bucket_center():
    """For 3h freq, pick obs closest to bucket center (not 12Z)."""
    # 3h bucket 09Z-12Z: center = 10:30Z
    # obs at 09:00Z → 1.5h from center; obs at 11:00Z → 0.5h from center
    df = pd.DataFrame({
        "SID": ["S1", "S1"],
        "ISO_TIME": pd.to_datetime([
            "2017-09-06 09:00", "2017-09-06 11:00"
        ]),
        "LAT": [20.0, 21.0],
        "LON": [-65.0, -66.0],
        "WMO_WIND": [100.0, 120.0],
        "WMO_PRES": [960.0, 950.0],
    })
    lookup = build_hurricane_lookup(df, freq="3h")
    entry = lookup["2017090609"]
    assert entry["storm_lats"] == "21.0"  # 11Z obs is closer to 10:30Z center


def test_build_lookup_prefers_obs_with_wind():
    """Among equidistant obs, prefer one with valid WMO_WIND."""
    df = pd.DataFrame({
        "SID": ["S1", "S1"],
        "ISO_TIME": pd.to_datetime([
            "2017-09-06 06:00", "2017-09-06 18:00"
        ]),
        "LAT": [20.0, 22.0],
        "LON": [-65.0, -67.0],
        "WMO_WIND": [float("nan"), 80.0],
        "WMO_PRES": [960.0, 950.0],
    })
    lookup = build_hurricane_lookup(df, freq="D")
    entry = lookup["20170906"]
    assert entry["max_wind_kts"] == 80.0
    assert entry["storm_lats"] == "22.0"


def test_build_lookup_multi_storm():
    """Multiple storms in the same bucket produce correct aggregation."""
    df = pd.DataFrame({
        "SID": ["S1", "S2"],
        "ISO_TIME": pd.to_datetime(["2017-09-06 12:00", "2017-09-06 12:00"]),
        "LAT": [20.0, 25.0],
        "LON": [-65.0, -70.0],
        "WMO_WIND": [155.0, 80.0],
        "WMO_PRES": [920.0, 970.0],
    })
    lookup = build_hurricane_lookup(df, freq="D")
    entry = lookup["20170906"]
    assert entry["n_storms"] == 2
    assert entry["max_wind_kts"] == 155.0
    assert entry["max_category"] == 5
    assert "S1" in entry["storm_ids"]
    assert "S2" in entry["storm_ids"]


def test_build_lookup_no_wind():
    """Storm with no WMO_WIND produces max_wind_kts=None and max_category=-1."""
    df = _make_ibtracs_rows(lats=[20.0], lons=[-65.0], winds=[float("nan")])
    lookup = build_hurricane_lookup(df, freq="D")
    entry = lookup["20170906"]
    assert entry["hurricane_present"] is True
    assert entry["max_wind_kts"] is None
    assert entry["max_category"] == -1


def test_build_lookup_empty():
    df = pd.DataFrame(columns=["SID", "ISO_TIME", "LAT", "LON", "WMO_WIND", "WMO_PRES"])
    assert build_hurricane_lookup(df) == {}


# ── enrich_image_rows ────────────────────────────────────────────────────────

def test_enrich_image_rows_defaults():
    """Timestamps not in lookup get safe defaults."""
    dates = pd.Series([pd.Timestamp("2016-02-15"), pd.Timestamp("2016-03-01")])
    result = enrich_image_rows(dates, lookup={}, freq="D")
    assert result["hurricane_present"].tolist() == [False, False]
    assert result["n_storms"].tolist() == [0, 0]
    assert result["max_category"].tolist() == [-1, -1]
    assert result["storm_ids"].tolist() == ["", ""]


def test_enrich_image_rows_daily_mixed():
    """Mix of storm and non-storm days with freq='D'."""
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
    result = enrich_image_rows(dates, lookup, freq="D")
    assert result.iloc[0]["hurricane_present"] == True
    assert result.iloc[0]["max_category"] == 5
    assert result.iloc[1]["hurricane_present"] == False
    assert result.iloc[1]["n_storms"] == 0


def test_enrich_image_rows_same_day_different_buckets():
    """With freq='3h', two images on the same calendar day but different
    buckets get independent labels."""
    lookup = {
        "2017090612": {
            "hurricane_present": True,
            "n_storms": 1,
            "max_wind_kts": 120.0,
            "max_category": 4,
            "storm_ids": "S1",
            "storm_lats": "22.0",
            "storm_lons": "-66.0",
        }
    }
    dates = pd.Series([
        pd.Timestamp("2017-09-06 12:30"),  # → bucket "2017090612" → storm
        pd.Timestamp("2017-09-06 09:00"),  # → bucket "2017090609" → no storm
    ])
    result = enrich_image_rows(dates, lookup, freq="3h")
    assert result.iloc[0]["hurricane_present"] == True
    assert result.iloc[1]["hurricane_present"] == False


def test_enrich_image_rows_daily_same_day_same_label():
    """With freq='D', two images on the same calendar day get the same label."""
    lookup = {
        "20170906": {
            "hurricane_present": True,
            "n_storms": 1,
            "max_wind_kts": 120.0,
            "max_category": 4,
            "storm_ids": "S1",
            "storm_lats": "22.0",
            "storm_lons": "-66.0",
        }
    }
    dates = pd.Series([
        pd.Timestamp("2017-09-06 00:00"),
        pd.Timestamp("2017-09-06 18:00"),
    ])
    result = enrich_image_rows(dates, lookup, freq="D")
    assert result.iloc[0]["hurricane_present"] == True
    assert result.iloc[1]["hurricane_present"] == True


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

    def test_full_pipeline_irma_daily(self):
        """Irma on 2017-09-06 should show as Cat 5 with daily resolution."""
        raw = load_ibtracs(IBTRACS_CSV, "2017-09-01", "2017-09-15")
        dom = filter_to_domain(raw, 15, 35, 260, 330)
        lookup = build_hurricane_lookup(dom, freq="D")
        entry = lookup.get("20170906", {})
        assert entry.get("hurricane_present") is True
        assert entry.get("max_wind_kts", 0) >= 150
        assert entry.get("max_category") == 5

    def test_full_pipeline_irma_subdaily(self):
        """Irma at 12Z on 2017-09-06 should also show as Cat 5 with 6h resolution."""
        raw = load_ibtracs(IBTRACS_CSV, "2017-09-01", "2017-09-15")
        dom = filter_to_domain(raw, 15, 35, 260, 330)
        lookup = build_hurricane_lookup(dom, freq="6h")
        entry = lookup.get("2017090612", {})
        assert entry.get("hurricane_present") is True
        assert entry.get("max_category") == 5

    def test_full_pipeline_quiet_day(self):
        """Mid-February 2016 should have no storms."""
        raw = load_ibtracs(IBTRACS_CSV, "2016-02-01", "2016-02-28")
        dom = filter_to_domain(raw, 15, 35, 260, 330)
        lookup = build_hurricane_lookup(dom, freq="D")
        assert "20160215" not in lookup
