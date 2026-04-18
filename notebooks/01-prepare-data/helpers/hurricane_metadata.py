"""IBTrACS hurricane metadata helpers.

Load, spatially filter, and temporally match IBTrACS storm observations
to ERA5 image timestamps at any temporal resolution, producing per-image
hurricane columns.

Functions
---------
load_ibtracs              – read & clean the IBTrACS CSV
filter_to_domain          – spatial bounding-box filter
infer_temporal_resolution – detect freq string from image dt column
build_hurricane_lookup    – aggregate IBTrACS obs into a time-bucketed lookup dict
enrich_image_rows         – map image datetimes to hurricane column values
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Saffir-Simpson classification
# ---------------------------------------------------------------------------

def wind_to_saffir_simpson(wind_kts: float) -> int:
    """Map WMO max sustained wind (knots) to Saffir-Simpson integer code.

    Returns
    -------
    -1  Tropical Depression  (< 34 kts or unknown)
     0  Tropical Storm       (34–63 kts)
     1  Category 1           (64–82 kts)
     2  Category 2           (83–95 kts)
     3  Category 3           (96–112 kts)
     4  Category 4           (113–136 kts)
     5  Category 5           (>= 137 kts)
    """
    if math.isnan(wind_kts):
        return -1
    if wind_kts < 34:
        return -1
    if wind_kts < 64:
        return 0
    if wind_kts < 83:
        return 1
    if wind_kts < 96:
        return 2
    if wind_kts < 113:
        return 3
    if wind_kts < 137:
        return 4
    return 5


# ---------------------------------------------------------------------------
# Load & clean
# ---------------------------------------------------------------------------

def load_ibtracs(
    csv_path: str | Any,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Read an IBTrACS CSV and return a cleaned DataFrame.

    Parameters
    ----------
    csv_path : path-like
        Path to ``ibtracs.NA.list.v04r01.csv`` (or global equivalent).
    start_date, end_date : str, optional
        ISO date strings (e.g. ``"2016-01-01"``). If given, rows outside
        this range are dropped.

    Returns
    -------
    pd.DataFrame
        Columns: SID, ISO_TIME (datetime), NATURE, LAT, LON, WMO_WIND, WMO_PRES
        (numeric columns coerced to float; invalid → NaN).
    """
    df = pd.read_csv(
        csv_path,
        skiprows=[1],  # row 2 is units metadata
        low_memory=False,
    )

    df["ISO_TIME"] = pd.to_datetime(df["ISO_TIME"], errors="coerce")
    for col in ("LAT", "LON", "WMO_WIND", "WMO_PRES"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Keep useful columns only
    keep = ["SID", "ISO_TIME", "NATURE", "LAT", "LON", "WMO_WIND", "WMO_PRES"]
    df = df[[c for c in keep if c in df.columns]].copy()

    # Drop rows with missing position or time
    df.dropna(subset=["ISO_TIME", "LAT", "LON"], inplace=True)

    # Temporal filter
    if start_date is not None:
        df = df[df["ISO_TIME"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        df = df[df["ISO_TIME"] <= pd.Timestamp(end_date) + pd.Timedelta(days=1)]

    df.reset_index(drop=True, inplace=True)
    return df


# ---------------------------------------------------------------------------
# Spatial filter
# ---------------------------------------------------------------------------

def filter_to_domain(
    df: pd.DataFrame,
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> pd.DataFrame:
    """Keep only rows whose (LAT, LON) falls inside the bounding box.

    Handles the convention mismatch between ERA5 longitudes (0–360 °E)
    and IBTrACS longitudes (negative for western hemisphere).
    """
    # Convert ERA5 eastern-positive bounds to signed degrees if needed
    if lon_min > 180:
        lon_min -= 360
    if lon_max > 180:
        lon_max -= 360

    mask = (
        (df["LAT"] >= lat_min)
        & (df["LAT"] <= lat_max)
        & (df["LON"] >= lon_min)
        & (df["LON"] <= lon_max)
    )
    return df.loc[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Temporal resolution helpers
# ---------------------------------------------------------------------------

def infer_temporal_resolution(dt_series: pd.Series) -> str:
    """Infer the temporal resolution of the image dataset from its timestamps.

    Computes the median gap between consecutive sorted timestamps and maps
    it to a pandas frequency string.

    Parameters
    ----------
    dt_series : pd.Series[datetime64]
        The ``dt`` column of the image LanceDB table.

    Returns
    -------
    str
        One of ``"h"``, ``"3h"``, ``"6h"``, ``"12h"``, or ``"D"``.
    """
    ts = pd.to_datetime(dt_series).sort_values().reset_index(drop=True)
    if len(ts) < 2:
        return "D"

    median_gap = ts.diff().dropna().median()
    hours = median_gap.total_seconds() / 3600.0

    if hours <= 1:
        return "h"
    if hours <= 3:
        return "3h"
    if hours <= 6:
        return "6h"
    if hours <= 12:
        return "12h"
    return "D"


def _bucket_key(ts: pd.Timestamp, freq: str) -> str:
    """Floor *ts* to the freq bucket and return a string key."""
    bucketed = ts.floor(freq)
    if freq == "D":
        return bucketed.strftime("%Y%m%d")
    # minute-level freqs (future-proofing)
    if bucketed.minute != 0:
        return bucketed.strftime("%Y%m%d%H%M")
    return bucketed.strftime("%Y%m%d%H")


# ---------------------------------------------------------------------------
# Generic lookup builder
# ---------------------------------------------------------------------------

def build_hurricane_lookup(df: pd.DataFrame, freq: str = "D") -> dict[str, dict]:
    """Build a time-bucket-keyed lookup of aggregated hurricane columns.

    For each time bucket, selects one representative observation per storm
    (closest to the bucket center, preferring obs with valid WMO_WIND),
    then aggregates across storms.

    Parameters
    ----------
    df : pd.DataFrame
        IBTrACS data already filtered to domain and date range.
    freq : str
        Pandas frequency string controlling bucket size. Examples:
        ``"D"`` (daily), ``"6h"`` (6-hourly), ``"3h"`` (3-hourly), ``"h"`` (hourly).

    Returns
    -------
    dict[str, dict]
        ``{ "20170906": { "hurricane_present": True, ... }, ... }``  (daily)
        ``{ "2017090612": { ... }, ... }``  (sub-daily)
    """
    if df.empty:
        return {}

    df = df.copy()

    # Assign each obs to its freq bucket
    df["bucket"] = df["ISO_TIME"].apply(lambda t: _bucket_key(pd.Timestamp(t), freq))

    # Compute bucket center for "closest to center" ranking
    freq_td = pd.tseries.frequencies.to_offset(freq).nanos / 1e9  # seconds
    half_td = pd.Timedelta(seconds=freq_td / 2)

    def _bucket_center(ts: pd.Timestamp) -> pd.Timestamp:
        return ts.floor(freq) + half_td

    df["bucket_center"] = df["ISO_TIME"].apply(
        lambda t: _bucket_center(pd.Timestamp(t))
    )
    df["secs_from_center"] = (
        (df["ISO_TIME"] - df["bucket_center"]).abs().dt.total_seconds()
    )
    df["has_wind"] = df["WMO_WIND"].notna()

    # Sort so best observation per (storm, bucket) comes first:
    #   1. has valid wind data  (True > False → descending)
    #   2. closest to bucket center (ascending)
    df.sort_values(
        ["SID", "bucket", "has_wind", "secs_from_center"],
        ascending=[True, True, False, True],
        inplace=True,
    )
    # One row per storm per bucket
    df.drop_duplicates(subset=["SID", "bucket"], keep="first", inplace=True)

    lookup: dict[str, dict] = {}
    for bucket_key, grp in df.groupby("bucket"):
        max_wind = grp["WMO_WIND"].max()
        max_cat = wind_to_saffir_simpson(
            max_wind if not pd.isna(max_wind) else float("nan")
        )
        lookup[bucket_key] = {
            "hurricane_present": True,
            "n_storms": len(grp),
            "max_wind_kts": float(max_wind) if not pd.isna(max_wind) else None,
            "max_category": max_cat,
            "storm_ids": ",".join(grp["SID"].values),
            "storm_lats": ",".join(f"{x:.1f}" for x in grp["LAT"].values),
            "storm_lons": ",".join(f"{x:.1f}" for x in grp["LON"].values),
        }
    return lookup


# ---------------------------------------------------------------------------
# Row enrichment
# ---------------------------------------------------------------------------

_DEFAULTS: dict[str, Any] = {
    "hurricane_present": False,
    "n_storms": 0,
    "max_wind_kts": None,
    "max_category": -1,
    "storm_ids": "",
    "storm_lats": "",
    "storm_lons": "",
}


def enrich_image_rows(
    dt_series: pd.Series,
    lookup: dict[str, dict],
    freq: str = "D",
) -> pd.DataFrame:
    """Map a Series of image datetimes to hurricane-column values.

    Parameters
    ----------
    dt_series : pd.Series[datetime64]
        One entry per image row.
    lookup : dict
        Output of :func:`build_hurricane_lookup`.
    freq : str
        The same frequency string used when building *lookup*.

    Returns
    -------
    pd.DataFrame
        Seven columns, same length as *dt_series*.
    """
    records = []
    for dt in dt_series:
        key = _bucket_key(pd.Timestamp(dt), freq)
        records.append(lookup.get(key, _DEFAULTS))

    out = pd.DataFrame(records)
    out = out[list(_DEFAULTS.keys())]
    return out
