"""Time helpers: ET-local calendar features and the weekly window grid used for backtests.

Markets run on US Eastern wall-clock (Friday noon ET -> next Friday noon ET). Using
``America/New_York`` keeps the day-of-week / hour features aligned with how Elon actually
experiences the day and absorbs DST automatically.
"""
from __future__ import annotations

import datetime as dt
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

ET = ZoneInfo("America/New_York")
HOURS_PER_WEEK = 24 * 7


def utc_ts(x) -> pd.Timestamp:
    """Coerce any datetime/Timestamp (naive or tz-aware) to a UTC-aware ``pd.Timestamp``."""
    t = pd.Timestamp(x)
    return t.tz_localize("UTC") if t.tz is None else t.tz_convert("UTC")


def to_et(s: pd.Series | pd.DatetimeIndex) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(s)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    return idx.tz_convert(ET)


def dow_hour(ts_utc: pd.Series) -> pd.DataFrame:
    """Return day-of-week (0=Mon) and hour-of-day (0-23) in ET for a UTC timestamp series."""
    et = to_et(ts_utc)
    return pd.DataFrame({"dow": et.dayofweek, "hour": et.hour}, index=ts_utc.index)


def week_cell_index(ts_utc: pd.Series) -> np.ndarray:
    """Map timestamps to an index 0..167 over the week (dow*24 + hour), in ET."""
    et = to_et(ts_utc)
    return (et.dayofweek.values * 24 + et.hour.values).astype(int)


def hours_since(times_utc: np.ndarray, ref_utc: dt.datetime) -> np.ndarray:
    """Float hours of each timestamp relative to ``ref_utc`` (can be negative)."""
    ref = pd.Timestamp(ref_utc).tz_convert("UTC") if pd.Timestamp(ref_utc).tz else pd.Timestamp(
        ref_utc, tz="UTC"
    )
    t = pd.DatetimeIndex(times_utc)
    if t.tz is None:
        t = t.tz_localize("UTC")
    return (t.tz_convert("UTC").asi8 - ref.value) / 3.6e12  # ns -> hours


def et_cell_of_offset(window_start_utc: dt.datetime, hours_offset: float) -> int:
    """Week cell (dow*24+hour, ET) for a point ``hours_offset`` hours after window start."""
    t = utc_ts(window_start_utc) + pd.Timedelta(hours=hours_offset)
    et = t.tz_convert(ET)
    return int(et.dayofweek) * 24 + int(et.hour)


def window_grid(
    history_start_utc: dt.datetime,
    anchor_end_utc: dt.datetime,
    length_days: float,
    step_days: float,
    n_windows: int,
) -> list[tuple[dt.datetime, dt.datetime]]:
    """Generate up to ``n_windows`` past windows of ``length_days``, stepping back by ``step_days``
    (DST-safe via ET local time). Used to calibrate parameters per market *duration* by replaying
    many synthetic windows of that length on the continuous tweet history.
    """
    out: list[tuple[dt.datetime, dt.datetime]] = []
    end_et = utc_ts(anchor_end_utc).tz_convert(ET)
    for k in range(n_windows):
        w_end_et = end_et - pd.Timedelta(days=step_days * k)
        w_start_et = w_end_et - pd.Timedelta(days=length_days)
        w_start = w_start_et.tz_convert("UTC").to_pydatetime()
        w_end = w_end_et.tz_convert("UTC").to_pydatetime()
        if w_start < history_start_utc:
            break
        out.append((w_start, w_end))
    return list(reversed(out))


def weekly_grid(
    history_start_utc: dt.datetime,
    anchor_end_utc: dt.datetime,
    n_weeks: int,
) -> list[tuple[dt.datetime, dt.datetime]]:
    """Generate up to ``n_weeks`` past 7-day windows ending at the same ET wall-clock as anchor.

    Walks backward from ``anchor_end_utc`` in 7-day steps (DST-safe via ET local time),
    keeping windows fully inside the available history.
    """
    out: list[tuple[dt.datetime, dt.datetime]] = []
    end_et = utc_ts(anchor_end_utc).tz_convert(ET)
    for k in range(1, n_weeks + 1):
        w_end_et = end_et - pd.Timedelta(weeks=k)
        w_start_et = w_end_et - pd.Timedelta(weeks=1)
        w_start = w_start_et.tz_convert("UTC").to_pydatetime()
        w_end = w_end_et.tz_convert("UTC").to_pydatetime()
        if w_start < history_start_utc:
            break
        out.append((w_start, w_end))
    return list(reversed(out))
