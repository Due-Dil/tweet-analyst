"""Per-market-type calibration.

Markets come in different durations (2-day, 3-day, 7-day). The default sharpening γ and the
backtest validation were done on 7-day markets, so applying them to a 2-day market extrapolates.

The continuous tweet history lets us fix this: for any duration we replay *many synthetic windows
of that length* and fit the sharpening γ that best calibrates the model for that duration. So a
2-day market is calibrated on hundreds of 2-day windows, not on the handful of real 2-day markets.

``calibrate_gamma`` runs the per-duration backtest; ``gamma_for_duration`` returns the calibrated γ
(snapping to the nearest calibrated duration), with a runtime cache the app can refresh.
"""
from __future__ import annotations

import datetime as dt

import pandas as pd

from . import backtest as BT
from . import data as D
from . import windows as W

# Calibrated γ per market duration (days), measured by calibrate_gamma over synthetic windows on a
# common recent ~112-day span (June 2026). Finding: γ is nearly duration-invariant once the regime is
# held fixed — the real driver of γ is the regime, not the market length. Use the "recalibrate" path
# to refresh these as the regime drifts.
DEFAULT_GAMMA_BY_DURATION: dict[int, float] = {2: 1.10, 3: 1.05, 7: 1.15}

# Runtime overrides (e.g. from the app's "recalibrate" button) take precedence.
_RUNTIME_GAMMA: dict[int, float] = {}


def calibrate_gamma(
    posts: pd.DataFrame,
    anchor_end: dt.datetime,
    duration_days: float,
    # Calibrate every duration over the SAME recent span so they reflect the current regime and are
    # comparable (γ is regime-sensitive: ~1.2 on recent weeks, <1 if the volatile Dec-Jan period is
    # included). lookback ≈ 16 weeks -> the 7-day result reproduces the established γ.
    lookback_days: float = 112.0,
    max_windows: int = 90,
    checkpoints: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 0.9),
    n_sims: int = 3000,
    seed: int = 11,
    level_prior_strength: float | None = 1.0,
) -> dict:
    """Fit the sharpening γ for markets of ``duration_days`` by replaying synthetic windows over the
    most recent ``lookback_days`` (so all durations share one recent regime)."""
    calib_start = anchor_end - dt.timedelta(days=lookback_days)
    hist_start = max(posts["created_at"].min().to_pydatetime(), calib_start)
    n_windows = min(max_windows, int(lookback_days / max(duration_days, 0.5)) + 1)
    grid = W.window_grid(hist_start, anchor_end, duration_days,
                         step_days=duration_days, n_windows=n_windows)
    if not grid:
        return {"duration_days": duration_days, "gamma": DEFAULT_GAMMA_BY_DURATION.get(7, 1.2),
                "ll_before": float("nan"), "ll_after": float("nan"), "n_windows": 0}
    res = BT.run_backtest(
        posts, anchor_end, checkpoints=checkpoints, n_sims=n_sims, seed=seed,
        level_prior_strength=level_prior_strength, grid=grid,
    )
    g, ll0, ll1 = BT.fit_sharpening(res.prob_matrix, res.true_idx)
    return {
        "duration_days": round(float(duration_days), 1), "gamma": round(float(g), 2),
        "ll_before": round(float(ll0), 3), "ll_after": round(float(ll1), 3),
        "n_windows": len(grid),
    }


def gamma_for_duration(duration_days: float) -> float:
    """Return the calibrated γ for a market's duration, snapping to the nearest calibrated bucket.

    Priority: persistent cache (auto-recalibrated on the current regime) > runtime override >
    built-in defaults. The bucket whose duration is closest wins (2.0-day -> 2-day γ, 6.9 -> 7-day).
    """
    table = dict(DEFAULT_GAMMA_BY_DURATION)
    for dur, info in load_calibration().items():
        table[dur] = info["gamma"]
    table.update(_RUNTIME_GAMMA)
    if not table:
        return 1.2
    nearest = min(table, key=lambda d: abs(d - duration_days))
    return table[nearest]


def set_runtime_gamma(duration_days: float, gamma: float) -> None:
    """Store a freshly-calibrated γ for a duration bucket (rounded to whole days)."""
    _RUNTIME_GAMMA[int(round(duration_days))] = float(gamma)


# --------------------------------------------------------------------------- #
# Persistent calibration cache + automatic regime-refresh
# --------------------------------------------------------------------------- #
def _calib_conn():
    con = D._conn()  # shares data/cache.db
    con.execute(
        """CREATE TABLE IF NOT EXISTS gamma_calibration(
               duration INTEGER PRIMARY KEY, gamma REAL, n_windows INTEGER,
               ll_before REAL, ll_after REAL, calibrated_at TEXT)"""
    )
    return con


def load_calibration() -> dict[int, dict]:
    """Return the persisted per-duration calibration {duration: {gamma, calibrated_at, ...}}."""
    con = _calib_conn()
    rows = con.execute(
        "SELECT duration, gamma, n_windows, ll_before, ll_after, calibrated_at FROM gamma_calibration"
    ).fetchall()
    con.close()
    return {
        int(r[0]): {"gamma": r[1], "n_windows": r[2], "ll_before": r[3],
                    "ll_after": r[4], "calibrated_at": r[5]}
        for r in rows
    }


def store_calibration(duration_days: float, result: dict) -> None:
    con = _calib_conn()
    with con:
        con.execute(
            """INSERT INTO gamma_calibration(duration, gamma, n_windows, ll_before, ll_after, calibrated_at)
               VALUES(?,?,?,?,?,?)
               ON CONFLICT(duration) DO UPDATE SET gamma=excluded.gamma, n_windows=excluded.n_windows,
                   ll_before=excluded.ll_before, ll_after=excluded.ll_after,
                   calibrated_at=excluded.calibrated_at""",
            (int(round(duration_days)), float(result["gamma"]), int(result["n_windows"]),
             float(result.get("ll_before") or 0.0), float(result.get("ll_after") or 0.0),
             dt.datetime.now(dt.timezone.utc).isoformat()),
        )
    con.close()


def _age_days(iso: str) -> float:
    return (dt.datetime.now(dt.timezone.utc)
            - pd.Timestamp(iso).tz_convert("UTC").to_pydatetime()).total_seconds() / 86400.0


def is_stale(duration_days: float, max_age_days: float = 7.0) -> bool:
    info = load_calibration().get(int(round(duration_days)))
    return info is None or _age_days(info["calibrated_at"]) > max_age_days


def auto_recalibrate(
    posts: pd.DataFrame,
    anchor_end: dt.datetime,
    durations=(2, 3, 7),
    max_age_days: float = 7.0,
    n_sims: int = 3000,
    on_start=None,
) -> list[dict]:
    """Recalibrate (and persist) γ for any duration whose cached value is missing or older than
    ``max_age_days``. Returns the list of results that were (re)calibrated. Persistence means this
    actually runs at most once per ``max_age_days`` across sessions/restarts."""
    done = []
    for dur in durations:
        if is_stale(dur, max_age_days):
            if on_start:
                on_start(dur)
            r = calibrate_gamma(posts, anchor_end, dur, n_sims=n_sims)
            store_calibration(dur, r)
            done.append(r)
    return done
