"""Seasonal intensity: the day-of-week x hour-of-day posting profile, recency-weighted.

This captures two of the three requested effects:
  * day-of-week  -> different volume each weekday
  * hour-of-day  -> intraday rhythm (sleep, peak hours, ...)

The third effect (bursting / batches) is layered on top by the Hawkes kernel in ``hawkes.py``.

Design
------
We separate *shape* from *level*:
  * ``shape[c]`` = expected fraction of a week's tweets landing in week-cell ``c`` (c = dow*24+hour,
    ET), summing to 1 over the 168 cells. Estimated from a recency-weighted, circularly-smoothed
    histogram so the profile reflects Elon's *current* rhythm, not last winter's.
  * ``weekly_totals`` = recent realized weekly counts (with recency weights) used to bootstrap the
    overall *level* and its uncertainty. Keeping the level uncertain is what stops the forecast
    from being overconfident.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from . import windows as W

N_CELLS = 168  # 7 days * 24 hours


@dataclass
class IntensityModel:
    shape: np.ndarray            # (168,) fractions, sum=1  -> per-week-cell share
    cell_rate: np.ndarray        # (168,) tweets/hour, recency-weighted (for heatmaps)
    weekly_totals: np.ndarray    # recent weekly counts
    weekly_weights: np.ndarray   # recency weights for those weeks (sum=1)
    mean_level: float            # recency-weighted mean weekly total
    half_life_days: float

    def heatmap(self) -> np.ndarray:
        """(7, 24) tweets/hour grid, rows=Mon..Sun, cols=hour ET."""
        return self.cell_rate.reshape(7, 24)

    def sample_level(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Bootstrap ``n`` plausible weekly levels from recent weeks (captures level uncertainty)."""
        if len(self.weekly_totals) == 0:
            return np.full(n, self.mean_level)
        idx = rng.choice(len(self.weekly_totals), size=n, p=self.weekly_weights)
        # smooth the discrete bootstrap with mild Gamma noise around each sampled week
        base = self.weekly_totals[idx].astype(float)
        return np.maximum(0.0, rng.gamma(shape=np.maximum(base, 1.0), scale=1.0))

    def sample_level_conditional(
        self, rng: np.random.Generator, n: int,
        n_obs: int, elapsed_mass: float, prior_strength: float = 0.5,
    ) -> np.ndarray:
        """Weekly level shrunk from the empirical prior toward the within-window pace.

        Starts from the **bootstrap prior** (``sample_level``), which carries the *true* week-to-week
        overdispersion (std ≈ 40, not the Poisson √mean ≈ 14 of a naïve Gamma — that under-states the
        spread and makes the forecast over-confident with little data). Each prior draw is shrunk
        toward the pace-implied level ``n_obs / elapsed_mass`` with weight ``w = e / (e + prior_strength)``
        that grows as more of the week's seasonal mass elapses. So at n_obs≈0 the spread stays
        honestly wide (the forecast is *not* falsely sharp), and late in the window it follows the
        realized pace. ``prior_strength`` is the crossover (w=½ at elapsed_mass == prior_strength).
        """
        L_prior = self.sample_level(rng, n)
        e = float(elapsed_mass)
        if e <= 1e-6 or self.mean_level <= 0:
            return L_prior
        L_obs = n_obs / e
        w = e / (e + float(prior_strength))
        return np.maximum(0.0, (1.0 - w) * L_prior + w * L_obs)


def _recency_weights(ages_days: np.ndarray, half_life_days: float) -> np.ndarray:
    w = np.exp(-np.log(2.0) * ages_days / half_life_days)
    s = w.sum()
    return w / s if s > 0 else np.full_like(w, 1.0 / len(w))


def fit_intensity(
    posts: pd.DataFrame,
    now: dt.datetime,
    half_life_days: float = 28.0,
    smooth_sigma_hours: float = 1.5,
    history_weeks: int = 30,
    level_weeks: int = 8,
    level_half_life_days: float = 14.0,
) -> IntensityModel:
    """Estimate the seasonal profile and level distribution from ``posts`` as of ``now``.

    The *shape* (day/hour profile) uses the long history (``half_life_days``) since the rhythm is
    stable. The *level* (weekly volume) is non-stationary and strongly autocorrelated (lag-1
    r≈0.67), so it is estimated from only the most recent ``level_weeks`` weeks with a shorter
    ``level_half_life_days`` — empirically this tracks the current regime far better (a ~4-6 week
    window beats a 28-day-over-30-week average by ~11% MAE) and removes most of the over-dispersion.
    """
    posts = posts[posts["created_at"] < W.utc_ts(now)]
    times = posts["created_at"]

    # ---- shape: recency-weighted histogram over 168 week-cells, circularly smoothed ----
    cells = W.week_cell_index(times)
    age_days = (W.utc_ts(now) - times).dt.total_seconds().values / 86400.0
    wts = np.exp(-np.log(2.0) * age_days / half_life_days)
    hist = np.bincount(cells, weights=wts, minlength=N_CELLS).astype(float)
    hist = gaussian_filter1d(hist, sigma=smooth_sigma_hours, mode="wrap")
    hist += 1e-9
    shape = hist / hist.sum()

    # cell_rate (tweets/hour) for display: normalize weighted counts back to a rate scale.
    # Effective sample size of weeks behind the weighted histogram:
    eff_weeks = wts.sum() / max(wts.max(), 1e-9)  # rough; only used for heatmap scaling
    total_weighted = wts.sum()
    # Convert to tweets/hour per cell: share-of-week * mean_weekly_total / 1 hour.
    # (mean level computed below; fill after.)

    # ---- level: realized weekly totals over the aligned weekly grid ----
    if len(times):
        hist_start = times.min().to_pydatetime()
    else:
        hist_start = now - dt.timedelta(days=7 * history_weeks)
    grid = W.weekly_grid(hist_start, now, history_weeks)
    grid = grid[-level_weeks:]  # level tracks only the recent regime (autocorrelated, non-stationary)
    totals, ages = [], []
    tv = times.values
    for (ws, we) in grid:
        ws_ns = W.utc_ts(ws).value
        we_ns = W.utc_ts(we).value
        n = int(((tv >= np.datetime64(ws_ns, "ns")) & (tv < np.datetime64(we_ns, "ns"))).sum())
        totals.append(n)
        mid = ws + (we - ws) / 2
        ages.append((now - mid).total_seconds() / 86400.0)
    totals = np.array(totals, dtype=float)
    ages = np.array(ages, dtype=float)
    if len(totals):
        wk_w = _recency_weights(ages, level_half_life_days)
        mean_level = float(np.sum(totals * wk_w))
    else:
        wk_w = np.array([])
        mean_level = float(len(times)) / max(eff_weeks, 1.0)

    cell_rate = shape * mean_level  # tweets per (week-cell) == tweets/hour since cells are 1h

    return IntensityModel(
        shape=shape,
        cell_rate=cell_rate,
        weekly_totals=totals,
        weekly_weights=wk_w if len(wk_w) else np.array([]),
        mean_level=mean_level,
        half_life_days=half_life_days,
    )
