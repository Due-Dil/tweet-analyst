"""High-level entrypoint shared by the CLI and the Streamlit app."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from . import calibration
from . import data as D
from . import model as M


# Sharpening factor fit on a 16-week backtest (corrects residual under-confidence). Walked down
# 1.85 -> 1.45 (shorter level window) -> 1.20 (within-window pace conditioning) as each change fixed
# more over-dispersion at its source. The app can re-fit it from the backtest panel; 1.0 = raw.
CALIBRATED_GAMMA = 1.35


@dataclass
class ForecastRun:
    market: D.MarketEvent
    window_start: dt.datetime
    window_end: dt.datetime
    now: dt.datetime
    fit: M.ModelFit
    forecast: M.Forecast
    table: list[dict]
    gamma: float = 1.0


def run_forecast(
    slug_or_url: str,
    handle: str = D.DEFAULT_HANDLE,
    now: Optional[dt.datetime] = None,
    n_sims: int = 20000,
    history_days: int = 240,
    refresh: bool = True,
    seed: int = 12345,
    gamma: float | None = None,   # None -> auto-select by market duration (calibration.gamma_for_duration)
    half_life_days: float = 28.0,  # recency half-life for the day×hour seasonal profile
    fit_days: float = 90.0,        # window of history used to fit the Hawkes burst kernel
) -> ForecastRun:
    slug = D.slug_from_url(slug_or_url)
    market = D.get_market(slug)
    window_start, window_end = D.resolve_window(slug, market, handle)

    if now is None:
        now = dt.datetime.now(dt.timezone.utc)

    if gamma is None:  # pick the γ calibrated for this market's duration
        duration_days = (window_end - window_start).total_seconds() / 86400.0
        gamma = calibration.gamma_for_duration(duration_days)

    if refresh:
        D.ensure_history(handle, days=history_days)
    posts = D.load_posts(
        handle, start=now - dt.timedelta(days=history_days), end=now
    )

    fit = M.fit_model(posts, now, fit_days=fit_days, half_life_days=half_life_days)
    fc = M.forecast(fit, window_start, window_end, n_sims=n_sims,
                    rng=np.random.default_rng(seed))
    table = M.bracket_probabilities(market.brackets, fc.samples, gamma=gamma)
    return ForecastRun(market, window_start, window_end, now, fit, fc, table, gamma=gamma)


def table_dataframe(run: ForecastRun) -> pd.DataFrame:
    df = pd.DataFrame(run.table)
    df = df.rename(
        columns={
            "label": "Tranche",
            "model_prob": "Proba modèle",
            "market_price": "Prix marché",
            "edge": "Edge",
        }
    )
    return df[["Tranche", "Proba modèle", "Prix marché", "Edge"]]
