"""Orchestration: fit (seasonal + Hawkes) and forecast the final weekly tweet count.

Forecast logic at wall-clock ``now`` inside a market window [window_start, window_end]:
    final_total = n_obs (known, exact)  +  simulated tweets over (now, window_end]
We Monte-Carlo the remaining window with the Hawkes simulator, drawing a fresh bootstrapped
weekly *level* per path (level uncertainty) and seeding self-excitation from the real recent
tweets (burst momentum). Bracket probabilities are just the empirical frequencies of the
simulated final totals.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import hawkes as H
from . import windows as W
from .data import Bracket
from .intensity import IntensityModel, fit_intensity


@dataclass
class ModelFit:
    intensity: IntensityModel
    hawkes: H.HawkesParams
    now: dt.datetime
    posts: pd.DataFrame


@dataclass
class Forecast:
    samples: np.ndarray      # (n_sims,) simulated final totals
    n_obs: int               # tweets already observed in the window
    hours_elapsed: float
    hours_remaining: float
    window_start: dt.datetime
    window_end: dt.datetime

    @property
    def settled(self) -> bool:
        return self.hours_remaining <= 0

    def summary(self) -> dict:
        q = np.percentile(self.samples, [5, 25, 50, 75, 95])
        return {
            "mean": float(self.samples.mean()),
            "median": float(q[2]),
            "p5": float(q[0]), "p25": float(q[1]), "p75": float(q[3]), "p95": float(q[4]),
            "n_obs": self.n_obs,
            "hours_remaining": self.hours_remaining,
        }


def _hours(a: dt.datetime, b: dt.datetime) -> float:
    return (W.utc_ts(b) - W.utc_ts(a)).total_seconds() / 3600.0


def fit_model(
    posts: pd.DataFrame,
    now: dt.datetime,
    fit_days: float = 90.0,
    half_life_days: float = 28.0,
) -> ModelFit:
    """Estimate the seasonal intensity and Hawkes burst kernel as of ``now``."""
    intens = fit_intensity(posts, now, half_life_days=half_life_days)

    # Hawkes fit window: recent events (relative hours), seasonal background g(t)=168*shape[cell].
    start = W.utc_ts(now) - pd.Timedelta(days=fit_days)
    mask = (posts["created_at"] >= start) & (posts["created_at"] < W.utc_ts(now))
    evs = posts.loc[mask, "created_at"]
    T = _hours(start.to_pydatetime(), now)
    event_hours = (evs - start).dt.total_seconds().values / 3600.0
    g_event = 168.0 * intens.shape[W.week_cell_index(evs)]
    # g integral over [0, T]: sum seasonal weight over each integer hour-cell in the window
    g_int = float(
        sum(
            168.0 * intens.shape[W.et_cell_of_offset(start.to_pydatetime(), k)]
            for k in range(int(np.ceil(T)))
        )
    )
    hp = H.fit_hawkes(np.sort(event_hours), T, g_event[np.argsort(event_hours)], g_int)
    return ModelFit(intensity=intens, hawkes=hp, now=now, posts=posts)


def forecast(
    fit: ModelFit,
    window_start: dt.datetime,
    window_end: dt.datetime,
    n_sims: int = 20000,
    rng: np.random.Generator | None = None,
    level_prior_strength: float | None = 1.0,  # best on 16-week backtest (logloss 1.95->1.89)
) -> Forecast:
    rng = rng or np.random.default_rng(12345)
    now = fit.now
    posts = fit.posts

    eff_now = max(W.utc_ts(now), W.utc_ts(window_start))
    n_obs = int(
        (
            (posts["created_at"] >= W.utc_ts(window_start))
            & (posts["created_at"] < eff_now)
        ).sum()
    )
    hours_elapsed = _hours(window_start, eff_now.to_pydatetime())
    T_remaining = _hours(eff_now.to_pydatetime(), window_end)

    if T_remaining <= 0:  # settled
        return Forecast(
            samples=np.full(n_sims, n_obs),
            n_obs=n_obs, hours_elapsed=hours_elapsed, hours_remaining=0.0,
            window_start=window_start, window_end=window_end,
        )

    # Seasonal background per integer hour offset over the remaining window.
    now_offset = _hours(window_start, eff_now.to_pydatetime())
    n_h = int(np.ceil(T_remaining)) + 1
    g_window = np.array(
        [
            168.0 * fit.intensity.shape[
                W.et_cell_of_offset(window_start, now_offset + k)
            ]
            for k in range(n_h)
        ]
    )

    # Self-excitation momentum from real recent tweets (a few burst timescales back).
    beta = fit.hawkes.beta
    lookback_h = min(72.0, 8.0 / max(beta, 1e-3))
    seed_start = eff_now - pd.Timedelta(hours=lookback_h)
    seed_evs = posts.loc[
        (posts["created_at"] >= seed_start) & (posts["created_at"] < eff_now), "created_at"
    ]
    ages_h = (eff_now - seed_evs).dt.total_seconds().values / 3600.0
    Z0 = H.seed_decay_sum(ages_h, beta)

    # Level conditioned on the within-window pace so far (fraction of weekly seasonal mass elapsed).
    # level_prior_strength=None disables conditioning (old prior-only behavior, for A/B testing).
    if level_prior_strength is None:
        levels = fit.intensity.sample_level(rng, n_sims)
    else:
        elapsed_mass = float(
            sum(fit.intensity.shape[W.et_cell_of_offset(window_start, k)]
                for k in range(int(round(now_offset))))
        )
        levels = fit.intensity.sample_level_conditional(
            rng, n_sims, n_obs, elapsed_mass, prior_strength=level_prior_strength
        )
    remaining = H.simulate_remaining(
        rng, g_window, T_remaining, levels, fit.hawkes.alpha, beta, np.full(n_sims, Z0)
    )
    return Forecast(
        samples=n_obs + remaining,
        n_obs=n_obs, hours_elapsed=hours_elapsed, hours_remaining=T_remaining,
        window_start=window_start, window_end=window_end,
    )


def daily_forecast(
    fit: ModelFit,
    window_start: dt.datetime,
    window_end: dt.datetime,
    n_sims: int = 4000,
    rng: np.random.Generator | None = None,
) -> list[dict]:
    """Per-ET-day tweet counts over the market window: actuals for elapsed days, simulated bands
    (median, p10, p90) for the current partial day and future days.

    Returns one dict per ET calendar day with: date, actual (tweets so far that day),
    est_median/est_p10/est_p90 (that day's total), and status in {passé, partiel, futur}.
    """
    rng = rng or np.random.default_rng(7)
    posts = fit.posts
    ws, we = W.utc_ts(window_start), W.utc_ts(window_end)
    eff_now = max(W.utc_ts(fit.now), ws)
    T_remaining = (we - eff_now).total_seconds() / 3600.0

    et_dates = pd.date_range(
        ws.tz_convert(W.ET).normalize(), we.tz_convert(W.ET).normalize(), freq="D"
    ).date

    # --- simulate remaining, bucketed by ET day ---
    sim_by_date: dict = {}
    if T_remaining > 0:
        edges, date_seq, cur, cur_date = [], [], eff_now, eff_now.tz_convert(W.ET).date()
        while cur < we:
            nxt_mid = pd.Timestamp(cur_date, tz=W.ET) + pd.Timedelta(days=1)
            bnd = min(nxt_mid.tz_convert("UTC"), we)
            edges.append((bnd - eff_now).total_seconds() / 3600.0)
            date_seq.append(cur_date)
            cur, cur_date = bnd, cur_date + dt.timedelta(days=1)

        now_offset = (eff_now - ws).total_seconds() / 3600.0
        n_h = int(np.ceil(T_remaining)) + 1
        g_window = np.array([
            168.0 * fit.intensity.shape[W.et_cell_of_offset(window_start, now_offset + k)]
            for k in range(n_h)
        ])
        beta = fit.hawkes.beta
        lookback_h = min(72.0, 8.0 / max(beta, 1e-3))
        seed_evs = posts.loc[
            (posts["created_at"] >= eff_now - pd.Timedelta(hours=lookback_h))
            & (posts["created_at"] < eff_now), "created_at"
        ]
        Z0 = H.seed_decay_sum((eff_now - seed_evs).dt.total_seconds().values / 3600.0, beta)
        n_obs_win = int(((posts["created_at"] >= ws) & (posts["created_at"] < eff_now)).sum())
        elapsed_mass = float(
            sum(fit.intensity.shape[W.et_cell_of_offset(window_start, k)]
                for k in range(int(round(now_offset))))
        )
        levels = fit.intensity.sample_level_conditional(rng, n_sims, n_obs_win, elapsed_mass)
        buckets = H.simulate_remaining_daily(
            rng, g_window, T_remaining, levels, fit.hawkes.alpha, beta,
            np.full(n_sims, Z0), np.array(edges),
        )
        sim_by_date = {d: buckets[:, i] for i, d in enumerate(date_seq)}

    # --- assemble per day ---
    out = []
    for d in et_dates:
        d_start = max((pd.Timestamp(d, tz=W.ET)).tz_convert("UTC"), ws)
        d_end = min((pd.Timestamp(d, tz=W.ET) + pd.Timedelta(days=1)).tz_convert("UTC"), we)
        actual = int(
            ((posts["created_at"] >= d_start) & (posts["created_at"] < min(eff_now, d_end))).sum()
        )
        # Hours of this calendar day that fall inside the market window (clipped at noon-to-noon
        # edges). < ~23h => an edge half-day, which naturally has a lower expected total.
        window_hours = (d_end - d_start).total_seconds() / 3600.0
        half_day = window_hours < 23.0
        if d in sim_by_date:
            tot = actual + sim_by_date[d]
            status = "partiel" if d_start < eff_now < d_end else "futur"
            out.append({
                "date": d, "actual": actual, "status": status,
                "window_hours": window_hours, "half_day": half_day,
                "est_median": float(np.median(tot)),
                "est_p10": float(np.percentile(tot, 10)),
                "est_p90": float(np.percentile(tot, 90)),
            })
        else:  # fully elapsed day -> known
            out.append({
                "date": d, "actual": actual, "status": "passé",
                "window_hours": window_hours, "half_day": half_day,
                "est_median": float(actual), "est_p10": float(actual), "est_p90": float(actual),
            })
    return out


def bracket_probabilities(
    brackets: list[Bracket], samples: np.ndarray, gamma: float = 1.0
) -> list[dict]:
    """Probability mass of the simulated totals in each bracket, with market price + edge.

    ``gamma`` applies a sharpening recalibration (q ∝ p**gamma, renormalized) to correct the
    model's mild under-confidence measured in backtests. gamma=1.0 leaves probabilities raw.
    """
    n = len(samples)
    raw = []
    for b in brackets:
        hi = b.high if np.isfinite(b.high) else np.inf
        raw.append(float(((samples >= b.low) & (samples <= hi)).sum()) / n)
    probs = np.array(raw)
    if gamma != 1.0:
        q = np.power(np.clip(probs, 1e-12, 1.0), gamma)
        probs = q / q.sum()
    out = []
    for b, p in zip(brackets, probs):
        p = float(p)
        yes_price = b.yes_price
        no_price = b.no_price if b.no_price is not None else (
            None if yes_price is None else 1.0 - yes_price
        )
        # YES bet: pay yes_price, win 1 if total lands in bracket   -> edge = P(in) - yes_price
        # NO  bet: pay no_price,  win 1 if total NOT in bracket      -> edge = P(out) - no_price
        edge_yes = None if yes_price is None else p - yes_price
        edge_no = None if no_price is None else (1.0 - p) - no_price
        side, best_edge, best_price = "—", None, None
        if edge_yes is not None or edge_no is not None:
            ey = edge_yes if edge_yes is not None else -9
            en = edge_no if edge_no is not None else -9
            if max(ey, en) > 0:
                if ey >= en:
                    side, best_edge, best_price = "OUI", edge_yes, yes_price
                else:
                    side, best_edge, best_price = "NON", edge_no, no_price
        out.append(
            {
                "label": b.label, "low": b.low, "high": b.high,
                "model_prob": p, "market_price": yes_price,  # back-compat (yes price)
                "yes_price": yes_price, "no_price": no_price,
                "edge": edge_yes,            # back-compat (yes edge)
                "edge_yes": edge_yes, "edge_no": edge_no,
                "best_side": side, "best_edge": best_edge, "best_price": best_price,
            }
        )
    return out


def confidence_report(
    table: list[dict], samples: np.ndarray, hours_remaining: float | None = None
) -> dict:
    """Confidence diagnostics for the current week, built on the 'distance-to-edge' insight.

    Late-week reliability is driven less by *how many* tweets remain than by whether the projected
    total sits comfortably inside a bracket or right on a 20-wide boundary (where a single burst
    flips the outcome). We therefore report:
      * top_prob   — the model's probability on its single most-likely bracket (its raw confidence)
      * margin     — tweets between the projected (median) total and the nearest bracket boundary
      * sigma_rem  — std of the simulated final total (remaining uncertainty)
      * safety     — margin / sigma_rem (>~1.5 = comfortably inside; <~0.7 = on the edge)
      * regime     — human label: confident/edge-of-bracket/intermediate
    """
    top = max(table, key=lambda r: r["model_prob"])
    proj = float(np.median(samples))
    edges = sorted({r["low"] for r in table} | {
        r["high"] + 1 for r in table if np.isfinite(r["high"])
    })
    margin = min((abs(proj - e) for e in edges), default=float("nan"))
    sigma_rem = float(samples.std())
    safety = margin / sigma_rem if sigma_rem > 1e-9 else float("inf")
    young = hours_remaining is not None and hours_remaining > 96  # >~4 days left
    if safety >= 1.5:
        regime = "Tranche-centrée — modèle confiant, edge limité"
    elif safety <= 0.7 and young:
        regime = "Début de semaine — incertitude normale (trop tôt pour trader)"
    elif safety <= 0.7:
        regime = "Bord de tranche — incertitude exploitable (opportunité)"
    else:
        regime = "Intermédiaire"
    return {
        "top_label": top["label"], "top_prob": float(top["model_prob"]),
        "proj_total": proj, "margin": float(margin), "sigma_rem": sigma_rem,
        "safety": float(safety), "regime": regime, "young_week": bool(young),
    }
