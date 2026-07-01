"""Goodness-of-fit DIAGNOSTICS for one market: where does each *realized* feature of Elon's tweet
stream sit inside the model's own predictive distribution — central (normal) or in a tail (extreme)?

The model (seasonal intensity + Hawkes burst kernel) is fit as of the window OPEN (only pre-window
data). We then Monte-Carlo the *full* window many times, recording event TIMES (not just counts), and
compute the same battery of statistics on every simulated path and on the realized path:

  * activité globale        — total tweets in the window
  * tweets par jour (ET)    — per-ET-day counts                 (vector, with model band)
  * tweets par heure (ET)   — counts by ET hour-of-day          (vector, with model band)
  * nombre de bursts        — # clusters (gap-based, gap < gap_mult/β)
  * taille des bursts       — mean / max tweets per cluster
  * durée des bursts        — median cluster span (minutes)
  * fraction en burst       — share of tweets inside a cluster  (burstiness)
  * timing des bursts (ET)  — burst-start counts by ET hour-of-day (vector, with model band)

Each realized value is placed at its **percentile** within the simulated distribution; a verdict
(normal / limite / extrême + direction) flags tail behaviour. This is the interactive, per-parameter
generalization of ``analyze_model_fit.py`` (which gave point KS/percentile checks).
"""
from __future__ import annotations

import dataclasses
import datetime as dt

import numpy as np
import pandas as pd

from . import hawkes as H
from . import windows as W
from .model import ModelFit


# --------------------------------------------------------------------------- #
# Event-time path simulator (full window, fit as of OPEN) — mirrors model.forecast() setup but
# records the simulated event TIMES so we can recompute burst statistics on each path.
# --------------------------------------------------------------------------- #
def simulate_window_paths(
    fit: ModelFit,
    window_start: dt.datetime,
    window_end: dt.datetime,
    n_sims: int = 2000,
    rng: np.random.Generator | None = None,
    level_lookback_days: float = 3.0,
    prior_strength: float = 0.5,
    seed_lookback_h: float = 72.0,
) -> list[np.ndarray]:
    """Return ``n_sims`` arrays of event times (hours from ``window_start``), simulating the whole
    window from open under the model's predictive distribution at open. Level is conditioned on the
    recent continuous stream and burst momentum is seeded from real pre-window tweets, exactly as the
    deployed ``model.forecast`` does, so the diagnostic compares the realized path against what the
    model actually predicted at open."""
    rng = rng or np.random.default_rng(7)
    posts = fit.posts
    ws, we = W.utc_ts(window_start), W.utc_ts(window_end)
    T = (we - ws).total_seconds() / 3600.0
    alpha, beta = fit.hawkes.alpha, fit.hawkes.beta

    # Seasonal background weight per integer hour offset over the window.
    n_h = int(np.ceil(T)) + 1
    g_window = np.array([
        168.0 * fit.intensity.shape[W.et_cell_of_offset(window_start, k)] for k in range(n_h)
    ])

    # Level conditioned on the recent continuous stream up to open (window-agnostic).
    lb_h = level_lookback_days * 24.0
    lb_ref = (ws - pd.Timedelta(hours=lb_h)).to_pydatetime()
    cond_count = int(((posts["created_at"] >= W.utc_ts(lb_ref)) & (posts["created_at"] < ws)).sum())
    cond_mass = float(sum(fit.intensity.shape[W.et_cell_of_offset(lb_ref, k)]
                          for k in range(int(round(lb_h)))))
    levels = fit.intensity.sample_level_conditional(rng, n_sims, cond_count, cond_mass,
                                                    prior_strength=prior_strength)

    # Burst momentum at open from real pre-window tweets.
    seed_evs = posts.loc[(posts["created_at"] >= ws - pd.Timedelta(hours=seed_lookback_h))
                         & (posts["created_at"] < ws), "created_at"]
    Z0 = H.seed_decay_sum((ws - seed_evs).dt.total_seconds().values / 3600.0, beta)

    Gmax = float(g_window.max()) if g_window.size else 1.0
    mu_scale = np.maximum(levels * (1.0 - alpha) / 168.0, 0.0)
    ab = alpha * beta
    paths: list[np.ndarray] = []
    for s in range(n_sims):
        ms = mu_scale[s]
        bg_max = ms * Gmax            # background upper bound over the window for this sim
        t, Z = 0.0, float(Z0)
        ev: list[float] = []
        # Ogata thinning, identical to hawkes.simulate_remaining but we keep the accepted times.
        while True:
            lam_bar = bg_max + ab * Z + 1e-12
            w = rng.exponential(1.0 / lam_bar)
            t_new = t + w
            if t_new >= T:
                break
            Z *= np.exp(-beta * w)               # decay self-excitation to the proposal time
            hour_idx = int(t_new)
            g_now = g_window[hour_idx] if hour_idx < g_window.size else g_window[-1]
            lam_true = ms * g_now + ab * Z
            if rng.random() <= lam_true / lam_bar:
                ev.append(t_new)
                Z += 1.0                         # accepted event raises future intensity
            t = t_new
        paths.append(np.array(ev, dtype=float))
    return paths


# --------------------------------------------------------------------------- #
# Burst detection (gap-based clustering) and per-path statistics
# --------------------------------------------------------------------------- #
def burst_clusters(ev_h: np.ndarray, beta: float, gap_mult: float = 3.0,
                   min_size: int = 2) -> list[tuple[float, float, int]]:
    """Split a sorted event-time array into clusters: a new cluster starts whenever the gap since the
    previous event exceeds ``gap_mult / β`` hours (a few burst timescales). Returns clusters of at
    least ``min_size`` events as ``(start_h, end_h, size)``. The same rule is applied to realized and
    simulated paths so the burst statistics are comparable."""
    ev = np.sort(np.asarray(ev_h, dtype=float))
    if ev.size == 0:
        return []
    gap_thr = gap_mult / beta if beta > 0 else float("inf")
    clusters, start, prev, size = [], ev[0], ev[0], 1
    for t in ev[1:]:
        if t - prev <= gap_thr:
            size += 1
        else:
            clusters.append((start, prev, size))
            start, size = t, 1
        prev = t
    clusters.append((start, prev, size))
    return [c for c in clusters if c[2] >= min_size]


def _et_hour_hist(times_utc: pd.DatetimeIndex) -> np.ndarray:
    """Counts by ET hour-of-day (length 24)."""
    if len(times_utc) == 0:
        return np.zeros(24)
    h = times_utc.tz_convert(W.ET).hour.values
    return np.bincount(h, minlength=24).astype(float)


def path_stats(ev_h: np.ndarray, window_start: dt.datetime, window_end: dt.datetime,
               beta: float, gap_mult: float = 3.0) -> dict:
    """Battery of statistics for one path of in-window event times (hours from window_start)."""
    ws = W.utc_ts(window_start)
    we = W.utc_ts(window_end)
    ev = np.sort(np.asarray(ev_h, dtype=float))
    times = ws + pd.to_timedelta(ev, unit="h")
    times = pd.DatetimeIndex(times)

    clusters = burst_clusters(ev, beta, gap_mult=gap_mult)
    sizes = np.array([c[2] for c in clusters], dtype=float)
    durs_min = np.array([(c[1] - c[0]) * 60.0 for c in clusters], dtype=float)
    n_in_bursts = float(sizes.sum())

    # per ET calendar day within the window
    et_dates = pd.date_range(ws.tz_convert(W.ET).normalize(),
                             we.tz_convert(W.ET).normalize(), freq="D").date
    et_day = times.tz_convert(W.ET).date if len(times) else np.array([])
    per_day = np.array([int(np.sum(et_day == d)) for d in et_dates], dtype=float)

    # burst-start times by ET hour-of-day
    burst_starts = ws + pd.to_timedelta([c[0] for c in clusters], unit="h")
    burst_starts = pd.DatetimeIndex(burst_starts) if len(clusters) else pd.DatetimeIndex([])

    return {
        "total": float(ev.size),
        "per_day": per_day,                                   # vector (len = #ET days)
        "per_hour": _et_hour_hist(times),                     # vector (24,)
        "n_bursts": float(len(clusters)),
        "mean_burst_size": float(sizes.mean()) if sizes.size else 0.0,
        "max_burst_size": float(sizes.max()) if sizes.size else 0.0,
        "median_burst_dur": float(np.median(durs_min)) if durs_min.size else 0.0,
        "burst_fraction": n_in_bursts / ev.size if ev.size else 0.0,
        "burst_hour": _et_hour_hist(burst_starts),            # vector (24,)
        "_et_dates": list(et_dates),
    }


# --------------------------------------------------------------------------- #
# Diagnose: place each realized statistic inside the simulated distribution
# --------------------------------------------------------------------------- #
_SCALARS = [
    ("total", "Activité globale (tweets)", "haut = plus prolifique que prévu"),
    ("n_bursts", "Nombre de bursts", "haut = plus de salves que prévu"),
    ("mean_burst_size", "Taille moyenne d'un burst", "haut = salves plus grosses"),
    ("max_burst_size", "Plus gros burst", "haut = salve record"),
    ("median_burst_dur", "Durée médiane d'un burst (min)", "haut = salves plus étalées"),
    ("burst_fraction", "Part des tweets en burst", "haut = plus grégaire/clusterisé"),
]


def _verdict(pct: float) -> tuple[str, str]:
    """(label, direction) from a percentile in [0,1]."""
    direction = "haut" if pct >= 0.5 else "bas"
    if pct < 0.05 or pct > 0.95:
        return "extrême", direction
    if pct < 0.15 or pct > 0.85:
        return "limite", direction
    return "normal", direction


def _scalar_row(name: str, label: str, hint: str, realized: float, sims: np.ndarray) -> dict:
    sims = np.asarray(sims, dtype=float)
    # percentile of realized within the simulated distribution (mid-rank for ties)
    pct = float((np.sum(sims < realized) + 0.5 * np.sum(sims == realized)) / max(len(sims), 1))
    label_v, direction = _verdict(pct)
    return {
        "key": name, "param": label, "hint": hint, "realized": float(realized),
        "p5": float(np.percentile(sims, 5)), "p50": float(np.percentile(sims, 50)),
        "p95": float(np.percentile(sims, 95)), "pct": pct,
        "verdict": label_v, "direction": direction, "_sims": sims,
    }


def _band(realized_vec: np.ndarray, sim_vecs: list[np.ndarray]) -> dict:
    """Per-bin realized vs model median and 5–95 band, plus a per-bin percentile."""
    arr = np.array([v for v in sim_vecs if len(v) == len(realized_vec)], dtype=float)
    realized_vec = np.asarray(realized_vec, dtype=float)
    p5 = np.percentile(arr, 5, axis=0)
    p50 = np.percentile(arr, 50, axis=0)
    p95 = np.percentile(arr, 95, axis=0)
    pct = (np.sum(arr < realized_vec, axis=0) + 0.5 * np.sum(arr == realized_vec, axis=0)) / len(arr)
    return {"realized": realized_vec, "p5": p5, "p50": p50, "p95": p95, "pct": pct}


def diagnose(
    fit: ModelFit,
    window_start: dt.datetime,
    window_end: dt.datetime,
    n_sims: int = 2000,
    rng: np.random.Generator | None = None,
    gap_mult: float = 3.0,
    eval_end: dt.datetime | None = None,
) -> dict:
    """Market-vs-model diagnostic. Fits/simulates the full window from open and situates the realized
    tweet stream against the model's predictive distribution for every feature.

    For an ONGOING market pass ``eval_end=now``: paths are still simulated over the whole window (so
    the seed/level are right), but statistics on both the realized and simulated paths are evaluated
    only over the elapsed portion ``[open, eval_end]`` — an apples-to-apples "is Elon tracking the
    model so far?" check. For a finished market leave ``eval_end=None`` (uses the full window)."""
    rng = rng or np.random.default_rng(7)
    ws, we = W.utc_ts(window_start), W.utc_ts(window_end)
    eval_we = we if eval_end is None else min(we, W.utc_ts(eval_end))
    eval_end_dt = eval_we.to_pydatetime()
    T_eval = (eval_we - ws).total_seconds() / 3600.0
    beta = fit.hawkes.beta
    posts = fit.posts

    # realized event times over the evaluated portion (hours from open)
    rev = posts.loc[(posts["created_at"] >= ws) & (posts["created_at"] < eval_we), "created_at"]
    realized_ev = np.sort((rev - ws).dt.total_seconds().values / 3600.0)
    realized = path_stats(realized_ev, window_start, eval_end_dt, beta, gap_mult=gap_mult)

    # simulate the full window, then truncate each path to the evaluated portion
    paths = simulate_window_paths(fit, window_start, window_end, n_sims=n_sims, rng=rng)
    sims = [path_stats(p[p < T_eval], window_start, eval_end_dt, beta, gap_mult=gap_mult)
            for p in paths]

    scalars = [
        _scalar_row(k, lab, hint, realized[k], np.array([s[k] for s in sims]))
        for (k, lab, hint) in _SCALARS
    ]
    per_day = _band(realized["per_day"], [s["per_day"] for s in sims])
    per_day["dates"] = realized["_et_dates"]
    per_hour = _band(realized["per_hour"], [s["per_hour"] for s in sims])
    burst_hour = _band(realized["burst_hour"], [s["burst_hour"] for s in sims])

    return {
        "scalars": scalars,
        "per_day": per_day,
        "per_hour": per_hour,
        "burst_hour": burst_hour,
        "meta": {
            "realized_total": int(realized["total"]),
            "n_sims": len(paths),
            "alpha": float(fit.hawkes.alpha), "beta": float(beta),
            "burst_timescale_min": float(1.0 / beta * 60.0) if beta > 0 else float("nan"),
            "gap_threshold_min": float(gap_mult / beta * 60.0) if beta > 0 else float("nan"),
            "window_hours": float((we - ws).total_seconds() / 3600.0),
            "eval_hours": float(T_eval),
            "partial": bool(eval_we < we),
        },
    }

