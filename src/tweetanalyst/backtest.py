"""Backtest: replay past weeks and score the forecast's calibration.

For each recent completed week and several in-week checkpoints, we refit using *only* data
available at that moment, forecast the final total, and compare the predicted bracket
distribution to what actually happened. Outputs per-forecast scores (log-loss, multiclass
Brier) and the raw (predicted-prob, hit) pairs needed for a reliability/calibration curve.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import model as M
from . import windows as W

# Standard 20-wide brackets used across all weeks for scoring (mirrors the market design).
def standard_brackets(width: int = 20, n: int = 26) -> list[tuple[float, float, str]]:
    """Generate ``n`` brackets of ``width`` tweets (last one open-ended), matching how Polymarket
    structures these markets. Width varies by market: ~20 for weekly, ~25 for short 2-3 day markets.
    """
    out = [(0.0, width - 1.0, f"<{width}")]
    for lo in range(width, width * (n - 1), width):
        out.append((float(lo), float(lo + width - 1), f"{lo}-{lo+width-1}"))
    top = width * (n - 1)
    out.append((float(top), np.inf, f"{top}+"))
    return out


def _bracket_index(total: float, brs: list) -> int:
    for i, (lo, hi, _) in enumerate(brs):
        if lo <= total <= hi:
            return i
    return len(brs) - 1


@dataclass
class BacktestResult:
    records: pd.DataFrame       # one row per (week, checkpoint)
    reliability: pd.DataFrame   # binned predicted-prob vs empirical hit-rate
    log_loss: float
    brier: float
    prob_matrix: np.ndarray     # (n_forecasts, n_brackets) predicted distributions
    true_idx: np.ndarray        # (n_forecasts,) realized bracket index


def sharpen(probs: np.ndarray, gamma: float) -> np.ndarray:
    """Temperature/sharpening recalibration: q ∝ p**gamma, renormalized (gamma>1 sharpens)."""
    q = np.power(np.clip(probs, 1e-12, 1.0), gamma)
    return q / q.sum(axis=-1, keepdims=True)


def fit_sharpening(
    prob_matrix: np.ndarray, true_idx: np.ndarray, grid: np.ndarray | None = None
) -> tuple[float, float, float]:
    """Find gamma minimizing mean log-loss on the backtest. Returns (gamma, ll_before, ll_after)."""
    if grid is None:
        grid = np.linspace(0.5, 3.5, 61)
    rows = np.arange(len(true_idx))
    ll0 = float(-np.log(np.clip(prob_matrix[rows, true_idx], 1e-12, 1.0)).mean())
    best_g, best_ll = 1.0, ll0
    for g in grid:
        q = sharpen(prob_matrix, g)
        ll = float(-np.log(np.clip(q[rows, true_idx], 1e-12, 1.0)).mean())
        if ll < best_ll:
            best_ll, best_g = ll, float(g)
    return best_g, ll0, best_ll


def run_backtest(
    posts: pd.DataFrame,
    anchor_end: dt.datetime,
    n_weeks: int = 12,
    checkpoints: tuple[float, ...] = (0.0, 0.35, 0.6, 0.85),
    n_sims: int = 4000,
    seed: int = 7,
    level_prior_strength: float | None = 0.5,
    grid: list | None = None,
    brackets: list | None = None,
) -> BacktestResult:
    # Score against the given bracket structure (so calibration matches the *actual* market: ~20-wide
    # weekly, ~25-wide for short markets). Defaults to standard 20-wide.
    brs = brackets if brackets is not None else standard_brackets(width=20)
    hist_start = posts["created_at"].min().to_pydatetime()
    if grid is None:
        grid = W.weekly_grid(hist_start, anchor_end, n_weeks)
    rng = np.random.default_rng(seed)

    rows = []
    rel_pairs = []  # (pred_prob, hit)
    prob_rows: list[np.ndarray] = []
    true_rows: list[int] = []
    for (ws, we) in grid:
        actual = int(
            (
                (posts["created_at"] >= W.utc_ts(ws)) & (posts["created_at"] < W.utc_ts(we))
            ).sum()
        )
        true_idx = _bracket_index(actual, brs)
        for frac in checkpoints:
            now = W.utc_ts(ws) + (W.utc_ts(we) - W.utc_ts(ws)) * frac
            fit = M.fit_model(posts, now.to_pydatetime())
            fc = M.forecast(fit, ws, we, n_sims=n_sims, rng=rng,
                            level_prior_strength=level_prior_strength)
            samp = fc.samples
            probs = np.array(
                [
                    ((samp >= lo) & (samp <= (hi if np.isfinite(hi) else np.inf))).mean()
                    for (lo, hi, _) in brs
                ]
            )
            probs = np.clip(probs, 1e-6, 1.0)
            probs /= probs.sum()
            ll = -np.log(probs[true_idx])
            brier = float(((probs - np.eye(len(brs))[true_idx]) ** 2).sum())
            rows.append(
                {
                    "week_end": we, "checkpoint": frac, "actual": actual,
                    "n_obs": fc.n_obs, "pred_true_bracket": float(probs[true_idx]),
                    "log_loss": float(ll), "brier": brier,
                }
            )
            for k, p in enumerate(probs):
                rel_pairs.append((float(p), int(k == true_idx)))
            prob_rows.append(probs)
            true_rows.append(true_idx)

    records = pd.DataFrame(rows)
    rel = pd.DataFrame(rel_pairs, columns=["pred", "hit"])
    # reliability curve: bin predictions, compare mean predicted vs empirical hit-rate
    bins = np.linspace(0, 1, 11)
    rel["bin"] = pd.cut(rel["pred"], bins, include_lowest=True)
    reliability = (
        rel.groupby("bin", observed=True)
        .agg(mean_pred=("pred", "mean"), hit_rate=("hit", "mean"), n=("hit", "size"))
        .reset_index(drop=True)
        .dropna()
    )
    return BacktestResult(
        records=records,
        reliability=reliability,
        log_loss=float(records["log_loss"].mean()) if len(records) else float("nan"),
        brier=float(records["brier"].mean()) if len(records) else float("nan"),
        prob_matrix=np.array(prob_rows) if prob_rows else np.empty((0, len(brs))),
        true_idx=np.array(true_rows, dtype=int),
    )
