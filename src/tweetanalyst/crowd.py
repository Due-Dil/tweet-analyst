"""Crowd-behaviour analysis on the Elon tweet markets (Phase A — aggregate, price-based).

Studies how the *market price* (the crowd's collective belief) behaves over resolved markets, to find
exploitable patterns. Phase A uses only data we already cache — CLOB price history + realized winners
(via ``pathbacktest.enumerate_resolved_series``) — so it needs no model re-fit and runs fast.

Two questions it answers:
  1. **Tail mispricing by period** — are extreme/longshot brackets systematically over- or under-priced,
     and in which part of the window (start/mid/end)? Measured by *calibration*: bucket every
     (market, bracket, τ) by price and compare the price to the realized win-frequency. A realized
     frequency BELOW the price means the bracket was overpriced (classic longshot bias).
  2. **Overreaction to bursts** — does the price over-shoot on a jump and then revert? Measured by a
     jump→forward-return event study and the lag-1 autocorrelation of price changes (negative
     autocorr / positive reversion = overreaction). Jumps are cross-checked against real tweet bursts.
"""
from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd

from . import data as D  # noqa: F401  (kept for symmetry / future trade-level work)
from . import histbacktest as HB
from . import pathbacktest as PB
from . import windows as W

_TAUS = (0.1, 0.25, 0.4, 0.55, 0.7, 0.85, 0.95)
_PHASES = ([0.0, 0.33, 0.66, 1.01], ["début", "milieu", "fin"])
_PRICE_BUCKETS = [0.0, 0.02, 0.05, 0.10, 0.25, 0.50, 0.75, 1.01]


# --------------------------------------------------------------------------- #
# Observation collection: (market, bracket, τ) -> price, won, tail flags
# --------------------------------------------------------------------------- #
def collect_observations(
    posts: pd.DataFrame, anchor_end: dt.datetime, taus: tuple[float, ...] = _TAUS,
    durations: tuple[float, ...] | None = None, max_markets: int | None = None, progress: bool = True,
) -> pd.DataFrame:
    markets = PB.enumerate_resolved_series(posts, anchor_end, durations=durations, max_markets=max_markets)
    if progress:
        print(f"marchés résolus: {len(markets)}", flush=True)
    rows = []
    for mi, mkt in enumerate(markets):
        pc = {tok: HB.fetch_prices(tok, mkt.window_start, mkt.window_end)
              for (_, _, _, tok) in mkt.brackets}
        span = W.utc_ts(mkt.window_end) - W.utc_ts(mkt.window_start)
        dur = span.total_seconds() / 86400.0
        n = len(mkt.brackets)
        for tau in taus:
            now = (W.utc_ts(mkt.window_start) + span * tau).to_pydatetime()
            prices = HB.market_probs_at(mkt, now, pc)  # raw forward-filled YES price per bracket
            if float(np.sum(prices)) < 0.5:            # no real book at this τ -> skip the snapshot
                continue
            for rank, (lo, hi, lab, _) in enumerate(mkt.brackets):
                price = float(prices[rank])
                if price <= 0.0 or price >= 1.0:
                    continue
                rows.append({
                    "slug": mkt.slug, "dur_days": round(dur, 1), "tau": tau, "label": lab,
                    "rank": rank, "n_brackets": n, "price": price,
                    "won": int(lab == mkt.winner),
                    "tail": rank in (0, n - 1), "low_tail": rank == 0, "high_tail": rank == n - 1,
                })
        if progress:
            print(f"[{mi+1}/{len(markets)}] {mkt.slug}", flush=True)
    return pd.DataFrame(rows)


def _phase(obs: pd.DataFrame) -> pd.Series:
    return pd.cut(obs["tau"], _PHASES[0], labels=_PHASES[1], right=False)


# --------------------------------------------------------------------------- #
# Calibration (longshot bias) by price bucket × window phase
# --------------------------------------------------------------------------- #
def calibration(obs: pd.DataFrame) -> pd.DataFrame:
    o = obs.copy()
    o["phase"] = _phase(o)
    o["bucket"] = pd.cut(o["price"], _PRICE_BUCKETS, right=False)
    g = (o.groupby(["phase", "bucket"], observed=True)
         .agg(n=("won", "size"), prix_moyen=("price", "mean"), freq_réelle=("won", "mean"))
         .reset_index())
    g["écart"] = g["freq_réelle"] - g["prix_moyen"]   # <0 = surcoté (overpriced), >0 = sous-coté
    return g


# --------------------------------------------------------------------------- #
# Tail vs center, by phase
# --------------------------------------------------------------------------- #
def tail_summary(obs: pd.DataFrame) -> pd.DataFrame:
    o = obs.copy()
    o["phase"] = _phase(o)
    o["type"] = np.where(o["tail"], "tail", "centre")

    def agg(d: pd.DataFrame) -> pd.Series:
        return pd.Series({"n": len(d), "prix_moyen": d["price"].mean(),
                          "freq_réelle": d["won"].mean(),
                          "écart": d["won"].mean() - d["price"].mean()})

    return o.groupby(["phase", "type"], observed=True).apply(agg).reset_index()


# --------------------------------------------------------------------------- #
# Overreaction: jump -> forward-return event study + return autocorrelation
# --------------------------------------------------------------------------- #
def overreaction(
    posts: pd.DataFrame, anchor_end: dt.datetime, durations: tuple[float, ...] | None = None,
    jump: float = 0.08, horizon_h: float = 6.0, max_markets: int | None = None, progress: bool = True,
) -> tuple[dict, pd.DataFrame]:
    markets = PB.enumerate_resolved_series(posts, anchor_end, durations=durations, max_markets=max_markets)
    tv = posts["created_at"].values
    rets: list[float] = []
    events: list[dict] = []
    for mi, mkt in enumerate(markets):
        for (lo, hi, lab, tok) in mkt.brackets:
            ph = HB.fetch_prices(tok, mkt.window_start, mkt.window_end)
            if ph.empty or len(ph) < 5:
                continue
            ph = ph[ph["p"] > 0].sort_values("t")
            t = ph["t"].values.astype(float)
            p = ph["p"].values.astype(float)
            rets.extend(np.diff(p).tolist())
            for i in range(1, len(p)):
                jmp = p[i] - p[i - 1]                       # ~1-bar (hourly) change
                if abs(jmp) < jump:
                    continue
                fwd_idx = int(np.searchsorted(t, t[i] + horizon_h * 3600))
                if fwd_idx >= len(p):
                    continue
                fwd = p[fwd_idx] - p[i]                     # change over the next ~horizon_h
                # tweets in the hour before the jump (did it follow a real burst?)
                lo_t = np.datetime64(int((t[i] - 3600) * 1e9), "ns")
                hi_t = np.datetime64(int(t[i] * 1e9), "ns")
                burst = int(((tv >= lo_t) & (tv < hi_t)).sum())
                events.append({"slug": mkt.slug, "label": lab, "jump": jmp, "forward": fwd,
                               "tweets_prev_h": burst})
        if progress and (mi + 1) % 25 == 0:
            print(f"[{mi+1}/{len(markets)}]", flush=True)
    rets_a = np.array(rets)
    ev = pd.DataFrame(events)
    ac1 = float(np.corrcoef(rets_a[:-1], rets_a[1:])[0, 1]) if len(rets_a) > 2 else float("nan")
    summary = {"n_returns": int(len(rets_a)), "autocorr_lag1": ac1, "n_jumps": int(len(ev))}
    if len(ev):
        up, dn = ev[ev["jump"] > 0], ev[ev["jump"] < 0]
        # reversion score: >0 means price tends to move BACK after a jump (overreaction)
        summary["fwd_after_up_jump"] = float(up["forward"].mean()) if len(up) else float("nan")
        summary["fwd_after_down_jump"] = float(dn["forward"].mean()) if len(dn) else float("nan")
        summary["reversion_score"] = float((-ev["forward"] * np.sign(ev["jump"])).mean())
        summary["jumps_after_burst_pct"] = float((ev["tweets_prev_h"] >= 3).mean())
    return summary, ev


# --------------------------------------------------------------------------- #
# The overreaction turned into a TRADEABLE rule, backtested at real prices
# --------------------------------------------------------------------------- #
def fade_backtest(
    posts: pd.DataFrame, anchor_end: dt.datetime, durations: tuple[float, ...] | None = None,
    jump: float = 0.08, horizon_h: float = 6.0, spread: float = 0.02, direction: str = "up",
    burst_only: bool = False, cooldown: bool = True, max_markets: int | None = None,
) -> tuple[dict, pd.DataFrame]:
    """Fade a price jump and exit after ``horizon_h`` — the overreaction rule, priced realistically.

    An UP jump in a bracket's YES price is faded by **buying NO** (betting the spike reverts); a DOWN
    jump by buying YES. We pay a half-``spread`` on entry and exit. ``burst_only`` keeps only jumps that
    followed a real tweet burst (≥3 in the prior hour). ``cooldown`` blocks a new entry on the same
    bracket until the open fade has been exited (avoids overlapping, double-counted trades).

    Returns (summary, trades). ``ret`` is the return per $ staked on the fade (NO is cheap when YES is
    high, so a small YES reversion is a larger % on the NO leg)."""
    markets = PB.enumerate_resolved_series(posts, anchor_end, durations=durations, max_markets=max_markets)
    tv = posts["created_at"].values
    hs = spread / 2.0
    trades: list[dict] = []
    for mkt in markets:
        for (lo, hi, lab, tok) in mkt.brackets:
            ph = HB.fetch_prices(tok, mkt.window_start, mkt.window_end)
            if ph.empty or len(ph) < 5:
                continue
            ph = ph[ph["p"] > 0].sort_values("t")
            t = ph["t"].values.astype(float)
            p = ph["p"].values.astype(float)
            last_exit = -np.inf
            for i in range(1, len(p)):
                if cooldown and t[i] < last_exit:
                    continue
                jmp = p[i] - p[i - 1]
                if abs(jmp) < jump or (direction == "up" and jmp <= 0) or \
                        (direction == "down" and jmp >= 0):
                    continue
                if burst_only:
                    lo_t = np.datetime64(int((t[i] - 3600) * 1e9), "ns")
                    hi_t = np.datetime64(int(t[i] * 1e9), "ns")
                    if int(((tv >= lo_t) & (tv < hi_t)).sum()) < 3:
                        continue
                fwd_idx = int(np.searchsorted(t, t[i] + horizon_h * 3600))
                if fwd_idx >= len(p):
                    continue
                p_i, p_fwd = p[i], p[fwd_idx]
                if jmp > 0:   # fade up = buy NO (profits if YES reverts down)
                    entry = min(max(1.0 - p_i + hs, 0.01), 0.99)
                    exit_ = min(max(1.0 - p_fwd - hs, 0.0), 0.99)
                else:         # fade down = buy YES
                    entry = min(max(p_i + hs, 0.01), 0.99)
                    exit_ = min(max(p_fwd - hs, 0.0), 0.99)
                trades.append({"slug": mkt.slug, "label": lab, "jump": float(jmp),
                               "p_i": float(p_i), "p_fwd": float(p_fwd),
                               "ret": float(exit_ / entry - 1.0)})
                last_exit = t[i] + horizon_h * 3600
    tr = pd.DataFrame(trades)
    summary = {"direction": direction, "jump": jump, "horizon_h": horizon_h, "spread": spread,
               "burst_only": burst_only, "n_trades": len(tr)}
    if len(tr):
        summary.update({"roi_moyen": float(tr["ret"].mean()), "roi_median": float(tr["ret"].median()),
                        "hit_rate": float((tr["ret"] > 0).mean()),
                        "total_ret_per_$": float(tr["ret"].sum())})
    return summary, tr


def _fade_observations(markets, jump: float, baseline_max: float, horizon_h: float,
                       spread: float) -> pd.DataFrame:
    """Per-bar NO-fade observations over a market list: {grp, price, fwd_yes, ret_no, window_end}.
    ``grp`` is 'up_jump' (1-bar move ≥ jump) or 'baseline' (|move| < baseline_max)."""
    hs = spread / 2.0
    rows = []
    for mkt in markets:
        for (lo, hi, lab, tok) in mkt.brackets:
            ph = HB.fetch_prices(tok, mkt.window_start, mkt.window_end)
            if ph.empty or len(ph) < 5:
                continue
            ph = ph[ph["p"] > 0].sort_values("t")
            t = ph["t"].values.astype(float)
            p = ph["p"].values.astype(float)
            for i in range(1, len(p)):
                jmp = p[i] - p[i - 1]
                if jmp >= jump:
                    grp = "up_jump"
                elif abs(jmp) < baseline_max:
                    grp = "baseline"
                else:
                    continue
                fwd_idx = int(np.searchsorted(t, t[i] + horizon_h * 3600))
                if fwd_idx >= len(p):
                    continue
                p_i, p_fwd = p[i], p[fwd_idx]
                entry = min(max(1.0 - p_i + hs, 0.01), 0.99)   # buy NO
                exit_ = min(max(1.0 - p_fwd - hs, 0.0), 0.99)
                rows.append({"grp": grp, "price": p_i, "fwd_yes": p_fwd - p_i,
                             "ret_no": exit_ / entry - 1.0, "window_end": mkt.window_end})
    return pd.DataFrame(rows)


def fade_vs_baseline(
    posts: pd.DataFrame, anchor_end: dt.datetime, durations: tuple[float, ...] | None = None,
    jump: float = 0.08, baseline_max: float = 0.02, horizon_h: float = 6.0, spread: float = 0.02,
    max_markets: int | None = None,
) -> pd.DataFrame:
    """Isolate PURE overreaction from time-drift: NO bought right after an up-jump vs NO bought at a
    quiet bar, same horizon, **matched by entry price level**. If up-jump beats the price-matched
    baseline, the jump predicts extra reversion (real overreaction). Per-(price bucket × group) table."""
    markets = PB.enumerate_resolved_series(posts, anchor_end, durations=durations, max_markets=max_markets)
    df = _fade_observations(markets, jump, baseline_max, horizon_h, spread)
    if df.empty:
        return df
    df["bucket"] = pd.cut(df["price"], [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.01], right=False)
    return (df.groupby(["bucket", "grp"], observed=True)
            .agg(n=("ret_no", "size"), fwd_yes_moyen=("fwd_yes", "mean"),
                 ret_no_moyen=("ret_no", "mean"), hit=("ret_no", lambda x: float((x > 0).mean())))
            .reset_index())


def fade_walkforward(
    posts: pd.DataFrame, anchor_end: dt.datetime, durations: tuple[float, ...] | None = None,
    train_frac: float = 0.6, zone: tuple[float, float] = (0.25, 0.75), jump: float = 0.08,
    baseline_max: float = 0.02, horizon_h: float = 6.0, spread: float = 0.02,
) -> pd.DataFrame:
    """Out-of-sample check of the mid-zone fade edge. Markets are split CHRONOLOGICALLY by close date
    (train = earliest ``train_frac``, test = the rest). On each split we measure, in the price zone
    ``zone``, the up-jump NO-return minus the price-matched baseline (the pure overreaction edge). If
    the TEST edge stays positive and close to TRAIN, the rule isn't overfit."""
    markets = PB.enumerate_resolved_series(posts, anchor_end, durations=durations)
    markets = sorted(markets, key=lambda m: W.utc_ts(m.window_end))
    n = len(markets)
    cut = markets[max(0, int(train_frac * n) - 1)].window_end if n else anchor_end
    df = _fade_observations(markets, jump, baseline_max, horizon_h, spread)
    df = df[(df["price"] >= zone[0]) & (df["price"] <= zone[1])]
    rows = []
    for name, sub in [("train", df[df["window_end"] <= cut]), ("test", df[df["window_end"] > cut])]:
        up = sub[sub["grp"] == "up_jump"]["ret_no"]
        base = sub[sub["grp"] == "baseline"]["ret_no"]
        rows.append({"split": name, "n_up": len(up), "n_base": len(base),
                     "up_ret": float(up.mean()) if len(up) else float("nan"),
                     "base_ret": float(base.mean()) if len(base) else float("nan"),
                     "edge": (float(up.mean() - base.mean()) if len(up) and len(base) else float("nan")),
                     "up_hit": float((up > 0).mean()) if len(up) else float("nan")})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# LIVE fade signal: detect a recent up-spike landing in the mid-price zone
# --------------------------------------------------------------------------- #
def detect_fade_signals(
    slug: str, handle: str = "elonmusk", jump: float = 0.08,
    zone: tuple[float, float] = (0.25, 0.75), lookback_h: float = 2.0,
) -> list[dict]:
    """Live overreaction alerts for an active market: brackets whose YES price has spiked up by ≥
    ``jump`` over the last ``lookback_h`` hours AND now sits in ``zone`` (the validated fade region).
    Each signal suggests buying NO (fade the spike). Fetches fresh CLOB prices (not the disk cache)."""
    import json
    import requests

    now = dt.datetime.now(dt.timezone.utc)
    try:
        ev = requests.get(f"{PB.GAMMA}/events", params={"slug": slug}, timeout=20).json()
    except Exception:  # noqa: BLE001
        return []
    if not ev:
        return []
    e = ev[0]
    if e.get("closed"):
        return []
    start_ts = int(now.timestamp() - lookback_h * 3600)
    end_ts = int(now.timestamp())
    out = []
    for m in e.get("markets", []):
        label = m.get("groupItemTitle") or ""
        toks, oc = m.get("clobTokenIds"), m.get("outcomes")
        if isinstance(toks, str):
            toks = json.loads(toks)
        if isinstance(oc, str):
            oc = json.loads(oc)
        if not toks:
            continue
        yi = ([o.lower() for o in oc].index("yes") if oc and "yes" in [o.lower() for o in oc] else 0)
        token = toks[yi]
        try:
            hist = requests.get(HB.CLOB, params={"market": token, "startTs": start_ts,
                                                 "endTs": end_ts, "fidelity": 10}, timeout=20).json()
            h = hist.get("history", []) if isinstance(hist, dict) else []
        except Exception:  # noqa: BLE001
            continue
        if len(h) < 2:
            continue
        p_then, p_now = float(h[0]["p"]), float(h[-1]["p"])
        jmp = p_now - p_then
        if jmp >= jump and zone[0] <= p_now <= zone[1]:
            out.append({"tranche": label, "prix_yes": p_now, "saut": jmp,
                        "prix_no": round(1.0 - p_now, 3), "côté": "NON",
                        "raison": f"pic +{jmp:.0%} sur {lookback_h:.0f}h → fade (zone {zone[0]:.2f}–{zone[1]:.2f})"})
    return sorted(out, key=lambda s: -s["saut"])
