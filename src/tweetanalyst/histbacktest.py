"""Historical backtest against *resolved* Polymarket markets, with real market prices.

Unlike ``backtest.py`` (which only scores the model against realized outcomes), this replays the
model at many in-week timepoints τ AND pulls the market's historical price at τ, so we can answer:
  * how reliable is the model vs the market as the week unfolds?
  * how much edge is still available at each τ (and does a simple strategy make money)?

Ground truth = realized tweet count from XTracker (validated to match resolution on all 18 weeks).
Market prices = CLOB price-history per bracket YES token. Resolved markets enumerated from Gamma.
"""
from __future__ import annotations

import datetime as dt
import sqlite3
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import requests

from . import calibration as CAL
from . import data as D
from . import model as M
from . import windows as W

CLOB = "https://clob.polymarket.com/prices-history"
_MONTHS = ["january", "february", "march", "april", "may", "june",
           "july", "august", "september", "october", "november", "december"]


# --------------------------------------------------------------------------- #
# Resolved-market enumeration
# --------------------------------------------------------------------------- #
@dataclass
class ResolvedMarket:
    slug: str
    window_start: dt.datetime
    window_end: dt.datetime
    realized: int
    winner: str
    brackets: list  # list of (low, high, label, yes_token)


def _slug(ws: dt.datetime, we: dt.datetime) -> str:
    a = W.utc_ts(ws).tz_convert(W.ET)
    b = W.utc_ts(we).tz_convert(W.ET)
    return f"elon-musk-of-tweets-{_MONTHS[a.month-1]}-{a.day}-{_MONTHS[b.month-1]}-{b.day}"


def enumerate_resolved(
    posts: pd.DataFrame, anchor_end: dt.datetime, n_weeks: int = 18
) -> list[ResolvedMarket]:
    import json

    tv = posts["created_at"].values
    anchor = W.utc_ts(anchor_end).tz_convert(W.ET)
    out = []
    for k in range(1, n_weeks + 2):
        we_et = anchor - pd.Timedelta(weeks=k - 1)
        ws_et = we_et - pd.Timedelta(weeks=1)
        ws = ws_et.tz_convert("UTC").to_pydatetime()
        we = we_et.tz_convert("UTC").to_pydatetime()
        slug = _slug(ws, we)
        try:
            d = requests.get("https://gamma-api.polymarket.com/events",
                             params={"slug": slug}, timeout=20).json()
        except Exception:  # noqa: BLE001
            continue
        if not d or not d[0].get("closed"):
            continue
        e = d[0]
        realized = int(((tv >= np.datetime64(W.utc_ts(ws).value, "ns"))
                        & (tv < np.datetime64(W.utc_ts(we).value, "ns"))).sum())
        brackets, winner = [], None
        for m in e["markets"]:
            label = m.get("groupItemTitle") or ""
            lo, hi = D._parse_bracket_bounds(label)
            toks = m.get("clobTokenIds")
            oc = m.get("outcomes")
            op = m.get("outcomePrices")
            if isinstance(toks, str):
                toks = json.loads(toks)
            if isinstance(oc, str):
                oc = json.loads(oc)
            if isinstance(op, str):
                op = json.loads(op)
            yi = [o.lower() for o in oc].index("yes") if oc and "yes" in [o.lower() for o in oc] else 0
            yes_token = toks[yi] if toks else None
            brackets.append((lo, hi, label, yes_token))
            if op and float(op[yi]) > 0.5:
                winner = label
        brackets.sort(key=lambda b: b[0])
        out.append(ResolvedMarket(slug, ws, we, realized, winner, brackets))
    return list(reversed(out))


# --------------------------------------------------------------------------- #
# CLOB price history (cached on disk)
# --------------------------------------------------------------------------- #
def _price_conn() -> sqlite3.Connection:
    con = D._conn()
    con.execute("""CREATE TABLE IF NOT EXISTS clob_prices(
                       token TEXT, t INTEGER, p REAL, PRIMARY KEY(token, t))""")
    return con


def fetch_prices(token: str, t0: dt.datetime, t1: dt.datetime, fidelity: int = 60) -> pd.DataFrame:
    """Return price history (t [UTC], p) for a YES token over [t0, t1], cached locally."""
    if token is None:
        return pd.DataFrame(columns=["t", "p"])
    con = _price_conn()
    have = con.execute("SELECT COUNT(*) FROM clob_prices WHERE token=?", (token,)).fetchone()[0]
    if have == 0:
        start = int(W.utc_ts(t0).timestamp()) - 3600
        end = int(W.utc_ts(t1).timestamp()) + 3600
        try:
            r = requests.get(CLOB, params={"market": token, "startTs": start,
                                           "endTs": end, "fidelity": fidelity}, timeout=30)
            hist = r.json().get("history", []) if r.status_code == 200 else []
        except Exception:  # noqa: BLE001
            hist = []
        rows = [(token, int(h["t"]), float(h["p"])) for h in hist]
        if rows:
            with con:
                con.executemany("INSERT OR IGNORE INTO clob_prices VALUES(?,?,?)", rows)
        else:
            with con:  # sentinel so we don't refetch empty tokens
                con.execute("INSERT OR IGNORE INTO clob_prices VALUES(?,?,?)", (token, 0, -1.0))
        time.sleep(0.08)
    df = pd.read_sql_query(
        "SELECT t, p FROM clob_prices WHERE token=? AND t>0 ORDER BY t", con, params=[token])
    con.close()
    return df


def market_probs_at(mkt: ResolvedMarket, tau_utc: dt.datetime,
                    price_cache: dict) -> np.ndarray:
    """Forward-filled YES price per bracket at τ, normalized to a probability vector."""
    ts = int(W.utc_ts(tau_utc).timestamp())
    raw = []
    for (lo, hi, label, tok) in mkt.brackets:
        ph = price_cache.get(tok)
        if ph is None or ph.empty:
            raw.append(0.0005)
            continue
        prior = ph[ph["t"] <= ts]
        raw.append(float(prior["p"].iloc[-1]) if len(prior) else float(ph["p"].iloc[0]))
    raw = np.array(raw)
    return raw


# --------------------------------------------------------------------------- #
# The backtest
# --------------------------------------------------------------------------- #
@dataclass
class HistBacktest:
    records: pd.DataFrame
    by_tau: pd.DataFrame


def run(
    posts: pd.DataFrame,
    markets: list[ResolvedMarket],
    taus: tuple[float, ...] = tuple(np.round(np.arange(0.0, 0.95, 0.1), 2)),
    n_sims: int = 4000,
    gamma: float = 1.45,
    edge_threshold: float = 0.05,
    seed: int = 11,
    progress: bool = True,
) -> HistBacktest:
    rng = np.random.default_rng(seed)
    rows = []
    for mi, mkt in enumerate(markets):
        # preload + cache all bracket price histories for this market
        price_cache = {tok: fetch_prices(tok, mkt.window_start, mkt.window_end)
                       for (_, _, _, tok) in mkt.brackets}
        bounds = [(lo, hi) for (lo, hi, _, _) in mkt.brackets]
        win_idx = next(i for i, (lo, hi, lab, _) in enumerate(mkt.brackets) if lab == mkt.winner) \
            if mkt.winner else int(np.argmin([abs(mkt.realized - lo) for lo, hi in bounds]))

        span = W.utc_ts(mkt.window_end) - W.utc_ts(mkt.window_start)
        for tau in taus:
            now = (W.utc_ts(mkt.window_start) + span * tau).to_pydatetime()
            # --- model (only data < now) ---
            fit = M.fit_model(posts, now)
            fc = M.forecast(fit, mkt.window_start, mkt.window_end, n_sims=n_sims, rng=rng)
            tbl = M.bracket_probabilities(
                [D.Bracket(lab, lo, hi, None) for (lo, hi, lab, _) in mkt.brackets],
                fc.samples, gamma=gamma)
            p_model = np.array([t["model_prob"] for t in tbl])
            # --- market at now ---
            p_mkt_raw = market_probs_at(mkt, now, price_cache)
            S = p_mkt_raw.sum()
            p_mkt = p_mkt_raw / S if S > 0 else np.full(len(p_mkt_raw), 1 / len(p_mkt_raw))
            # --- scores ---
            ll_model = -np.log(max(p_model[win_idx], 1e-6))
            ll_mkt = -np.log(max(p_mkt[win_idx], 1e-6))
            # --- trade: best positive edge vs RAW market price ---
            edges = p_model - p_mkt_raw
            b = int(np.argmax(edges))
            took = edges[b] > edge_threshold
            pnl = ((1.0 if b == win_idx else 0.0) - p_mkt_raw[b]) if took else 0.0
            rows.append({
                "slug": mkt.slug, "tau": tau, "n_obs": fc.n_obs,
                "p_model_true": float(p_model[win_idx]),
                "p_mkt_true": float(p_mkt[win_idx]),
                "ll_model": float(ll_model), "ll_mkt": float(ll_mkt),
                "max_edge": float(edges[b]), "took_trade": bool(took),
                "trade_win": bool(b == win_idx) if took else None,
                "pnl": float(pnl), "edge_taken": float(edges[b]) if took else 0.0,
            })
        if progress:
            print(f"[{mi+1}/{len(markets)}] {mkt.slug} done", flush=True)

    rec = pd.DataFrame(rows)
    g = rec.groupby("tau").agg(
        model_prob_true=("p_model_true", "mean"),
        market_prob_true=("p_mkt_true", "mean"),
        model_logloss=("ll_model", "mean"),
        market_logloss=("ll_mkt", "mean"),
        avg_max_edge=("max_edge", "mean"),
        trade_rate=("took_trade", "mean"),
        pnl_per_trade=("pnl", "mean"),
        n=("pnl", "size"),
    ).reset_index()
    # PnL conditional on actually trading, and hit rate
    taken = rec[rec.took_trade]
    if len(taken):
        cond = taken.groupby("tau").agg(pnl_if_traded=("pnl", "mean"),
                                        hit_rate=("trade_win", "mean")).reset_index()
        g = g.merge(cond, on="tau", how="left")
    return HistBacktest(records=rec, by_tau=g)


# --------------------------------------------------------------------------- #
# Strategy backtest: realized PnL of a fractional-Kelly plan at historical prices
# --------------------------------------------------------------------------- #
def strategy_backtest(
    posts: pd.DataFrame,
    anchor_end: dt.datetime,
    n_weeks: int = 16,
    taus: tuple[float, ...] = (0.4, 0.6, 0.8, 0.92),
    kelly_fraction: float = 0.25,
    edge_threshold: float = 0.04,
    bankroll: float = 1000.0,
    max_per_market_frac: float = 0.5,
    n_sims: int = 3000,
    seed: int = 13,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Replay the fractional-Kelly strategy on resolved markets at real historical prices.

    At each (market, τ): fit the model, read each bracket's real YES price at τ (NO = 1−YES), size
    bets by ¼-Kelly with the per-market cap, then settle on the realized outcome. Returns
    (records per market×τ, aggregated-by-τ ROI/hit-rate)."""
    markets = enumerate_resolved(posts, anchor_end, n_weeks)
    rng = np.random.default_rng(seed)
    rows = []
    for mkt in markets:
        price_cache = {tok: fetch_prices(tok, mkt.window_start, mkt.window_end)
                       for (_, _, _, tok) in mkt.brackets}
        win_idx = next(i for i, (lo, hi, lab, _) in enumerate(mkt.brackets) if lab == mkt.winner) \
            if mkt.winner else int(np.argmin([abs(mkt.realized - lo) for lo, hi, _, _ in mkt.brackets]))
        span = W.utc_ts(mkt.window_end) - W.utc_ts(mkt.window_start)
        dur_days = span.total_seconds() / 86400.0
        gamma = CAL.gamma_for_duration(dur_days)
        widths = [hi - lo + 1 for (lo, hi, _, _) in mkt.brackets if np.isfinite(hi)]
        bracket_width = float(np.median(widths)) if widths else 20.0
        for tau in taus:
            now = (W.utc_ts(mkt.window_start) + span * tau).to_pydatetime()
            fit = M.fit_model(posts, now)
            fc = M.forecast(fit, mkt.window_start, mkt.window_end, n_sims=n_sims, rng=rng)
            tbl = M.bracket_probabilities(
                [D.Bracket(lab, lo, hi, None) for (lo, hi, lab, _) in mkt.brackets],
                fc.samples, gamma=gamma)
            yes_prices = market_probs_at(mkt, now, price_cache)  # raw forward-filled YES price/bracket

            cands = []
            for i in range(len(mkt.brackets)):
                p_in = tbl[i]["model_prob"]
                yq = float(yes_prices[i])
                for side, price, p_side in (("OUI", yq, p_in), ("NON", 1.0 - yq, 1.0 - p_in)):
                    if price <= 0.0 or price >= 1.0:
                        continue
                    edge = p_side - price
                    if edge <= edge_threshold:
                        continue
                    f = kelly_fraction * edge / (1.0 - price)
                    cands.append({"i": i, "side": side, "price": price, "stake": bankroll * f})
            cap = bankroll * max_per_market_frac
            tot = sum(c["stake"] for c in cands)
            if tot > cap and tot > 0:
                for c in cands:
                    c["stake"] *= cap / tot

            staked = pnl = 0.0
            nbet = won = 0
            for c in cands:
                if c["stake"] < 0.5:
                    continue
                staked += c["stake"]
                nbet += 1
                win = (c["side"] == "OUI" and c["i"] == win_idx) or \
                      (c["side"] == "NON" and c["i"] != win_idx)
                if win:
                    pnl += c["stake"] * (1.0 / c["price"] - 1.0)
                    won += 1
                else:
                    pnl -= c["stake"]
            sigma_ratio = float(fc.samples.std()) / bracket_width  # forecast σ in bracket-widths
            rows.append({"week_end": mkt.window_end, "tau": tau, "dur_days": round(dur_days, 1),
                         "hours_remaining": round(fc.hours_remaining, 1),
                         "sigma_ratio": sigma_ratio,
                         "staked": staked, "pnl": pnl,
                         "roi": (pnl / staked) if staked > 0 else 0.0,
                         "n_bets": nbet, "hit_rate": (won / nbet) if nbet else float("nan")})

    rec = pd.DataFrame(rows)
    if rec.empty:
        return rec, rec
    g = rec.groupby("tau").apply(lambda d: pd.Series({
        "n_market_weeks": len(d),
        "avg_bets": d["n_bets"].mean(),
        "total_staked": d["staked"].sum(),
        "total_pnl": d["pnl"].sum(),
        "roi_aggregate": d["pnl"].sum() / d["staked"].sum() if d["staked"].sum() > 0 else 0.0,
        "roi_mean_per_week": d.loc[d["staked"] > 0, "roi"].mean(),
        "hit_rate": d["hit_rate"].mean(),
    })).reset_index()
    return rec, g
