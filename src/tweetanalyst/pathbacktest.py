"""Path-dependent (sequential) strategy backtest on *resolved* Polymarket markets.

Unlike ``histbacktest.strategy_backtest`` — which enters once at each τ *independently* and holds to
resolution — this replays the strategy as a **continuously-monitored, continuously-rebalanced**
position book: it walks one market week on a fine τ grid (the tool "watches the tweets all the time"),
carries positions forward, marks them to the real CLOB price at every step, and can **enter / reinforce
/ trim / exit / take profit** at any moment (not only near close). Every buy and sell pays a configurable
bid-ask **spread haircut**, so PnL is no longer an upper bound.

Three things make it duration-agnostic and richer than the old backtest:
  * **Enumeration via the Gamma series** ``elon-tweets`` (id 10000), with the true counting window
    parsed from each event's *description* (authoritative; the slug/startDate are unreliable). This
    auto-covers 2/3/7-day markets — currently the resolved history is essentially all 7-day, but
    short markets join automatically as they resolve.
  * **Continuous rebalancing rules** toggled per *variant*, so we can compare buy-and-hold vs an
    active book that takes profit by selling into inflated prices and cuts when the edge flips.
  * **ROI = realized PnL / total cash staked**, broken down by variant and by market duration.

Ground truth = realized tweet count from XTracker. Market prices = CLOB price-history per YES token.
Not financial advice — decision support whose own edge is uncertain.
"""
from __future__ import annotations

import datetime as dt
import re
import zoneinfo
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import requests

from . import calibration as CAL
from . import data as D
from . import histbacktest as HB
from . import model as M
from . import windows as W

GAMMA = "https://gamma-api.polymarket.com"
# Elon "# of tweets" recurring series on Gamma. 10000 = weekly (7-day), 10816 = "48h" (2-day, daily
# cadence). No 72h/3-day series exists. The backtest scans ALL of them so every resolved duration is
# covered (the weekly-only scan used to silently drop the 2-day history).
ELON_SERIES_IDS = (10000, 10816)
_ET = zoneinfo.ZoneInfo("America/New_York")
_MON = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june",
     "july", "august", "september", "october", "november", "december"])}
# Two description formats seen in the wild:
#   new: "...from June 19 12:00 PM ET to June 26, 2026 12:00 PM ET."  -> two "12:00 PM ET" anchors
#   old: "...between December 13 and December 20."                    -> no time, no year
_ANCHOR = re.compile(r"([A-Za-z]+)\s+(\d{1,2})(?:,\s*(\d{4}))?,?\s*12:00\s*PM\s*ET", re.I)
_BETWEEN = re.compile(
    r"between\s+([A-Za-z]+)\s+(\d{1,2})(?:,?\s*(\d{4}))?\s+and\s+([A-Za-z]+)\s+(\d{1,2})(?:,?\s*(\d{4}))?",
    re.I)


# --------------------------------------------------------------------------- #
# Enumeration of resolved markets (all durations) from the Gamma series
# --------------------------------------------------------------------------- #
def _mk_window(m1, d1, y1, m2, d2, y2, year_hint):
    if m1.lower() not in _MON or m2.lower() not in _MON:
        return None
    sm, em = _MON[m1.lower()], _MON[m2.lower()]
    ey = int(y2) if y2 else (int(y1) if y1 else (year_hint or dt.datetime.now().year))
    sy = int(y1) if y1 else (ey - 1 if sm > em else ey)   # Dec->Jan rollover
    try:
        a = dt.datetime(sy, sm, int(d1), 12, 0, tzinfo=_ET).astimezone(dt.timezone.utc)
        b = dt.datetime(ey, em, int(d2), 12, 0, tzinfo=_ET).astimezone(dt.timezone.utc)
    except ValueError:
        return None
    return a, b


def parse_window(description: str, year_hint: int | None = None) -> tuple[dt.datetime, dt.datetime] | None:
    """Authoritative counting window (UTC) parsed from the market description, or None.

    Tries the new "...12:00 PM ET..." phrasing first (two anchors), then the old "between X and Y"
    phrasing (no time/year -> noon ET assumed, year from ``year_hint`` e.g. the event's end date).
    A missing start-year is inferred from the end year with Dec->Jan rollover."""
    found = _ANCHOR.findall(description or "")
    if len(found) >= 2:
        (m1, d1, y1), (m2, d2, y2) = found[0], found[1]
        w = _mk_window(m1, d1, y1, m2, d2, y2, year_hint)
        if w:
            return w
    m = _BETWEEN.search(description or "")
    if m:
        m1, d1, y1, m2, d2, y2 = m.groups()
        return _mk_window(m1, d1, y1, m2, d2, y2, year_hint)
    return None


def _series_events(limit: int = 100, max_pages: int = 4) -> list[dict]:
    """All closed events across every Elon-tweet series (weekly + 48h), newest first."""
    out: list[dict] = []
    for sid in ELON_SERIES_IDS:
        for page in range(max_pages):
            try:
                d = requests.get(f"{GAMMA}/events", params={
                    "series_id": sid, "closed": "true", "limit": limit,
                    "offset": page * limit, "order": "endDate", "ascending": "false"}, timeout=30).json()
            except Exception:  # noqa: BLE001
                break
            if not isinstance(d, list) or not d:
                break
            out.extend(d)
            if len(d) < limit:
                break
    return out


def enumerate_resolved_series(
    posts: pd.DataFrame,
    anchor_end: dt.datetime,
    durations: tuple[float, ...] | None = None,   # e.g. (7,) or (2, 3, 7); None = all
    history_buffer_days: float = 14.0,            # require this much post history before window start
    max_markets: int | None = None,
) -> list[HB.ResolvedMarket]:
    """Resolved Elon markets across all durations, newest first, restricted to the span the local
    post history can actually score (window_start >= posts.min + buffer, window_end <= anchor)."""
    import json

    tv = posts["created_at"].values
    pmin = pd.Timestamp(posts["created_at"].min()).to_pydatetime()
    floor = pmin + dt.timedelta(days=history_buffer_days)
    anchor = W.utc_ts(anchor_end)
    out: list[HB.ResolvedMarket] = []
    for e in _series_events():
        if str(e.get("slug", "")).startswith("arch-"):
            continue  # archived duplicate (degenerate prices) — skip
        year_hint = None
        if e.get("endDate"):
            try:
                year_hint = pd.to_datetime(e["endDate"], utc=True).year
            except Exception:  # noqa: BLE001
                year_hint = None
        win = parse_window(e.get("description"), year_hint=year_hint)
        if not win:
            continue
        ws, we = win
        dur = round((we - ws).total_seconds() / 86400.0, 1)
        if dur <= 0 or dur > 9:
            continue
        if durations is not None and not any(abs(dur - d) < 0.6 for d in durations):
            continue
        if W.utc_ts(ws) < W.utc_ts(floor) or W.utc_ts(we) > anchor:
            continue
        realized = int(((tv >= np.datetime64(W.utc_ts(ws).value, "ns"))
                        & (tv < np.datetime64(W.utc_ts(we).value, "ns"))).sum())
        brackets, winner = [], None
        for m in e.get("markets", []):
            label = m.get("groupItemTitle") or ""
            lo, hi = D._parse_bracket_bounds(label)
            toks, oc, op = m.get("clobTokenIds"), m.get("outcomes"), m.get("outcomePrices")
            if isinstance(toks, str):
                toks = json.loads(toks)
            if isinstance(oc, str):
                oc = json.loads(oc)
            if isinstance(op, str):
                op = json.loads(op)
            yi = ([o.lower() for o in oc].index("yes")
                  if oc and "yes" in [o.lower() for o in oc] else 0)
            brackets.append((lo, hi, label, toks[yi] if toks else None))
            if op and float(op[yi]) > 0.5:
                winner = label
        if not brackets or winner is None or any(b[3] is None for b in brackets):
            continue  # need a resolved winner + CLOB tokens for every bracket
        brackets.sort(key=lambda b: b[0])
        out.append(HB.ResolvedMarket(e["slug"], ws, we, realized, winner, brackets))
        if max_markets and len(out) >= max_markets:
            break
    return out


# --------------------------------------------------------------------------- #
# Strategy rules (a "variant" is one switch configuration)
# --------------------------------------------------------------------------- #
@dataclass
class Rules:
    name: str
    kelly_fraction: float = 0.25
    edge_entry: float = 0.04          # open/keep only while edge exceeds this
    edge_exit: float = 0.0            # sell the position once edge falls below this (model changed)
    max_sigma_ratio: float = 1.2      # sharpness gate for NEW entries (exits always allowed)
    max_per_market_frac: float = 0.40
    reinforce: bool = True            # add toward the Kelly target when under it
    reinforce_gap: float = 0.30       # only add if target exceeds current value by >30% (anti-churn)
    trim: bool = True                 # sell down toward target when over it
    trim_gap: float = 0.40
    take_profit: float | None = 0.90  # sell the whole position if held-side price >= this (lock gain)
    stop_on_negative_edge: bool = True
    joint_kelly: bool = False         # size the market JOINTLY (horse-race Kelly over the bracket
                                      # partition, YES-only) instead of independent per-bracket Kelly.
                                      # Corrects the correlation inflation; deploys less, more honest.


# Pre-defined variants compared head-to-head. The "joint" variants apply the correlation correction.
def default_variants() -> list[Rules]:
    return [
        Rules("V0_hold", reinforce=False, trim=False, take_profit=None,
              stop_on_negative_edge=False, edge_exit=-9.0),
        Rules("V1_rebalance", take_profit=None),
        Rules("V2_profit_take", take_profit=0.90),
        Rules("V3_reactive", take_profit=0.85, edge_exit=0.0, reinforce_gap=0.20),
        # --- correlation-corrected (joint Kelly over the partition) ---
        Rules("V0_joint_hold", joint_kelly=True, reinforce=False, trim=False, take_profit=None,
              stop_on_negative_edge=False, edge_exit=-9.0),
        Rules("V1_joint", joint_kelly=True, take_profit=None),
    ]


# --------------------------------------------------------------------------- #
# Joint (correlation-aware) Kelly sizing over the bracket partition
# --------------------------------------------------------------------------- #
def kelly_horserace(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> tuple[np.ndarray, float]:
    """Joint full-Kelly allocation over mutually-exclusive bracket outcomes (Smoczynski-Tomkins).

    The brackets of a market compete for ONE outcome, so independent per-bracket Kelly double-counts
    correlated bets and over-deploys capital (the "inflation" caveat). This sizes the JOINT bet: given
    model probs ``p`` (renormalized over the partition) and YES prices ``q``, it returns ``f[i]`` =
    fraction of bankroll to bet YES on bracket i, plus the cash reserve ``R``. Only positive-edge
    brackets (with a real price) are retained; the optimum keeps ``R`` in cash and bets
    ``f_i = p_i − R·q_i`` on the retained set, with Σf_i + R = 1. Betting YES across the partition
    spans the whole outcome space, so explicit NO bets are unnecessary (and would be redundant)."""
    p = np.asarray(p, float).copy()
    q = np.clip(np.asarray(q, float), eps, 1.0 - eps)
    s = p.sum()
    if s <= 0:
        return np.zeros(len(p)), 1.0
    p = p / s
    n = len(p)
    # candidates: positive edge AND a real (non-sentinel) price
    S = [i for i in range(n) if p[i] > q[i] and q[i] >= 0.01]
    S.sort(key=lambda i: p[i] / q[i], reverse=True)
    R = 1.0
    while S:
        P = float(sum(p[i] for i in S))
        Q = float(sum(q[i] for i in S))
        R = (1.0 - P) / (1.0 - Q) if (1.0 - Q) > eps else 0.0
        worst = S[-1]                              # lowest p_i/q_i in the retained set
        if p[worst] / q[worst] <= R + eps:
            S.pop()
        else:
            break
    f = np.zeros(n)
    for i in S:
        f[i] = max(0.0, p[i] - R * q[i])
    return f, R


# --------------------------------------------------------------------------- #
# Per-market path simulation
# --------------------------------------------------------------------------- #
@dataclass
class _Pos:
    shares: float = 0.0
    cost: float = 0.0   # cash paid in (basis)

    @property
    def value_at(self):  # convenience set later
        return self.shares


def _grid_nows(ws: dt.datetime, we: dt.datetime, step_h: float) -> list[dt.datetime]:
    span_h = (W.utc_ts(we) - W.utc_ts(ws)).total_seconds() / 3600.0
    n = max(1, int(span_h // step_h))
    return [(W.utc_ts(ws) + pd.Timedelta(hours=step_h * k)).to_pydatetime() for k in range(1, n + 1)]


@dataclass
class MarketTape:
    """The model signal + real market prices at every τ step for one market — computed ONCE, then
    replayed by each variant (variants differ only in trading rules, not in the model). This both
    cuts compute ~Nx (one Monte-Carlo pass, not one per variant) and makes variants exactly
    comparable (identical signal)."""
    mkt: HB.ResolvedMarket
    win_idx: int
    bracket_width: float
    dur_days: float
    steps: list  # each: {"tau", "p_model": np.ndarray, "yes": np.ndarray, "sigma_ratio"}


def build_tape(
    posts: pd.DataFrame,
    mkt: HB.ResolvedMarket,
    price_cache: dict,
    step_h: float,
    n_sims: int,
    rng: np.random.Generator,
) -> MarketTape:
    win_idx = next(i for i, (lo, hi, lab, _) in enumerate(mkt.brackets) if lab == mkt.winner)
    widths = [hi - lo + 1 for (lo, hi, _, _) in mkt.brackets if np.isfinite(hi)]
    bracket_width = float(np.median(widths)) if widths else 20.0
    span = W.utc_ts(mkt.window_end) - W.utc_ts(mkt.window_start)
    dur_days = span.total_seconds() / 86400.0
    gamma = CAL.gamma_for_duration(dur_days)
    steps = []
    for now in _grid_nows(mkt.window_start, mkt.window_end, step_h):
        yes = HB.market_probs_at(mkt, now, price_cache)  # raw forward-filled YES price/bracket
        # Skip steps with no real market: across mutually-exclusive brackets the YES prices sum to ~1
        # when a book exists; a near-zero sum means missing/sentinel prices (illiquid 2-day books or
        # archived dupes). Betting into the 0.0005 sentinel would mint fake 1000x edges — never do it.
        if float(np.sum(yes)) < 0.5:
            continue
        tau = float((W.utc_ts(now) - W.utc_ts(mkt.window_start)) / span)
        fit = M.fit_model(posts, now)
        fc = M.forecast(fit, mkt.window_start, mkt.window_end, n_sims=n_sims, rng=rng)
        tbl = M.bracket_probabilities(
            [D.Bracket(lab, lo, hi, None) for (lo, hi, lab, _) in mkt.brackets],
            fc.samples, gamma=gamma)
        steps.append({
            "tau": tau,
            "p_model": np.array([t["model_prob"] for t in tbl]),
            "yes": yes,
            "sigma_ratio": float(fc.samples.std()) / bracket_width,
        })
    return MarketTape(mkt, win_idx, bracket_width, dur_days, steps)


def replay_variant(
    tape: MarketTape,
    rules: Rules,
    bankroll: float,
    spread: float,
    late_spread: float | None = None,   # wider spread for τ>0.9 (stale late prices); None = same
    record_actions: bool = False,
) -> dict:
    """Replay one variant's continuous-rebalancing rules over a precomputed market tape; settle at
    resolution. Returns {staked, pnl, roi, n_trades, ...} aggregated over the week."""
    mkt = tape.mkt
    win_idx = tape.win_idx
    bounds = [(lo, hi, lab) for (lo, hi, lab, _) in mkt.brackets]
    cap = bankroll * rules.max_per_market_frac

    pos: dict[tuple[int, str], _Pos] = {}
    staked = 0.0
    realized_pnl = 0.0   # PnL from sells (cash in − basis of sold shares)
    n_trades = 0
    actions = []

    def half_spread(tau: float) -> float:
        s = spread if (late_spread is None or tau < 0.9) else late_spread
        return s / 2.0

    for step in tape.steps:
        tau = step["tau"]
        hs = half_spread(tau)
        p_model = step["p_model"]
        yes_prices = step["yes"]
        gate_ok = step["sigma_ratio"] <= rules.max_sigma_ratio

        # ---- per-step target stake by (bracket, side) ----
        # NAIVE: independent ¼-Kelly per bracket-side, each capped at max_per_market_frac (the sum can
        #        exceed bankroll → the correlation inflation).
        # JOINT: horse-race Kelly over the partition (YES-only); Σ targets ≤ kelly_fraction of bankroll
        #        by construction → correlation-aware, deploys less.
        target_stake: dict[tuple[int, str], float] = {}
        if rules.joint_kelly:
            f, _ = kelly_horserace(p_model, yes_prices)
            for i in range(len(mkt.brackets)):
                target_stake[(i, "OUI")] = bankroll * rules.kelly_fraction * float(f[i])
                target_stake[(i, "NON")] = 0.0   # spanned by YES on the partition; no redundant NO
        else:
            for i in range(len(mkt.brackets)):
                yq = float(yes_prices[i])
                if yq <= 0.0 or yq >= 1.0:
                    continue
                for side, price, p_side in (("OUI", yq, float(p_model[i])),
                                            ("NON", 1.0 - yq, 1.0 - float(p_model[i]))):
                    edge = p_side - price
                    target_stake[(i, side)] = min(
                        bankroll * rules.kelly_fraction * max(edge, 0.0) / (1.0 - price), cap)

        for i in range(len(mkt.brackets)):
            p_in = float(p_model[i])
            yq = float(yes_prices[i])
            if yq <= 0.0 or yq >= 1.0:
                continue
            for side, price, p_side in (("OUI", yq, p_in), ("NON", 1.0 - yq, 1.0 - p_in)):
                key = (i, side)
                p = pos.get(key)
                cur_val = (p.shares * price) if p else 0.0   # mark-to-market value
                edge = p_side - price

                # ---- exits first (always allowed, even past the sharpness gate) ----
                if p and p.shares > 0:
                    sell_all = False
                    why = ""
                    if rules.take_profit is not None and price >= rules.take_profit:
                        sell_all, why = True, "take_profit"
                    elif rules.stop_on_negative_edge and edge < rules.edge_exit:
                        sell_all, why = True, "edge_exit"
                    if sell_all:
                        proceeds = p.shares * max(price - hs, 0.0)
                        realized_pnl += proceeds - p.cost
                        n_trades += 1
                        if record_actions:
                            actions.append({"slug": mkt.slug, "tau": round(float(tau), 3),
                                             "act": why, "tranche": bounds[i][2], "side": side,
                                             "price": price, "shares": -p.shares})
                        pos.pop(key, None)
                        continue
                    if rules.trim and cur_val > 0:
                        target = target_stake.get(key, 0.0)
                        if target <= 0 or cur_val > target * (1.0 + rules.trim_gap):
                            # sell down to target (or fully if target 0)
                            keep_val = max(target, 0.0)
                            sell_val = cur_val - keep_val
                            frac = min(1.0, sell_val / cur_val)
                            sh = p.shares * frac
                            proceeds = sh * max(price - hs, 0.0)
                            realized_pnl += proceeds - p.cost * frac
                            p.cost *= (1.0 - frac)
                            p.shares *= (1.0 - frac)
                            n_trades += 1
                            if record_actions:
                                actions.append({"slug": mkt.slug, "tau": round(float(tau), 3),
                                                 "act": "trim", "tranche": bounds[i][2], "side": side,
                                                 "price": price, "shares": -sh})
                            if p.shares <= 1e-9:
                                pos.pop(key, None)
                            continue

                # ---- entries / reinforcement (need the sharpness gate; positive-edge/Kelly target) ----
                if not gate_ok:
                    continue
                target = target_stake.get(key, 0.0)
                # naive: gate on edge>edge_entry; joint: the Kelly set already defines what to bet
                entry_ok = target > 0 and (rules.joint_kelly or edge > rules.edge_entry)
                if not entry_ok:
                    continue
                if p is None or p.shares == 0:
                    add = target                       # ENTER
                    act = "enter"
                elif rules.reinforce and cur_val < target * (1.0 - rules.reinforce_gap):
                    add = target - cur_val             # REINFORCE toward target
                    act = "reinforce"
                else:
                    continue
                if add < 1.0:                           # drop dust trades
                    continue
                pe = min(price + hs, 0.999)
                sh = add / pe
                p = pos.setdefault(key, _Pos())
                p.shares += sh
                p.cost += add
                staked += add
                n_trades += 1
                if record_actions:
                    actions.append({"slug": mkt.slug, "tau": round(float(tau), 3), "act": act,
                                    "tranche": bounds[i][2], "side": side, "price": price, "shares": sh})

    # ---- settle remaining open positions at resolution ----
    for (i, side), p in pos.items():
        win = (side == "OUI" and i == win_idx) or (side == "NON" and i != win_idx)
        payout = p.shares * (1.0 if win else 0.0)
        realized_pnl += payout - p.cost

    return {"slug": mkt.slug, "week_end": mkt.window_end, "dur_days": round(tape.dur_days, 1),
            "staked": staked, "pnl": realized_pnl,
            "roi": (realized_pnl / staked) if staked > 0 else 0.0,
            "n_trades": n_trades, "won": realized_pnl > 0, "actions": actions}


# --------------------------------------------------------------------------- #
# Top-level run: every variant over every resolved market
# --------------------------------------------------------------------------- #
@dataclass
class PathBacktest:
    records: pd.DataFrame      # one row per (variant, market)
    by_variant: pd.DataFrame
    by_variant_duration: pd.DataFrame
    actions: pd.DataFrame      # detailed action log (optional)


def run_path(
    posts: pd.DataFrame,
    anchor_end: dt.datetime,
    variants: list[Rules] | None = None,
    durations: tuple[float, ...] | None = None,
    step_h: float = 2.0,
    n_sims: int = 2500,
    bankroll: float = 1000.0,
    spread: float = 0.02,
    late_spread: float | None = 0.04,
    max_markets: int | None = None,
    record_actions: bool = True,
    seed: int = 17,
    progress: bool = True,
) -> PathBacktest:
    variants = variants or default_variants()
    markets = enumerate_resolved_series(posts, anchor_end, durations=durations, max_markets=max_markets)
    if progress:
        print(f"resolved markets: {len(markets)} "
              f"(durations: {sorted({m.window_end and round((W.utc_ts(m.window_end)-W.utc_ts(m.window_start)).total_seconds()/86400,1) for m in markets})})",
              flush=True)
    rows, acts = [], []
    for mi, mkt in enumerate(markets):
        price_cache = {tok: HB.fetch_prices(tok, mkt.window_start, mkt.window_end)
                       for (_, _, _, tok) in mkt.brackets}
        # ONE Monte-Carlo pass per market; every variant replays the same tape.
        rng = np.random.default_rng(seed + mi)
        tape = build_tape(posts, mkt, price_cache, step_h, n_sims, rng)
        for rules in variants:
            r = replay_variant(tape, rules, bankroll, spread,
                               late_spread=late_spread, record_actions=record_actions)
            rows.append({"variant": rules.name, **{k: r[k] for k in
                        ("slug", "week_end", "dur_days", "staked", "pnl", "roi", "n_trades", "won")}})
            for a in r["actions"]:
                acts.append({"variant": rules.name, **a})
        if progress:
            print(f"[{mi+1}/{len(markets)}] {mkt.slug} done", flush=True)

    rec = pd.DataFrame(rows)
    if rec.empty:
        empty = pd.DataFrame()
        return PathBacktest(rec, empty, empty, pd.DataFrame(acts))

    def _agg(d: pd.DataFrame) -> pd.Series:
        st = d["staked"].sum()
        return pd.Series({
            "n_markets": int((d["staked"] > 0).sum()),
            "total_staked": st,
            "total_pnl": d["pnl"].sum(),
            "roi_aggregate": (d["pnl"].sum() / st) if st > 0 else 0.0,
            "roi_mean_per_week": d.loc[d["staked"] > 0, "roi"].mean(),
            "roi_std_per_week": d.loc[d["staked"] > 0, "roi"].std(),
            "win_rate_weeks": d.loc[d["staked"] > 0, "won"].mean(),
            "avg_trades": d["n_trades"].mean(),
        })

    by_variant = rec.groupby("variant").apply(_agg).reset_index()
    by_vd = rec.groupby(["variant", "dur_days"]).apply(_agg).reset_index()
    return PathBacktest(rec, by_variant, by_vd, pd.DataFrame(acts))
