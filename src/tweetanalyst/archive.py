"""Proactive archival of RESOLVED Elon markets into the local cache, so backtests stay fully offline
and survive Polymarket's limited CLOB price-history retention.

Polymarket's ``prices-history`` endpoint only serves recent data at fine granularity; once a market is
old enough its minute-level history is dropped. This module captures each resolved market at **1-minute
fidelity** (the finest available — always down-sampleable to hourly, never the reverse):

  * YES *and* NO token price series per bracket    -> table ``clob_prices``   (INSERT OR IGNORE)
  * executed trade tape per bracket market          -> table ``market_trades`` (real fills/volume)
  * resolution + liquidity metadata                 -> table ``resolved_markets``
    (window, winner, realized, bracket→token map, event volume / open-interest / per-bracket spread)

Entry points:
  * ``archive_recent(...)``  — cheap incremental sweep of the newest resolved markets (wired into the
    app: a freshly-closed market is captured automatically on the next launch).
  * ``archive_all(...)``     — full backfill across every scoreable resolved market (run once / weekly).

Idempotent: a market whose ``resolved_markets.enriched_at`` is set is skipped (no re-fetch). ``INSERT
OR IGNORE`` on prices/trades means re-runs never duplicate rows.
"""
from __future__ import annotations

import datetime as dt
import json
import time

import pandas as pd
import requests

from . import data as D
from . import histbacktest as HB
from . import pathbacktest as PB
from . import windows as W

ARCHIVE_FIDELITY = 1  # minutes — finest granularity the CLOB endpoint serves
GAMMA = "https://gamma-api.polymarket.com"
TRADES = "https://data-api.polymarket.com/trades"


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
def _ensure_meta_table(con) -> None:
    con.execute("""CREATE TABLE IF NOT EXISTS resolved_markets(
                       slug         TEXT PRIMARY KEY,
                       window_start TEXT, window_end TEXT, duration_days REAL,
                       winner       TEXT, realized INTEGER,
                       brackets_json TEXT,           -- [[lo, hi, label, yes_token], ...]
                       n_price_points INTEGER, fidelity_min INTEGER,
                       archived_at  TEXT)""")
    # additive migrations (safe on an existing table)
    for col, decl in [("extra_json", "TEXT"), ("n_trades", "INTEGER"), ("enriched_at", "TEXT")]:
        try:
            con.execute(f"ALTER TABLE resolved_markets ADD COLUMN {col} {decl}")
        except Exception:  # noqa: BLE001  (column already exists)
            pass
    con.execute("""CREATE TABLE IF NOT EXISTS market_trades(
                       condition_id TEXT, slug TEXT, ts INTEGER, side TEXT,
                       size REAL, price REAL, asset TEXT, wallet TEXT,
                       PRIMARY KEY(asset, ts, wallet, price, size, side))""")
    con.execute("CREATE INDEX IF NOT EXISTS idx_trades_slug ON market_trades(slug)")


def _is_enriched(slug: str, con) -> bool:
    row = con.execute("SELECT enriched_at FROM resolved_markets WHERE slug=?", (slug,)).fetchone()
    return row is not None and row[0] is not None


def is_archived(slug: str, con=None) -> bool:
    own = con is None
    con = con or D._conn()
    _ensure_meta_table(con)
    hit = _is_enriched(slug, con)
    if own:
        con.close()
    return hit


# --------------------------------------------------------------------------- #
# Remote fetch helpers
# --------------------------------------------------------------------------- #
def _fetch_token_fine(token: str, t0: dt.datetime, t1: dt.datetime, con,
                      fidelity: int = ARCHIVE_FIDELITY) -> int:
    """Fetch one token's (YES or NO) price history at ``fidelity`` minutes and store it. Always pulls
    the fine series (upgrades an hourly-cached token); (token, t) PK ignores collisions."""
    if token is None:
        return 0
    start = int(W.utc_ts(t0).timestamp()) - 3600
    end = int(W.utc_ts(t1).timestamp()) + 3600
    try:
        r = requests.get(HB.CLOB, params={"market": token, "startTs": start, "endTs": end,
                                          "fidelity": fidelity}, timeout=30)
        hist = r.json().get("history", []) if r.status_code == 200 else []
    except Exception:  # noqa: BLE001
        hist = []
    rows = [(token, int(h["t"]), float(h["p"])) for h in hist]
    if rows:
        with con:
            con.executemany("INSERT OR IGNORE INTO clob_prices VALUES(?,?,?)", rows)
    else:
        with con:
            con.execute("INSERT OR IGNORE INTO clob_prices VALUES(?,?,?)", (token, 0, -1.0))
    time.sleep(0.07)
    return len(rows)


def _gamma_event_markets(slug: str) -> tuple[dict, dict]:
    """Per-bracket {label: {yes, no, condition_id, volume, spread}} + event scalars, from Gamma."""
    try:
        e = requests.get(GAMMA + "/events", params={"slug": slug}, timeout=20).json()
    except Exception:  # noqa: BLE001
        return {}, {}
    ev = e[0] if isinstance(e, list) and e else (e if isinstance(e, dict) else {})
    out = {}
    for m in ev.get("markets", []):
        label = m.get("groupItemTitle") or ""
        toks = m.get("clobTokenIds")
        toks = json.loads(toks) if isinstance(toks, str) else (toks or [])
        oc = m.get("outcomes")
        oc = json.loads(oc) if isinstance(oc, str) else (oc or [])
        yi = ([o.lower() for o in oc].index("yes")
              if oc and "yes" in [o.lower() for o in oc] else 0)
        out[label] = {
            "yes": toks[yi] if toks else None,
            "no": toks[1 - yi] if toks and len(toks) > 1 else None,
            "condition_id": m.get("conditionId"),
            "volume": _f(m.get("volume")), "spread": _f(m.get("spread")),
        }
    scalars = {"event_volume": _f(ev.get("volume")), "open_interest": _f(ev.get("openInterest"))}
    return out, scalars


def _fetch_trades(condition_id: str, slug: str, con, max_pages: int = 12) -> int:
    """Paginate the executed-trade tape for one bracket market (by conditionId) into market_trades."""
    if not condition_id:
        return 0
    n = 0
    for page in range(max_pages):
        try:
            r = requests.get(TRADES, params={"market": condition_id, "limit": 500,
                                             "offset": page * 500}, timeout=20)
            d = r.json() if r.status_code == 200 else []
        except Exception:  # noqa: BLE001
            break
        if not isinstance(d, list) or not d:
            break
        rows = [(condition_id, slug, int(t["timestamp"]), t.get("side"), _f(t.get("size")),
                 _f(t.get("price")), t.get("asset"), t.get("proxyWallet")) for t in d if t.get("timestamp")]
        with con:
            con.executemany(
                "INSERT OR IGNORE INTO market_trades VALUES(?,?,?,?,?,?,?,?)", rows)
        n += len(d)
        if len(d) < 500:
            break
        time.sleep(0.1)
    return n


# --------------------------------------------------------------------------- #
# Archive one / sweep
# --------------------------------------------------------------------------- #
def archive_one(mkt: HB.ResolvedMarket, con=None, fidelity: int = ARCHIVE_FIDELITY,
                force: bool = False, with_trades: bool = True) -> dict:
    """Archive one resolved market: YES+NO price series (1-min) + trade tape + metadata/scalars.
    Skips (no network) if already enriched, unless ``force``."""
    own = con is None
    con = con or HB._price_conn()
    _ensure_meta_table(con)
    if not force and _is_enriched(mkt.slug, con):
        if own:
            con.close()
        return {"slug": mkt.slug, "status": "déjà archivé", "points": 0, "trades": 0}

    gmarkets, scalars = _gamma_event_markets(mkt.slug)
    n_pts = n_trades = 0
    bracket_extra = {}
    for (lo, hi, label, yes_tok) in mkt.brackets:
        info = gmarkets.get(label, {})
        no_tok = info.get("no")
        n_pts += _fetch_token_fine(yes_tok, mkt.window_start, mkt.window_end, con, fidelity)
        n_pts += _fetch_token_fine(no_tok, mkt.window_start, mkt.window_end, con, fidelity)
        if with_trades:
            n_trades += _fetch_trades(info.get("condition_id"), mkt.slug, con)
        bracket_extra[label] = {"no_token": no_tok, "condition_id": info.get("condition_id"),
                                "volume": info.get("volume"), "spread": info.get("spread")}

    dur = round((W.utc_ts(mkt.window_end) - W.utc_ts(mkt.window_start)).total_seconds() / 86400.0, 2)
    now_iso = dt.datetime.now(dt.timezone.utc).isoformat()
    extra = {**scalars, "brackets": bracket_extra}
    with con:
        con.execute("INSERT OR REPLACE INTO resolved_markets VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", (
            mkt.slug, W.utc_ts(mkt.window_start).isoformat(), W.utc_ts(mkt.window_end).isoformat(),
            dur, mkt.winner, int(mkt.realized),
            json.dumps([[b[0], b[1], b[2], b[3]] for b in mkt.brackets]),
            n_pts, fidelity, now_iso, json.dumps(extra), n_trades, now_iso))
    if own:
        con.close()
    return {"slug": mkt.slug, "status": "archivé", "points": n_pts, "trades": n_trades}


def _sweep(markets: list, fidelity: int, with_trades: bool, on_progress=None) -> dict:
    con = HB._price_conn()
    _ensure_meta_table(con)
    new, skipped, pts, trd = [], 0, 0, 0
    for i, mkt in enumerate(markets):
        if _is_enriched(mkt.slug, con):
            skipped += 1
        else:
            res = archive_one(mkt, con, fidelity, with_trades=with_trades)
            new.append(res["slug"])
            pts += res["points"]
            trd += res["trades"]
        if on_progress:
            on_progress(i + 1, len(markets), mkt.slug)
    con.close()
    return {"new": new, "skipped": skipped, "points_added": pts, "trades_added": trd,
            "scanned": len(markets)}


def archive_recent(handle: str = "elonmusk", lookback: int = 12,
                   fidelity: int = ARCHIVE_FIDELITY, with_trades: bool = True, on_progress=None) -> dict:
    """Cheap incremental sweep of the ``lookback`` newest resolved markets. Near-instant after the
    first pass since already-enriched markets are skipped."""
    posts = D.load_posts(handle)
    now = dt.datetime.now(dt.timezone.utc)
    markets = PB.enumerate_resolved_series(posts, now, durations=None, max_markets=lookback)
    return _sweep(markets, fidelity, with_trades, on_progress)


def archive_all(handle: str = "elonmusk", durations: tuple | None = None,
                fidelity: int = ARCHIVE_FIDELITY, with_trades: bool = True, on_progress=None) -> dict:
    """Full backfill across every scoreable resolved market (all durations by default)."""
    posts = D.load_posts(handle)
    now = dt.datetime.now(dt.timezone.utc)
    markets = PB.enumerate_resolved_series(posts, now, durations=durations)
    return _sweep(markets, fidelity, with_trades, on_progress)


def archive_status(con=None) -> pd.DataFrame:
    """What's archived so far (one row per market), newest first."""
    own = con is None
    con = con or D._conn()
    _ensure_meta_table(con)
    df = pd.read_sql_query(
        "SELECT slug, duration_days, winner, realized, n_price_points, n_trades, fidelity_min, "
        "enriched_at FROM resolved_markets ORDER BY window_end DESC", con)
    if own:
        con.close()
    return df
