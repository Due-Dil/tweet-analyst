"""Freeze ALL data for one market into a timestamped folder, for a later model-vs-market post-mortem.

    python snapshot_market.py [slug]        # default: the June 27-29 market

Captures, into ``data/postmortem/<slug>/snapshot_<UTC>/``:
  * meta.json        — window, brackets, current/closed status, winner (if resolved), realized count
  * posts.csv        — every XTracker post in the counting window (the resolution truth)
  * prices.csv       — full CLOB price history per bracket (1-min fidelity), long format
  * trajectory.csv   — at each τ checkpoint: model prob vs market YES price per bracket (the core of
                       "how did the model behave vs the market"), with n_obs and realized-so-far

Re-run AFTER the close (window end) to get the definitive, resolved snapshot. Nothing is overwritten —
each run writes a new timestamped folder.
"""
import sys, json, datetime as dt
sys.path.insert(0, "src")
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from tweetanalyst import data as D, model as M, calibration as CAL, windows as W, pathbacktest as PB

SLUG = sys.argv[1] if len(sys.argv) > 1 else "elon-musk-of-tweets-june-27-june-29"
GRID_MIN = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0   # model-trajectory grid step (minutes)
GAMMA, CLOB = "https://gamma-api.polymarket.com", "https://clob.polymarket.com/prices-history"
DATA_API = "https://data-api.polymarket.com"


def fetch_clob(token, start_ts, end_ts, fidelity=1):
    try:
        r = requests.get(CLOB, params={"market": token, "startTs": int(start_ts),
                                       "endTs": int(end_ts), "fidelity": fidelity}, timeout=40)
        return r.json().get("history", []) if r.status_code == 200 else []
    except Exception:  # noqa: BLE001
        return []


def fetch_trades(condition_id, max_pages=60):
    """All trades for one bracket (its conditionId), tick-level (timestamp to the second)."""
    out, off = [], 0
    for _ in range(max_pages):
        try:
            d = requests.get(f"{DATA_API}/trades", params={"market": condition_id, "limit": 500,
                                                           "offset": off}, timeout=30).json()
        except Exception:  # noqa: BLE001
            break
        if not isinstance(d, list) or not d:
            break
        out.extend(d)
        off += len(d)
        if len(d) < 500:
            break
    return out


def main():
    now = dt.datetime.now(dt.timezone.utc)
    ev = requests.get(f"{GAMMA}/events", params={"slug": SLUG}, timeout=20).json()
    if not ev:
        print(f"Aucun event pour slug={SLUG}"); return
    e = ev[0]
    ws, we = PB.parse_window(e.get("description"))
    span_s = (W.utc_ts(we) - W.utc_ts(ws)).total_seconds()
    dur_days = span_s / 86400.0
    gamma = CAL.gamma_for_duration(dur_days)

    # --- brackets + tokens + current prices + winner ---
    brackets, winner = [], None
    for m in e.get("markets", []):
        lab = m.get("groupItemTitle") or ""
        lo, hi = D._parse_bracket_bounds(lab)
        toks, oc, op = m.get("clobTokenIds"), m.get("outcomes"), m.get("outcomePrices")
        toks = json.loads(toks) if isinstance(toks, str) else toks
        oc = json.loads(oc) if isinstance(oc, str) else oc
        op = json.loads(op) if isinstance(op, str) else op
        yi = ([o.lower() for o in oc].index("yes") if oc and "yes" in [o.lower() for o in oc] else 0)
        cur_yes = float(op[yi]) if op else None
        if op and cur_yes is not None and cur_yes > 0.5:
            winner = lab
        brackets.append({"label": lab, "low": lo, "high": hi, "condition_id": m.get("conditionId"),
                         "yes_token": toks[yi] if toks else None, "cur_yes_price": cur_yes})
    brackets.sort(key=lambda b: b["low"])

    # --- posts (resolution truth) ---
    D.ensure_history("elonmusk", days=200)
    posts = D.load_posts("elonmusk", start=ws - dt.timedelta(days=120), end=now)
    in_win = posts[(posts.created_at >= W.utc_ts(ws)) & (posts.created_at < W.utc_ts(we))]
    realized = int(len(in_win))

    # --- output folder ---
    out = Path("backtest_data/postmortem") / SLUG / f"snapshot_{now.strftime('%Y%m%dT%H%M%SZ')}"
    out.mkdir(parents=True, exist_ok=True)

    # --- CLOB price history (full, 1-min) per bracket ---
    price_rows, price_series = [], {}
    for b in brackets:
        tok = b["yes_token"]
        if not tok:
            continue
        hist = fetch_clob(tok, W.utc_ts(ws).timestamp() - 3600, min(W.utc_ts(we).timestamp(),
                          now.timestamp()) + 3600)
        ser = [(int(h["t"]), float(h["p"])) for h in hist]
        price_series[b["label"]] = ser
        for t, p in ser:
            price_rows.append({"label": b["label"], "t": t,
                               "utc": dt.datetime.fromtimestamp(t, dt.timezone.utc).isoformat(), "p": p})
    pd.DataFrame(price_rows).to_csv(out / "prices.csv", index=False)

    # --- TICK-level trades per bracket (timestamps to the second) + YES tick series ---
    trade_rows, tick_series = [], {}
    for b in brackets:
        cid = b.get("condition_id")
        ticks = []
        for tr in (fetch_trades(cid) if cid else []):
            try:
                ts_i = int(tr["timestamp"]); price = float(tr["price"])
                is_yes = str(tr.get("outcome", "")).lower() == "yes"
                yes_p = price if is_yes else 1.0 - price
            except Exception:  # noqa: BLE001
                continue
            ticks.append((ts_i, yes_p))
            trade_rows.append({"label": b["label"], "t": ts_i,
                               "utc": dt.datetime.fromtimestamp(ts_i, dt.timezone.utc).isoformat(),
                               "outcome": tr.get("outcome"), "price": price, "yes_price": round(yes_p, 4),
                               "size": tr.get("size"), "side": tr.get("side"),
                               "wallet": tr.get("proxyWallet")})
        tick_series[b["label"]] = sorted(ticks)
    pd.DataFrame(trade_rows).to_csv(out / "trades.csv", index=False)

    def yes_tick_at(label, ts):  # last traded YES price at/<= ts (tick-level), fallback to 1-min bars
        ser = tick_series.get(label) or []
        prior = [p for (t, p) in ser if t <= ts]
        if prior:
            return prior[-1]
        bars = price_series.get(label, [])
        pb = [p for (t, p) in bars if t <= ts]
        return pb[-1] if pb else (bars[0][1] if bars else None)

    # --- FINE model trajectory: re-evaluate at EVERY tweet arrival + a 10-min grid (the model only
    #     moves at tweets + smooth time-decay, so this faithfully reconstructs its live reaction). ---
    rng = np.random.default_rng(11)
    end_cap = min(W.utc_ts(we), W.utc_ts(now))
    grid = pd.date_range(W.utc_ts(ws), end_cap, freq=f"{int(GRID_MIN)}min", tz="UTC")
    tweet_times = in_win["created_at"] + pd.Timedelta(seconds=1)  # just after each tweet lands
    eval_ts = sorted({int(x.timestamp()) for x in list(grid) + list(tweet_times)
                      if W.utc_ts(ws) < x <= end_cap})
    brs = [D.Bracket(b["label"], b["low"], b["high"], None) for b in brackets]
    traj = []
    for ts_i in eval_ts:
        nowk = dt.datetime.fromtimestamp(ts_i, dt.timezone.utc)
        tau = (ts_i - W.utc_ts(ws).timestamp()) / span_s
        fit = M.fit_model(posts, nowk)
        fc = M.forecast(fit, ws, we, n_sims=3000, rng=rng)
        tbl = M.bracket_probabilities(brs, fc.samples, gamma=gamma)
        n_obs_now = int(((posts.created_at >= W.utc_ts(ws)) & (posts.created_at < nowk)).sum())
        for b, t in zip(brackets, tbl):
            traj.append({"t": ts_i, "utc": nowk.isoformat(), "tau": round(tau, 4), "n_obs": n_obs_now,
                         "label": b["label"], "model_prob": round(t["model_prob"], 5),
                         "market_yes_price": yes_tick_at(b["label"], ts_i)})
    pd.DataFrame(traj).to_csv(out / "trajectory.csv", index=False)
    rows_done = len(eval_ts)

    # --- posts in window ---
    in_win.assign(created_at=in_win.created_at.astype(str)).to_csv(out / "posts.csv", index=False)

    meta = {"slug": SLUG, "snapshot_utc": now.isoformat(), "window_start": ws.isoformat(),
            "window_end": we.isoformat(), "dur_days": round(dur_days, 2), "gamma_applied": gamma,
            "closed": bool(e.get("closed")), "winner": winner, "realized_in_window": realized,
            "tau_at_snapshot": round((W.utc_ts(now) - W.utc_ts(ws)).total_seconds() / span_s, 3),
            "n_brackets": len(brackets), "brackets": brackets, "checkpoints": rows_done,
            "n_trades": len(trade_rows)}
    (out / "meta.json").write_text(json.dumps(meta, indent=2, default=str), encoding="utf-8")

    print(f"✅ Snapshot écrit : {out}")
    print(f"   τ={meta['tau_at_snapshot']}  realized_so_far={realized}  closed={meta['closed']}  "
          f"winner={winner}")
    print(f"   fichiers: meta.json, posts.csv ({realized}), prices.csv ({len(price_rows)} barres 1-min), "
          f"trades.csv ({len(trade_rows)} ticks), trajectory.csv ({len(traj)} = {rows_done} instants × "
          f"{len(brackets)} tranches)")
    if not meta["closed"]:
        print(f"   ⚠️ Marché encore ouvert (clôture {we.isoformat()}). Relance ce script après la "
              f"clôture pour la version finale RÉSOLUE.")


if __name__ == "__main__":
    main()
