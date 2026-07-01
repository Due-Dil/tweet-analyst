"""Tau-grid diagnostic backtest on RESOLVED 2-day markets: walk every market on a fine, regular time
grid (not just a single high-confidence entry) and record, at each step, the model's full probability
vector against the real CLOB market prices and the eventual winner.

Answers three questions:
  1. CALIBRATION OVER TIME — at what tau does the model's top pick actually win at the rate it claims?
     What's the ROI of buying that pick's YES (real spread, held to close) at each tau?
  2. STRUCTURAL BIAS — does the model systematically OVER-rate the bracket nearest the live pace
     (rel_rank=0, its own leader) and UNDER-rate brackets farther away, and does that gap shrink as
     the window progresses (info hypothesis from the user)?
  3. MARKET MISPRICING PATTERN — are certain bracket positions (by rank, by tau) systematically over-
     or under-priced by the MARKET itself (vs realized outcome), and what ROI results from buying the
     model's edge against that mispricing, at real CLOB prices with a spread haircut?

    python run_tau_backtest.py
"""
import sys, warnings, datetime as dt, time, dataclasses
warnings.filterwarnings("ignore")
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from tweetanalyst import data as D, pathbacktest as PB, model as M, calibration as CAL, windows as W

ANCHOR = dt.datetime(2026, 6, 30, 12, 0, tzinfo=dt.timezone.utc)
DURATIONS = (2,)                                   # start with 2-day markets per the request
TAUS = np.round(np.arange(0.05, 1.0, 0.05), 3)     # ~every 2.4h on a 48h window
N_SIMS = 3000
SPREAD = 0.03
KELLY = 0.25
BANKROLL = 1000.0


def build_records(mkt, posts) -> list[dict]:
    win_idx = next(i for i, (lo, hi, lab, _) in enumerate(mkt.brackets) if lab == mkt.winner)
    n_b = len(mkt.brackets)
    widths = [hi - lo + 1 for (lo, hi, _, _) in mkt.brackets if np.isfinite(hi)]
    bw = float(np.median(widths)) if widths else 20.0
    span = W.utc_ts(mkt.window_end) - W.utc_ts(mkt.window_start)
    dur_days = span.total_seconds() / 86400.0
    gamma = CAL.gamma_for_duration(dur_days)
    fit = M.fit_model(posts, mkt.window_start)
    brs = [D.Bracket(lab, lo, hi, None) for (lo, hi, lab, _) in mkt.brackets]
    labels = [lab for (_, _, lab, _) in mkt.brackets]
    pc = {tok: PB.HB.fetch_prices(tok, mkt.window_start, mkt.window_end) for (_, _, _, tok) in mkt.brackets}
    rng = np.random.default_rng(7)

    rows = []
    for tau in TAUS:
        now = (W.utc_ts(mkt.window_start) + span * float(tau)).to_pydatetime()
        yes = PB.HB.market_probs_at(mkt, now, pc)
        if float(np.sum(yes)) < 0.5:
            continue
        fc = M.forecast(dataclasses.replace(fit, now=now), mkt.window_start, mkt.window_end,
                        n_sims=N_SIMS, rng=rng)
        tbl = M.bracket_probabilities(brs, fc.samples, gamma=gamma)
        p = np.array([t["model_prob"] for t in tbl])
        model_rank = (-p).argsort().argsort() + 1            # 1 = model's favourite
        market_rank = (-yes).argsort().argsort() + 1          # 1 = market's favourite
        leader_idx = int(np.argmax(p))
        for i in range(n_b):
            rows.append({
                "slug": mkt.slug, "dur_days": round(dur_days, 1), "tau": float(tau),
                "bracket": labels[i], "bracket_idx": i, "n_brackets": n_b,
                "rel_rank": i - leader_idx,                    # 0=model leader, +/-1 adjacent, ...
                "model_prob": float(p[i]), "yes_price": float(yes[i]),
                "edge": float(p[i] - yes[i]),
                "model_rank": int(model_rank[i]), "market_rank": int(market_rank[i]),
                "is_winner": bool(i == win_idx),
            })
    return rows


def main():
    t0 = time.time()
    posts = D.load_posts("elonmusk")
    print(f"posts: {len(posts)}", flush=True)
    markets = PB.enumerate_resolved_series(posts, ANCHOR, durations=DURATIONS)
    print(f"marchés {DURATIONS}j résolus: {len(markets)}", flush=True)

    all_rows = []
    for mi, mkt in enumerate(markets):
        all_rows.extend(build_records(mkt, posts))
        if (mi + 1) % 10 == 0:
            print(f"  [{mi+1}/{len(markets)}] ({time.time()-t0:.0f}s)", flush=True)

    df = pd.DataFrame(all_rows)
    out_path = "backtest_data/tau_backtest_2d.csv"
    df.to_csv(out_path, index=False)
    print(f"\nsaved: {out_path}  ({len(df)} lignes, {df['slug'].nunique()} marchés, {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
