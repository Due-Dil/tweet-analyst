"""Sequential, continuously-rebalanced strategy backtest on resolved Polymarket markets.

Walks each resolved market on a fine τ grid (the tool "watches tweets all the time"), carrying a
position book it can enter / reinforce / trim / exit / take-profit at any moment — with a real
bid-ask spread haircut and real CLOB prices. Compares variants and breaks results down by duration.

    python run_pathbacktest.py                 # default: all durations, τ=2h, recent history
    python run_pathbacktest.py 7 6 12          # durations=7d, step=6h, max 12 markets (quick)
"""
import sys, warnings, datetime as dt, time
warnings.filterwarnings("ignore")
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
sys.path.insert(0, "src")
import pandas as pd
from tweetanalyst import data as D, pathbacktest as PB

ANCHOR = dt.datetime(2026, 6, 28, 12, 0, tzinfo=dt.timezone.utc)

# Optional CLI: [durations_csv] [step_h] [max_markets]
durations = None
step_h = 2.0
max_markets = None
if len(sys.argv) > 1 and sys.argv[1] not in ("-", "all"):
    durations = tuple(float(x) for x in sys.argv[1].split(","))
if len(sys.argv) > 2:
    step_h = float(sys.argv[2])
if len(sys.argv) > 3:
    max_markets = int(sys.argv[3])

posts = D.load_posts("elonmusk")
print(f"posts: {len(posts)}  ({posts.created_at.min()} → {posts.created_at.max()})", flush=True)

t0 = time.time()
res = PB.run_path(posts, ANCHOR, durations=durations, step_h=step_h,
                  n_sims=2500, max_markets=max_markets, record_actions=True)

res.records.to_csv("backtest_data/path_strategy/pathbacktest_records.csv", index=False)
res.by_variant.to_csv("backtest_data/path_strategy/pathbacktest_by_variant.csv", index=False)
res.by_variant_duration.to_csv("backtest_data/path_strategy/pathbacktest_by_variant_duration.csv", index=False)
res.actions.to_csv("backtest_data/path_strategy/pathbacktest_actions.csv", index=False)

pd.set_option("display.width", 220, "display.max_columns", 30)
print(f"\n=== PAR VARIANTE (toutes durées)  ({time.time()-t0:.0f}s) ===", flush=True)
print(res.by_variant.round(3).to_string(index=False), flush=True)
print("\n=== PAR VARIANTE × DURÉE ===", flush=True)
print(res.by_variant_duration.round(3).to_string(index=False), flush=True)
print("\nsaved: pathbacktest_records.csv, _by_variant.csv, _by_variant_duration.csv, _actions.csv",
      flush=True)
