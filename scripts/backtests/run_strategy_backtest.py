"""Backtest the fractional-Kelly strategy on resolved markets at real historical prices.

    python run_strategy_backtest.py            # default ¼-Kelly, edge>4pts
"""
import sys, warnings, datetime as dt, time
warnings.filterwarnings("ignore")
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
sys.path.insert(0, "src")
import pandas as pd
from tweetanalyst import data as D, histbacktest as HB

ANCHOR = dt.datetime(2026, 6, 26, 16, 0, tzinfo=dt.timezone.utc)
posts = D.load_posts("elonmusk", start=dt.datetime(2025, 10, 20, tzinfo=dt.timezone.utc), end=ANCHOR)
print("posts:", len(posts), flush=True)

t0 = time.time()
rec, g = HB.strategy_backtest(
    posts, ANCHOR, n_weeks=16, taus=(0.4, 0.6, 0.8, 0.92),
    kelly_fraction=0.25, edge_threshold=0.04, bankroll=1000, n_sims=3000)
rec.to_csv("backtest_data/historical/strategy_backtest_records.csv", index=False)
g.to_csv("backtest_data/historical/strategy_backtest_by_tau.csv", index=False)

pd.set_option("display.width", 220, "display.max_columns", 20)
print(f"\n=== ROI par instant d'entrée τ  ({time.time()-t0:.0f}s) ===")
print(g.round(3).to_string(index=False))
print(f"\nGLOBAL: misé ${rec.staked.sum():,.0f}, PnL ${rec.pnl.sum():,.0f}, "
      f"ROI {100*rec.pnl.sum()/rec.staked.sum():.1f}%")
print("saved: strategy_backtest_records.csv, strategy_backtest_by_tau.csv")
