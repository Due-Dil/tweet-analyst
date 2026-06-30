"""Backtest the overreaction rule: fade a price jump, exit after horizon_h (real prices, net spread).

    python run_fade_backtest.py
"""
import sys, warnings, datetime as dt, time
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
import pandas as pd
from tweetanalyst import data as D, crowd as C

ANCHOR = dt.datetime(2026, 6, 28, 12, 0, tzinfo=dt.timezone.utc)
pd.set_option("display.width", 200, "display.max_columns", 20)
posts = D.load_posts("elonmusk")
print(f"posts: {len(posts)}", flush=True)

t0 = time.time()
rows = []
# main comparison: up-fade (signal) vs down-fade (control), with/without burst filter
for direction, burst in [("up", False), ("up", True), ("down", False)]:
    s, _ = C.fade_backtest(posts, ANCHOR, direction=direction, burst_only=burst,
                           jump=0.08, horizon_h=6.0, spread=0.02)
    rows.append(s)
# grid over jump threshold & horizon for the up-fade (burst-confirmed)
for jmp in (0.06, 0.10, 0.15):
    for hz in (3.0, 6.0, 12.0):
        s, _ = C.fade_backtest(posts, ANCHOR, direction="up", burst_only=True,
                               jump=jmp, horizon_h=hz, spread=0.02)
        rows.append(s)

res = pd.DataFrame(rows)
res.to_csv("fade_backtest.csv", index=False)
print(f"\n=== Fade backtest ({time.time()-t0:.0f}s) — ret = rendement par $ misé, net de spread 2c ===")
cols = ["direction", "burst_only", "jump", "horizon_h", "n_trades", "roi_moyen", "roi_median", "hit_rate"]
print(res[cols].round(4).to_string(index=False))
print("\nLecture : 'up' burst_only = la règle (fader un pic post-burst). 'down' = contrôle. "
      "roi_moyen > 0 et hit_rate > 0.5 ⇒ edge réel. saved: fade_backtest.csv")
