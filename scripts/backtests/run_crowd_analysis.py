"""Phase A crowd-behaviour analysis: tail mispricing by period + overreaction to bursts.

    python run_crowd_analysis.py
"""
import sys, warnings, datetime as dt, time
warnings.filterwarnings("ignore")
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
sys.path.insert(0, "src")
import pandas as pd
from tweetanalyst import data as D, crowd as C

ANCHOR = dt.datetime(2026, 6, 28, 12, 0, tzinfo=dt.timezone.utc)
pd.set_option("display.width", 200, "display.max_columns", 20)

posts = D.load_posts("elonmusk")
print(f"posts: {len(posts)}", flush=True)

t0 = time.time()
obs = C.collect_observations(posts, ANCHOR, progress=False)
obs.to_csv("backtest_data/crowd/crowd_observations.csv", index=False)
print(f"\nobservations (marché×tranche×τ): {len(obs)}  ({time.time()-t0:.0f}s)")

print("\n=== CALIBRATION par phase × bucket de prix (écart = freq_réelle − prix ; <0 = SURCOTÉ) ===")
cal = C.calibration(obs)
cal.to_csv("backtest_data/crowd/crowd_calibration.csv", index=False)
print(cal.round(3).to_string(index=False))

print("\n=== TAILS vs CENTRE par phase ===")
ts = C.tail_summary(obs)
ts.to_csv("backtest_data/crowd/crowd_tails.csv", index=False)
print(ts.round(3).to_string(index=False))

print("\n=== SURRÉACTION (jump → rendement futur) ===")
summ, ev = C.overreaction(posts, ANCHOR, progress=False)
ev.to_csv("backtest_data/crowd/crowd_overreaction_events.csv", index=False)
for k, v in summ.items():
    print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
print("\nLecture : reversion_score > 0 ⇒ le prix revient après un saut (surréaction). "
      "autocorr_lag1 < 0 ⇒ mean-reversion court terme.")
print("saved: crowd_observations.csv, crowd_calibration.csv, crowd_tails.csv, crowd_overreaction_events.csv")
