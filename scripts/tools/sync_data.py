"""Weekly local-data sync: pull the latest tweets incrementally + archive freshly-resolved markets.

    python scripts/tools/sync_data.py

Designed to be run on a schedule (launchd, weekly) or manually. Everything is incremental and
idempotent — nothing already stored is ever re-downloaded:
  * tweets  → data.ensure_history (fetched_ranges bookkeeping; only missing ranges hit XTracker)
  * markets → archive.archive_recent (skips markets whose `enriched_at` is set; 1-min prices
              YES+NO + trade tape + Gamma scalars for anything newly resolved)

Keeps the local cache complete even during periods when the Streamlit app isn't opened, so
backtests/model updates always run fully offline on up-to-date data.
"""
import sys, datetime as dt
import warnings
warnings.filterwarnings("ignore")
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
sys.path.insert(0, "src")

import pandas as pd
from tweetanalyst import archive as A
from tweetanalyst import data as D

now = dt.datetime.now(dt.timezone.utc)
print(f"--- sync {now:%Y-%m-%d %H:%M} UTC ---", flush=True)

# 1. tweets: incremental refresh (only fetches ranges not already covered)
posts = D.ensure_history("elonmusk")
print(f"tweets: {len(posts):,} en local (dernier: {posts['created_at'].max()})", flush=True)

# 2. markets: archive anything newly resolved (skips everything already enriched)
res = A.archive_recent(handle="elonmusk", lookback=12)
print(f"marchés: {res['scanned']} scannés · {len(res['new'])} nouveaux archivés "
      f"({res['points_added']:,} pts prix, {res['trades_added']:,} trades) · {res['skipped']} déjà en stock")
if res["new"]:
    print("nouveaux:", ", ".join(res["new"]))

# 3. regenerate the tau-grid CSVs (power the reliability table + Backtest τ page) — only for a
#    duration that actually gained a resolved market, so we never recompute for nothing.
import subprocess
_con = D._conn()
_arch = pd.read_sql_query("SELECT duration_days FROM resolved_markets", _con)
_con.close()
for _dur, _csv in [("2", "backtest_data/tau_backtest_2d.csv"), ("7", "backtest_data/tau_backtest_7d.csv")]:
    n_arch = int(((_arch["duration_days"] < 4) if _dur == "2" else (_arch["duration_days"] >= 4)).sum())
    n_csv = pd.read_csv(_csv)["slug"].nunique() if _os.path.exists(_csv) else 0
    if n_arch > n_csv:
        print(f"τ-backtest {_dur}j : {n_csv} → {n_arch} marchés, régénération…", flush=True)
        subprocess.run([sys.executable, "scripts/backtests/run_tau_backtest.py", _dur], check=False)
    else:
        print(f"τ-backtest {_dur}j : à jour ({n_csv} marchés)")
print("OK")
