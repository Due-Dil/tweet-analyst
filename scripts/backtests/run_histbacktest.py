"""Run the historical backtest against resolved markets and save results + a plot."""
import sys, warnings, datetime as dt
warnings.filterwarnings("ignore")
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
sys.path.insert(0, "src")
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tweetanalyst import data as D, histbacktest as HB

ANCHOR = dt.datetime(2026, 6, 26, 16, 0, tzinfo=dt.timezone.utc)
posts = D.load_posts("elonmusk", start=dt.datetime(2025, 10, 20, tzinfo=dt.timezone.utc), end=ANCHOR)
print("posts:", len(posts), flush=True)

markets = HB.enumerate_resolved(posts, ANCHOR, n_weeks=18)
print("resolved markets:", len(markets), flush=True)

taus = tuple(np.round(np.arange(0.0, 0.96, 0.08), 2))
res = HB.run(posts, markets, taus=taus, n_sims=4000, gamma=1.45)

res.records.to_csv("backtest_data/historical/histbacktest_records.csv", index=False)
res.by_tau.to_csv("backtest_data/historical/histbacktest_by_tau.csv", index=False)

pd.set_option("display.width", 200, "display.max_columns", 30)
print("\n=== PAR INSTANT τ (fraction de la semaine ecoulee) ===", flush=True)
print(res.by_tau.round(3).to_string(index=False), flush=True)

# Plot: reliability vs edge over the week
g = res.by_tau
fig, ax1 = plt.subplots(figsize=(9, 5))
ax1.plot(g.tau, g.model_prob_true, "o-", color="#1f77b4", label="Modèle: proba sur bonne tranche")
ax1.plot(g.tau, g.market_prob_true, "s--", color="#ff7f0e", label="Marché: proba sur bonne tranche")
ax1.set_xlabel("fraction de la semaine écoulée (τ)")
ax1.set_ylabel("fiabilité (proba sur la tranche gagnante)")
ax1.set_ylim(0, 1)
ax2 = ax1.twinx()
ax2.bar(g.tau, g.pnl_per_trade, width=0.05, alpha=0.25, color="green", label="PnL moyen / semaine ($1 mise)")
ax2.axhline(0, color="gray", lw=0.6)
ax2.set_ylabel("edge / PnL moyen par semaine ($ par $1 misé)")
lines = ax1.get_legend_handles_labels()[0] + ax2.get_legend_handles_labels()[0]
labs = ax1.get_legend_handles_labels()[1] + ax2.get_legend_handles_labels()[1]
ax1.legend(lines, labs, loc="upper left", fontsize=8)
plt.title("Fiabilité du modèle vs edge disponible, au fil de la semaine (18 marchés résolus)")
plt.tight_layout()
plt.savefig("backtest_data/historical/histbacktest.png", dpi=110)
print("\nsaved: histbacktest_records.csv, histbacktest_by_tau.csv, histbacktest.png", flush=True)
