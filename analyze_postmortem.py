"""Post-mortem: how did the model behave vs the market on one resolved (or closing) market.

    python analyze_postmortem.py [slug]

Reads the richest snapshot in data/postmortem/<slug>/ (highest τ), determines the winning bracket from
the realized tweet count, and from trajectory.csv computes, over the window:
  * model vs market probability (and log-loss) on the WINNING bracket through time,
  * when the model first *locked* the winner (argmax & p≥0.5, and stayed),
  * a per-phase table, and a plot (model vs market on the winner, with tweet arrivals).
Saves analysis.csv + postmortem.png + prints a summary. Winner is inferred from the count, so it works
even before Gamma flips the resolution flag.
"""
import sys, json, datetime as dt
sys.path.insert(0, "src")
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SLUG = sys.argv[1] if len(sys.argv) > 1 else "elon-musk-of-tweets-june-27-june-29"
base = Path("backtest_data/postmortem") / SLUG
folders = sorted(base.glob("snapshot_*"))
if not folders:
    print(f"Aucun snapshot pour {SLUG}"); sys.exit(0)


def _tau(f):
    try:
        return json.loads((f / "meta.json").read_text()).get("tau_at_snapshot", 0)
    except Exception:  # noqa: BLE001
        return 0


folder = max(folders, key=_tau)
meta = json.loads((folder / "meta.json").read_text())
realized = meta["realized_in_window"]

winner = None
for b in meta["brackets"]:
    lo = b["low"]
    hi = b["high"] if b["high"] not in (None,) and np.isfinite(b["high"]) else float("inf")
    if lo <= realized <= hi:
        winner = b["label"]
        break

traj = pd.read_csv(folder / "trajectory.csv")
wt = traj[traj["label"] == winner].sort_values("tau").copy()
wt["model_ll"] = -np.log(wt["model_prob"].clip(1e-6, 1))
wt["market_ll"] = -np.log(wt["market_yes_price"].clip(1e-6, 1))

# when did the model lock the winner? (argmax model_prob == winner, p>=0.5, and stays so)
leader = traj.loc[traj.groupby("t")["model_prob"].idxmax()][["t", "label", "tau", "model_prob"]]
leader = leader.sort_values("tau")
locked_tau = None
ok = leader["label"] == winner
streak = ok & (leader["model_prob"] >= 0.5)
# first index from which all subsequent are True
arr = streak.values
for i in range(len(arr)):
    if arr[i] and arr[i:].all():
        locked_tau = float(leader["tau"].values[i])
        break

# per-phase summary on the winner
wt["phase"] = pd.cut(wt["tau"], [0, 0.33, 0.66, 1.01], labels=["début", "milieu", "fin"], right=False)
phase = wt.groupby("phase", observed=True).agg(
    model_prob=("model_prob", "mean"), market_prob=("market_yes_price", "mean"),
    model_logloss=("model_ll", "mean"), market_logloss=("market_ll", "mean")).reset_index()

wt.to_csv(folder / "analysis.csv", index=False)

# plot: model vs market probability on the winning bracket, with tweet arrivals
posts = pd.read_csv(folder / "posts.csv")
ws = pd.Timestamp(meta["window_start"]); we = pd.Timestamp(meta["window_end"])
span_h = (we - ws).total_seconds() / 3600.0
tw_tau = ((pd.to_datetime(posts["created_at"], utc=True) - ws).dt.total_seconds() / 3600.0 / span_h)

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(wt["tau"], wt["model_prob"], "-", color="#1f77b4", lw=2, label=f"Modèle P({winner})")
ax.plot(wt["tau"], wt["market_yes_price"], "-", color="#E8704C", lw=2, label=f"Marché P({winner})")
for x in tw_tau:
    ax.axvline(x, color="gray", alpha=0.12, lw=0.6)
if locked_tau is not None:
    ax.axvline(locked_tau, color="green", ls="--", lw=1.2, label=f"modèle verrouille (τ={locked_tau:.2f})")
ax.set_xlabel("τ (fraction de la fenêtre)"); ax.set_ylabel("probabilité de la tranche gagnante")
ax.set_ylim(0, 1); ax.legend(loc="best", fontsize=9)
ax.set_title(f"Post-mortem {SLUG} — gagnante {winner} ({realized} tweets) | barres grises = tweets")
plt.tight_layout(); plt.savefig(folder / "postmortem.png", dpi=110)

print(f"=== POST-MORTEM {SLUG} ===")
print(f"snapshot: {folder.name}  τ={meta['tau_at_snapshot']}  closed={meta['closed']}")
print(f"réalisé={realized} tweets → tranche GAGNANTE = {winner}")
print(f"modèle verrouille la gagnante à τ={locked_tau}" if locked_tau is not None
      else "modèle n'a jamais verrouillé la gagnante (argmax & p≥0.5 stable)")
mll, mkll = wt["model_ll"].mean(), wt["market_ll"].mean()
print(f"log-loss moyen sur la gagnante — modèle {mll:.3f} vs marché {mkll:.3f} "
      f"({'modèle mieux' if mll < mkll else 'marché mieux'})")
print("\n--- par phase (proba sur la gagnante) ---")
pd.set_option("display.width", 160)
print(phase.round(3).to_string(index=False))
print(f"\nsaved: {folder/'analysis.csv'}, {folder/'postmortem.png'}")
