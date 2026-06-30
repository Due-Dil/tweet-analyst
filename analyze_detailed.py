"""High-granularity post-mortem of one market: model vs market at tick/tweet resolution.

    python analyze_detailed.py [slug]

Uses the richest snapshot (trajectory + tick trades + posts). Produces:
  * winner bracket: model (fine steps) vs market (every trade tick), tweets marked
  * per-tweet reaction: how each tweet moved the model vs the market (CSV + scatter)
  * scores through time: model vs market log-loss on the winner + multiclass Brier
  * lead-lag: cross-correlation of model-change vs market-change (who moves first)
  * belief-mass evolution across all brackets (model & market)
Saves detailed CSVs + a multi-panel PNG.
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
folder = max(base.glob("snapshot_*"), key=lambda f: json.loads((f / "meta.json").read_text()).get("tau_at_snapshot", 0))
meta = json.loads((folder / "meta.json").read_text())
realized = meta["realized_in_window"]
ws = pd.Timestamp(meta["window_start"]); we = pd.Timestamp(meta["window_end"])
span_s = (we - ws).total_seconds()

winner = next(b["label"] for b in meta["brackets"]
              if b["low"] <= realized <= (b["high"] if b["high"] not in (None,) and np.isfinite(b["high"]) else 1e9))

traj = pd.read_csv(folder / "trajectory.csv")
trades = pd.read_csv(folder / "trades.csv")
posts = pd.read_csv(folder / "posts.csv")
post_ts = (pd.to_datetime(posts["created_at"], utc=True).astype("int64") // 10**9).values
post_ts = post_ts[(post_ts >= ws.timestamp()) & (post_ts < we.timestamp())]

def to_tau(ts):
    return (np.asarray(ts, float) - ws.timestamp()) / span_s

# ---------- 1. winner: model (fine) + market (ticks) ----------
wt = traj[traj["label"] == winner].sort_values("t")
wtr = trades[trades["label"] == winner].sort_values("t")
wtr_tau = to_tau(wtr["t"].values)

# market price (forward-filled ticks) on a common fine grid for scoring/lead-lag
grid_ts = np.arange(int(ws.timestamp()), int(we.timestamp()), 300)  # 5-min
def ff_price(label, ts_grid):
    tk = trades[trades["label"] == label].sort_values("t")
    if tk.empty:
        return np.full(len(ts_grid), np.nan)
    idx = np.searchsorted(tk["t"].values, ts_grid, side="right") - 1
    out = np.where(idx >= 0, tk["yes_price"].values[np.clip(idx, 0, len(tk) - 1)], np.nan)
    return out
def ff_model(label, ts_grid):
    tj = traj[traj["label"] == label].sort_values("t")
    idx = np.searchsorted(tj["t"].values, ts_grid, side="right") - 1
    return np.where(idx >= 0, tj["model_prob"].values[np.clip(idx, 0, len(tj) - 1)], np.nan)

g_tau = to_tau(grid_ts)
mkt_w = ff_price(winner, grid_ts)
mdl_w = ff_model(winner, grid_ts)

# ---------- 2. per-tweet reaction ----------
rows = []
for k, ts in enumerate(sorted(post_ts)):
    mdl_before = traj[(traj.label == winner) & (traj.t < ts)]["model_prob"]
    mdl_after = traj[(traj.label == winner) & (traj.t >= ts)]["model_prob"]
    mk_before = wtr[wtr.t < ts]["yes_price"]
    mk_after = wtr[(wtr.t >= ts) & (wtr.t < ts + 1800)]["yes_price"]  # next 30 min
    rows.append({"tweet": k + 1, "tau": round(float(to_tau(ts)), 3),
                 "model_before": (mdl_before.iloc[-1] if len(mdl_before) else np.nan),
                 "model_after": (mdl_after.iloc[0] if len(mdl_after) else np.nan),
                 "market_before": (mk_before.iloc[-1] if len(mk_before) else np.nan),
                 "market_after30m": (mk_after.iloc[-1] if len(mk_after) else np.nan)})
pt = pd.DataFrame(rows)
pt["dmodel"] = pt["model_after"] - pt["model_before"]
pt["dmarket"] = pt["market_after30m"] - pt["market_before"]
pt.to_csv(folder / "per_tweet.csv", index=False)

# ---------- 3. scores through time (winner log-loss + multiclass Brier) ----------
labels = [b["label"] for b in meta["brackets"]]
score_rows = []
for ts, tau in zip(grid_ts, g_tau):
    mdl = np.array([ff_model(l, np.array([ts]))[0] for l in labels])
    mkt = np.array([ff_price(l, np.array([ts]))[0] for l in labels])
    if np.isnan(mdl).any() or np.isnan(mkt).all():
        continue
    mkt = np.nan_to_num(mkt, nan=0.0)
    mdl = mdl / mdl.sum() if mdl.sum() > 0 else mdl
    mktn = mkt / mkt.sum() if mkt.sum() > 0 else mkt
    onehot = np.array([1.0 if l == winner else 0.0 for l in labels])
    wi = labels.index(winner)
    score_rows.append({"tau": round(float(tau), 3),
                       "model_ll": -np.log(max(mdl[wi], 1e-6)), "market_ll": -np.log(max(mktn[wi], 1e-6)),
                       "model_brier": float(((mdl - onehot) ** 2).sum()),
                       "market_brier": float(((mktn - onehot) ** 2).sum())})
sc = pd.DataFrame(score_rows)
sc.to_csv(folder / "scores_over_time.csv", index=False)

# ---------- 4. lead-lag: corr(Δmodel(t), Δmarket(t+lag)) ----------
dm = pd.Series(mdl_w).diff().values
dk = pd.Series(mkt_w).diff().values
mask = ~(np.isnan(dm) | np.isnan(dk))
lags = range(-6, 7)  # ±30 min (5-min steps)
ll = []
for lag in lags:
    if lag >= 0:
        a, b = dm[:len(dm) - lag], dk[lag:]
    else:
        a, b = dm[-lag:], dk[:len(dk) + lag]
    m = ~(np.isnan(a) | np.isnan(b))
    ll.append((lag * 5, float(np.corrcoef(a[m], b[m])[0, 1]) if m.sum() > 3 else np.nan))
lead = pd.DataFrame(ll, columns=["lag_min", "corr"])
best = lead.loc[lead["corr"].abs().idxmax()] if lead["corr"].notna().any() else None

# ---------- figure ----------
fig, ax = plt.subplots(2, 2, figsize=(14, 9))
a0 = ax[0, 0]
a0.plot(wt["tau"], wt["model_prob"], "-", color="#1f77b4", lw=2, label="modèle")
a0.plot(wtr_tau, wtr["yes_price"], ".", color="#E8704C", ms=3, alpha=0.5, label="marché (ticks)")
for x in to_tau(post_ts):
    a0.axvline(x, color="gray", alpha=0.10, lw=0.6)
a0.set_title(f"Gagnante {winner} ({realized} tw) — modèle vs marché (tick)"); a0.set_ylim(0, 1)
a0.set_xlabel("τ"); a0.legend(fontsize=8)

a1 = ax[0, 1]
a1.plot(sc["tau"], sc["model_ll"], color="#1f77b4", label="modèle")
a1.plot(sc["tau"], sc["market_ll"], color="#E8704C", label="marché")
a1.set_title("log-loss sur la gagnante (↓ mieux)"); a1.set_xlabel("τ"); a1.legend(fontsize=8)

a2 = ax[1, 0]
a2.scatter(pt["dmarket"], pt["dmodel"], s=18, alpha=0.6)
a2.axhline(0, color="gray", lw=0.5); a2.axvline(0, color="gray", lw=0.5)
a2.set_xlabel("Δ marché 30min après tweet"); a2.set_ylabel("Δ modèle au tweet")
a2.set_title("réaction par tweet (modèle vs marché)")

a3 = ax[1, 1]
a3.bar(lead["lag_min"], lead["corr"], width=3.5, color="#4C9BE8")
a3.axvline(0, color="gray", lw=0.6)
a3.set_xlabel("lag marché − modèle (min)"); a3.set_ylabel("corr(Δmodèle, Δmarché)")
a3.set_title("lead-lag (pic à droite = modèle devance)")
plt.tight_layout(); plt.savefig(folder / "postmortem_detailed.png", dpi=110)

print(f"=== POST-MORTEM DÉTAILLÉ {SLUG} — gagnante {winner} ({realized} tweets) ===")
print(f"instants trajectoire={len(traj)//len(labels)} | ticks marché(gagnante)={len(wtr)} | tweets={len(post_ts)}")
print(f"\nlog-loss moyen (gagnante): modèle {sc.model_ll.mean():.3f} vs marché {sc.market_ll.mean():.3f}")
print(f"Brier multiclasse moyen:   modèle {sc.model_brier.mean():.3f} vs marché {sc.market_brier.mean():.3f}")
print(f"\nréaction par tweet (moyenne |Δ|): modèle {pt.dmodel.abs().mean():.4f} vs marché {pt.dmarket.abs().mean():.4f}")
print(f"corr(Δmodèle, Δmarché par tweet) = {pt[['dmodel','dmarket']].corr().iloc[0,1]:.3f}")
if best is not None:
    blag, bcorr = float(best["lag_min"]), float(best["corr"])
    sense = "modèle DEVANCE le marché" if blag > 0 else ("marché devance le modèle" if blag < 0 else "simultané")
    print(f"lead-lag: corr max {bcorr:+.3f} à lag {blag:+.0f} min → {sense}")
print("\n--- 8 tweets au plus fort impact marché ---")
pd.set_option("display.width", 160)
print(pt.reindex(pt.dmarket.abs().sort_values(ascending=False).index).head(8)
      [["tweet", "tau", "model_before", "model_after", "market_before", "market_after30m", "dmodel", "dmarket"]].round(3).to_string(index=False))
print(f"\nsaved: per_tweet.csv, scores_over_time.csv, postmortem_detailed.png (dans {folder})")
