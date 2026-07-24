"""Price-dynamics deep-dive on 2-day markets, using the archived 1-minute YES prices.

Hunts for systematic, tradeable patterns in how bracket prices move — especially at the OPEN:
  [A] Mean price trajectory + mispricing (price − eventual win-rate) by RANK-at-open, across τ.
  [B] Early drift (open → first 15% of window) by rank and by absolute label — do certain brackets
      systematically rise or fade right after open?
  [C] Opening settle: the first ~3h move.
  [D] By absolute label (<40, 40-64, …): structural over/under-pricing at open.

    python analyze_price_dynamics.py
"""
import warnings, sys, json
warnings.filterwarnings("ignore")
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from tweetanalyst import data as D, windows as W

pd.set_option("display.width", 220, "display.max_columns", 30)
TAU = np.round(np.arange(0.0, 1.001, 0.02), 3)          # normalized-time grid
OPEN_TAU, EARLY_TAU = 0.03, 0.15                         # "just after open" and "end of early window"

con = D._conn()
mk = pd.read_sql_query(
    "SELECT slug, window_start, window_end, winner, brackets_json FROM resolved_markets "
    "WHERE duration_days < 4", con)

rows = []   # one row per (market, bracket): price path on TAU grid + metadata
for _, r in mk.iterrows():
    ws, we = pd.Timestamp(r["window_start"]), pd.Timestamp(r["window_end"])
    span = (we - ws).total_seconds()
    brs = json.loads(r["brackets_json"])
    paths = {}
    for (lo, hi, label, token) in brs:
        px = pd.read_sql_query(
            "SELECT t, p FROM clob_prices WHERE token=? AND t>0 ORDER BY t", con, params=[token])
        if len(px) < 5:
            continue
        frac = (px["t"].values - ws.timestamp()) / span
        # interpolate YES price onto the τ grid (clip to observed range)
        pth = np.interp(TAU, frac, px["p"].values, left=px["p"].values[0], right=px["p"].values[-1])
        paths[label] = pth
    if len(paths) < 3:
        continue
    labels = list(paths.keys())
    open_prices = {lab: paths[lab][np.argmin(np.abs(TAU - OPEN_TAU))] for lab in labels}
    order = sorted(labels, key=lambda l: -open_prices[l])   # rank 1 = priciest at open
    rank_of = {lab: i + 1 for i, lab in enumerate(order)}
    for lab in labels:
        rows.append({"slug": r["slug"], "label": lab, "rank_open": rank_of[lab],
                     "is_winner": lab == r["winner"], "path": paths[lab]})
con.close()
df = pd.DataFrame(rows)
P = np.vstack(df["path"].values)   # (n_rows, len(TAU))
print(f"=== {mk.shape[0]} marchés 2j · {len(df)} tranches · grille τ {len(TAU)} pts (1-min source) ===\n")

i_open = int(np.argmin(np.abs(TAU - OPEN_TAU)))
i_early = int(np.argmin(np.abs(TAU - EARLY_TAU)))

# --------------------------------------------------------------------------- #
# [A] By rank-at-open: price at open, eventual win-rate, mispricing, and drift
# --------------------------------------------------------------------------- #
print("--- [A] PAR RANG À L'OUVERTURE (rang 1 = plus cher à l'ouverture) ---")
print(f"{'rang':>5} {'n':>4} {'prix_ouv':>9} {'win_réel':>9} {'misprix_ouv':>12} "
      f"{'drift_0→15%':>12} {'prix_τ50%':>10}")
recs = []
for rk in range(1, 9):
    sub = df[df["rank_open"] == rk]
    if len(sub) < 8:
        continue
    Psub = np.vstack(sub["path"].values)
    win = sub["is_winner"].mean()
    po = Psub[:, i_open].mean()
    drift = (Psub[:, i_early] - Psub[:, i_open]).mean()
    p50 = Psub[:, int(np.argmin(np.abs(TAU - 0.5)))].mean()
    recs.append({"rank": rk, "n": len(sub), "p_open": po, "win": win,
                 "mispx": po - win, "drift": drift, "p50": p50})
    print(f"{rk:>5} {len(sub):>4} {po:>9.3f} {win:>9.3f} {po-win:>+12.3f} {drift:>+12.3f} {p50:>10.3f}")
print("\nmisprix_ouv>0 = tranche SURcotée à l'ouverture (fade/NON) · <0 = SOUScotée (achat OUI).")
print("drift_0→15% = variation moyenne du prix sur les 15 premiers % de la fenêtre (~7h).")

# --------------------------------------------------------------------------- #
# [B] Early drift significance by rank (is the open→early move reliable?)
# --------------------------------------------------------------------------- #
print("\n--- [B] FIABILITÉ DE LA DÉRIVE 0→15% PAR RANG (signe cohérent ?) ---")
print(f"{'rang':>5} {'n':>4} {'drift_moy':>10} {'%hausse':>8} {'écart-type':>10} {'t-stat':>7}")
for rk in range(1, 6):
    sub = df[df["rank_open"] == rk]
    if len(sub) < 8:
        continue
    Psub = np.vstack(sub["path"].values)
    d = Psub[:, i_early] - Psub[:, i_open]
    t = d.mean() / (d.std(ddof=1) / np.sqrt(len(d))) if d.std() > 0 else 0
    print(f"{rk:>5} {len(sub):>4} {d.mean():>+10.3f} {np.mean(d>0):>8.0%} {d.std():>10.3f} {t:>7.1f}")

# --------------------------------------------------------------------------- #
# [C] Opening settle — the first ~3h (τ 0→0.06) move by rank
# --------------------------------------------------------------------------- #
print("\n--- [C] MOUVEMENT DES 3 PREMIÈRES HEURES (τ 0→0.06) PAR RANG ---")
i_3h = int(np.argmin(np.abs(TAU - 0.06)))
i_0 = 1  # first grid point after open
for rk in range(1, 6):
    sub = df[df["rank_open"] == rk]
    if len(sub) < 8:
        continue
    Psub = np.vstack(sub["path"].values)
    d = Psub[:, i_3h] - Psub[:, i_0]
    print(f"  rang {rk}: {Psub[:,i_0].mean():.3f} → {Psub[:,i_3h].mean():.3f}  (Δ {d.mean():+.3f}, {np.mean(d>0):.0%} en hausse)")

# --------------------------------------------------------------------------- #
# [D] By absolute label — structural mispricing at open
# --------------------------------------------------------------------------- #
print("\n--- [D] PAR TRANCHE ABSOLUE (structurel) — à l'ouverture ---")
print(f"{'tranche':>10} {'n':>4} {'prix_ouv':>9} {'win_réel':>9} {'misprix':>9} {'drift_0→15%':>12}")
lab_order = ["<40","40-64","65-89","90-114","115-139","140-164","165-189","190-214","215-239","240+"]
for lab in lab_order:
    sub = df[df["label"] == lab]
    if len(sub) < 8:
        continue
    Psub = np.vstack(sub["path"].values)
    po = Psub[:, i_open].mean(); win = sub["is_winner"].mean()
    drift = (Psub[:, i_early] - Psub[:, i_open]).mean()
    print(f"{lab:>10} {len(sub):>4} {po:>9.3f} {win:>9.3f} {po-win:>+9.3f} {drift:>+12.3f}")

df.to_pickle("backtest_data/price_paths_2d.pkl")
print("\nsaved: backtest_data/price_paths_2d.pkl (trajectoires normalisées, pour deep-dive)")
