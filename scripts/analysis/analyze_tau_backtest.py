"""Analyze backtest_data/tau_backtest_2d.csv (produced by run_tau_backtest.py) for three questions:

  [A] CALIBRATION OVER TIME — at each tau, is the model's leader pick well-calibrated (claimed prob
      vs realized win-rate)? What's the ROI of buying it (real spread, held to close)?
  [B] STRUCTURAL BIAS — does the model over-rate its own leader bracket (rel_rank=0) and under-rate
      brackets farther away, narrowing as tau -> 1?
  [C] MARKET MISPRICING BY RANK/TAU — is the market itself mis-calibrated by rank position at a given
      tau (price vs realized win-rate)? What ROI from buying the model's edge against it?

    python analyze_tau_backtest.py
"""
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
import numpy as np
import pandas as pd

pd.set_option("display.width", 220, "display.max_columns", 20)
SPREAD, KELLY, BANKROLL = 0.03, 0.25, 1000.0

df = pd.read_csv("backtest_data/tau_backtest_2d.csv")
n_mkts = df["slug"].nunique()
print(f"=== {len(df)} lignes, {n_mkts} marchés 2j résolus ===\n")


def kelly_roi(sub: pd.DataFrame) -> dict:
    """Flat-stake AND quarter-Kelly ROI of buying YES at yes_price+spread/2, held to resolution."""
    pe = np.minimum(sub["yes_price"].values + SPREAD / 2.0, 0.999)
    win = sub["is_winner"].values.astype(float)
    # flat $1 stake per trade
    flat_pnl = (win - pe) / pe
    # quarter-Kelly stake sized by model edge (using this row's own model_prob as the Kelly prob)
    p = sub["model_prob"].values
    b = (1.0 - pe) / pe
    f = np.clip((p * b - (1 - p)) / np.maximum(b, 1e-9), 0.0, 1.0) * KELLY
    stake = BANKROLL * f
    shares = np.divide(stake, pe, out=np.zeros_like(stake), where=stake > 0)
    pnl = shares * win - stake
    staked = stake.sum()
    return {
        "n": len(sub), "win_rate": float(win.mean()),
        "roi_flat": float(flat_pnl.mean()),
        "roi_kelly": float(pnl.sum() / staked) if staked > 0 else float("nan"),
        "staked_kelly": float(staked),
    }


# --------------------------------------------------------------------------------------------- #
# [A] Calibration over time: the model's OWN leader pick (model_rank==1) by tau
# --------------------------------------------------------------------------------------------- #
print("--- [A] CALIBRATION DU LEADER MODÈLE PAR TAU (model_rank==1) ---")
print("tau : prob_moy_modele | win_rate_reel | ecart_calib | n | ROI_flat | ROI_quart-Kelly\n")
leader = df[df["model_rank"] == 1]
rows = []
for tau, g in leader.groupby("tau"):
    r = kelly_roi(g)
    rows.append({"tau": tau, "prob_modele_moy": g["model_prob"].mean(),
                 "win_rate": r["win_rate"], "ecart_calib": g["model_prob"].mean() - r["win_rate"],
                 "n": r["n"], "roi_flat": r["roi_flat"], "roi_kelly": r["roi_kelly"]})
calibA = pd.DataFrame(rows).round(3)
print(calibA.to_string(index=False))

# --------------------------------------------------------------------------------------------- #
# [B] Structural bias: model_prob - realized win-rate by (tau bucket, rel_rank bucket)
# --------------------------------------------------------------------------------------------- #
print("\n\n--- [B] BIAIS STRUCTUREL : modèle sur/sous-évalue selon la distance au leader (rel_rank) ---")


def rel_bucket(r):
    if r == 0:
        return "0 (leader)"
    if abs(r) == 1:
        return "+/-1 (adjacent)"
    return "+/-2+ (loin)"


df["rel_bucket"] = df["rel_rank"].apply(rel_bucket)
df["tau_third"] = pd.cut(df["tau"], [0, 0.34, 0.67, 1.0], labels=["début (tau<0.34)", "milieu", "fin (tau>0.67)"])

biasB = (df.groupby(["tau_third", "rel_bucket"], observed=True)
         .apply(lambda g: pd.Series({
             "n": len(g), "model_prob_moy": g["model_prob"].mean(),
             "win_rate_reel": g["is_winner"].mean(),
             "biais (prob-reel)": g["model_prob"].mean() - g["is_winner"].mean(),
         })).round(3))
print(biasB.to_string())
print("\nbiais > 0 = modèle SURévalue cette position ; biais < 0 = modèle SOUSévalue.")

# --------------------------------------------------------------------------------------------- #
# [C] Market mispricing by rank/tau + ROI of buying the model's edge against it
# --------------------------------------------------------------------------------------------- #
print("\n\n--- [C] CALIBRATION DU MARCHÉ PAR RANG (market_rank) ET TAU ---")
mkt_calib = (df.groupby(["tau_third", "market_rank"])
             .apply(lambda g: pd.Series({
                 "n": len(g), "prix_marche_moy": g["yes_price"].mean(),
                 "win_rate_reel": g["is_winner"].mean(),
                 "biais_marche (prix-reel)": g["yes_price"].mean() - g["is_winner"].mean(),
             })).round(3))
mkt_calib = mkt_calib[mkt_calib.index.get_level_values("market_rank") <= 4]
print(mkt_calib.to_string())
print("\nbiais_marche > 0 = marché SURpaie cette position (vend cher un truc qui gagne moins souvent) "
      "-> NON y serait profitable.\nbiais_marche < 0 = marché SOUSpaie -> OUI y serait profitable.")

print("\n\n--- [C2] ROI: exploiter l'edge modèle>marché, par (tau_third, rel_rank), seuils d'edge ---")
for edge_min in (0.03, 0.05, 0.08):
    rows = []
    for (tt, rb), g in df.groupby(["tau_third", "rel_bucket"], observed=True):
        sel = g[g["edge"] >= edge_min]
        if len(sel) < 5:
            continue
        r = kelly_roi(sel)
        rows.append({"tau": tt, "position": rb, "edge_min": edge_min, "n": r["n"],
                     "win_rate": r["win_rate"], "roi_flat": r["roi_flat"], "roi_kelly": r["roi_kelly"]})
    if rows:
        print(f"\nedge_min={edge_min}:")
        print(pd.DataFrame(rows).round(3).to_string(index=False))

print("\n\n--- [C3] Le symétrique : vendre NON là où le modèle dit que le marché sur-paie une tranche éloignée ---")
for edge_min in (0.03, 0.05):
    rows = []
    for (tt, rb), g in df.groupby(["tau_third", "rel_bucket"], observed=True):
        sel = g[g["edge"] <= -edge_min]  # model says this bracket is cheaper to fade (buy NO)
        if len(sel) < 5:
            continue
        pe = np.minimum(1.0 - sel["yes_price"].values + 0.03 / 2.0, 0.999)  # NO price + spread
        win_no = (~sel["is_winner"].values).astype(float)
        roi_flat = float(((win_no - pe) / pe).mean())
        rows.append({"tau": tt, "position": rb, "edge_min": edge_min, "n": len(sel),
                     "win_rate_NO": float(win_no.mean()), "roi_flat_NO": roi_flat})
    if rows:
        print(f"\nedge_min={edge_min} (vente YES / achat NO):")
        print(pd.DataFrame(rows).round(3).to_string(index=False))
