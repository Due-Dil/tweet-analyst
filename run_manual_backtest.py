"""Optimal strategy for MANUAL execution (low reactivity): one high-confidence entry, held to close.

Buying/selling is manual → no continuous rebalancing. So the strategy must: wait until the model is
*confident*, ENTER ONCE the joint-Kelly bet, and HOLD to resolution. We backtest this on every resolved
market and sweep the confidence gates to find the operating point with a high WIN-RATE at usable
coverage (you want to be right when you act).

Confidence gate (a market is entered at the FIRST τ where all hold):
  * sharpness   σ/bracket-width ≤ sigma_gate
  * conviction  top model bracket prob ≥ conf_thr
  * value       best joint-Kelly leg edge ≥ edge_thr
Entry pays a manual execution haircut (spread, wider than the automated case). Then settle at close.

    python run_manual_backtest.py
"""
import sys, warnings, datetime as dt, time, dataclasses
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")
import numpy as np, pandas as pd
from tweetanalyst import data as D, pathbacktest as PB, model as M, calibration as CAL, windows as W

ANCHOR = dt.datetime(2026, 6, 28, 12, 0, tzinfo=dt.timezone.utc)
BANKROLL, KELLY, SPREAD, SIGMA_GATE = 1000.0, 0.25, 0.03, 1.4
TAUS = np.round(np.arange(0.25, 0.98, 0.06), 3)   # entry-search grid (late-window is where it enters)


class FastTape:
    """Like pathbacktest.MarketTape but the Hawkes/seasonal fit is done ONCE per market (it's stable
    over a 2-7 day window) and only the cheap forecast is recomputed per τ — ~15x faster."""
    def __init__(self, mkt, win_idx, dur_days, steps):
        self.mkt, self.win_idx, self.dur_days, self.steps = mkt, win_idx, dur_days, steps


def fast_tape(posts, mkt, pc, n_sims=2000):
    win_idx = next(i for i, (lo, hi, lab, _) in enumerate(mkt.brackets) if lab == mkt.winner)
    widths = [hi - lo + 1 for (lo, hi, _, _) in mkt.brackets if np.isfinite(hi)]
    bw = float(np.median(widths)) if widths else 20.0
    span = W.utc_ts(mkt.window_end) - W.utc_ts(mkt.window_start)
    dur = span.total_seconds() / 86400.0
    gamma = CAL.gamma_for_duration(dur)
    fit = M.fit_model(posts, mkt.window_start)        # ONE Hawkes EM (the expensive part)
    brs = [D.Bracket(lab, lo, hi, None) for (lo, hi, lab, _) in mkt.brackets]
    rng = np.random.default_rng(7)
    steps = []
    for tau in TAUS:
        now = (W.utc_ts(mkt.window_start) + span * float(tau)).to_pydatetime()
        yes = PB.HB.market_probs_at(mkt, now, pc)
        if float(np.sum(yes)) < 0.5:
            continue
        fc = M.forecast(dataclasses.replace(fit, now=now), mkt.window_start, mkt.window_end,
                        n_sims=n_sims, rng=rng)
        tbl = M.bracket_probabilities(brs, fc.samples, gamma=gamma)
        steps.append({"tau": float(tau), "p_model": np.array([t["model_prob"] for t in tbl]),
                      "yes": yes, "sigma_ratio": float(fc.samples.std()) / bw})
    return FastTape(mkt, win_idx, dur, steps)


def replay_manual(tape, conf_thr, edge_thr, sigma_gate=SIGMA_GATE, kelly=KELLY,
                  bankroll=BANKROLL, spread=SPREAD):
    """First τ where (sharp + confident + edge): enter joint-Kelly (YES-only), hold to resolution."""
    win = tape.win_idx
    for step in tape.steps:
        p, yes, sr = step["p_model"], step["yes"], step["sigma_ratio"]
        if sr > sigma_gate or float(p.max()) < conf_thr:
            continue
        f, _ = PB.kelly_horserace(p, yes)
        if f.sum() <= 0:
            continue
        edges = [float(p[i] - yes[i]) for i in range(len(p)) if f[i] > 0]
        if not edges or max(edges) < edge_thr:
            continue
        staked = pnl = 0.0
        legs = 0
        for i in range(len(p)):
            if f[i] <= 0:
                continue
            stake = bankroll * kelly * float(f[i])
            if stake < 1.0:
                continue
            pe = min(float(yes[i]) + spread / 2.0, 0.999)
            shares = stake / pe
            pnl += shares * (1.0 if i == win else 0.0) - stake
            staked += stake
            legs += 1
        if staked <= 0:
            continue
        return {"entered": True, "tau": step["tau"], "staked": staked, "pnl": pnl,
                "won": pnl > 0, "top_prob": float(p.max()), "legs": legs, "dur": tape.dur_days}
    return {"entered": False, "dur": tape.dur_days}


posts = D.load_posts("elonmusk")
print(f"posts: {len(posts)}", flush=True)
markets = PB.enumerate_resolved_series(posts, ANCHOR, durations=None)
print(f"marchés résolus: {len(markets)} — construction des 'tapes' (modèle+prix)…", flush=True)

t0 = time.time()
tapes = []
for mi, mkt in enumerate(markets):
    pc = {tok: PB.HB.fetch_prices(tok, mkt.window_start, mkt.window_end) for (_, _, _, tok) in mkt.brackets}
    tapes.append(fast_tape(posts, mkt, pc))
    if (mi + 1) % 25 == 0:
        print(f"  [{mi+1}/{len(markets)}] ({time.time()-t0:.0f}s)", flush=True)

# ---- sweep confidence gates ----
n_by_dur = {"toutes": len(tapes),
            "7j": sum(1 for t in tapes if abs(t.dur_days - 7) < 1),
            "2j": sum(1 for t in tapes if abs(t.dur_days - 2) < 1)}
rows = []
for conf in (0.45, 0.55, 0.65, 0.75, 0.85):
    for edge in (0.03, 0.05, 0.08):
        ent = [r for r in (replay_manual(tp, conf, edge) for tp in tapes) if r["entered"]]
        for dur_label in ("toutes", "7j", "2j"):
            sub = ent if dur_label == "toutes" else [r for r in ent if abs(r["dur"] - int(dur_label[0])) < 1]
            if not sub:
                continue
            s = sum(r["staked"] for r in sub)
            rows.append({"conf_min": conf, "edge_min": edge, "durée": dur_label,
                         "couverture": len(sub) / max(1, n_by_dur[dur_label]),
                         "n_pris": len(sub), "win_rate": float(np.mean([r["won"] for r in sub])),
                         "roi": sum(r["pnl"] for r in sub) / s if s > 0 else 0.0,
                         "tau_moy": float(np.mean([r["tau"] for r in sub]))})

df = pd.DataFrame(rows)
df.to_csv("manual_backtest.csv", index=False)
pd.set_option("display.width", 200, "display.max_columns", 20)
print(f"\n=== STRATÉGIE MANUELLE — entrée unique à haute confiance, tenue à la clôture ({time.time()-t0:.0f}s) ===")
print("spread manuel=3c, ¼-Kelly, σ-gate=1.4. couverture = part des marchés effectivement tradés.")
print("\n--- TOUTES DURÉES ---")
print(df[df["durée"] == "toutes"].drop(columns="durée").round(3).to_string(index=False))
print("\n--- PAR DURÉE (edge_min=0.05) ---")
print(df[(df["edge_min"] == 0.05) & (df["durée"] != "toutes")].round(3).to_string(index=False))
print("\nsaved: manual_backtest.csv")
