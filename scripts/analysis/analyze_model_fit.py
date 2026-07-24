"""Does the model HOLD for one market? Goodness-of-fit of the tweet process vs our model.

    python analyze_model_fit.py [slug]

Fits the model as of the window OPEN (only data before it), then checks the realized in-window tweets
against each model component:
  1. LEVEL   — was the realized count inside the forecast distribution? (percentile)
  2. TIMING  — did tweets arrive following the seasonal day×hour profile?
  3. BURSTS  — is the clustering consistent with the Hawkes kernel? (does adding Hawkes fit better?)
  4. BURST DURATION — do intra-burst spacings match the kernel timescale 1/β?

Core test = **Ogata time-rescaling**: with the model's conditional intensity λ(t), the compensator
increments ξ_i = ∫_{t_{i-1}}^{t_i} λ should be i.i.d. Exp(1) if the model is correct. KS test on ξ.
We run it (a) with the predicted level, (b) level rescaled to the realized count (pure shape/burst),
and (c) seasonal-only (no Hawkes) — if (a/b) fit and (c) doesn't, the Hawkes burst layer is validated.
"""
import sys, json, datetime as dt
import os as _os
_os.chdir(_os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))
sys.path.insert(0, "src")
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tweetanalyst import data as D, model as M, hawkes as H, windows as W

SLUG = sys.argv[1] if len(sys.argv) > 1 else "elon-musk-of-tweets-june-27-june-29"
base = Path("backtest_data/postmortem") / SLUG
folder = max(base.glob("snapshot_*"), key=lambda f: json.loads((f / "meta.json").read_text()).get("tau_at_snapshot", 0))
meta = json.loads((folder / "meta.json").read_text())
ws = pd.Timestamp(meta["window_start"]).to_pydatetime()
we = pd.Timestamp(meta["window_end"]).to_pydatetime()
realized = meta["realized_in_window"]
T = (W.utc_ts(we) - W.utc_ts(ws)).total_seconds() / 3600.0

posts = D.load_posts("elonmusk")
fit = M.fit_model(posts, ws)
alpha, beta = fit.hawkes.alpha, fit.hawkes.beta
level = fit.intensity.mean_level
shape = fit.intensity.shape

# ---- 1. LEVEL: forecast distribution at open vs realized ----
fc = M.forecast(fit, ws, we, n_sims=20000, rng=np.random.default_rng(7))
samp = fc.samples
pct = float((samp < realized).mean())
print(f"=== MODÈLE vs MARCHÉ {SLUG} — fenêtre {T:.0f}h, réalisé={realized} tweets ===")
print(f"params au démarrage : niveau hebdo={level:.0f}, α(branching)={alpha:.2f}, "
      f"β={beta:.2f}/h → durée burst 1/β={1/beta*60:.0f} min")
print(f"\n[1] NIVEAU/COMPTE : prévu médiane={np.median(samp):.0f}, "
      f"intervalle 90%=[{np.percentile(samp,5):.0f},{np.percentile(samp,95):.0f}] | "
      f"réalisé {realized} = **percentile {100*pct:.0f}%** "
      f"({'central, EN LIGNE' if 0.1 < pct < 0.9 else 'en QUEUE — niveau mal anticipé'})")

# ---- events (hours from ws) + seed excitation at open ----
ev = posts[(posts.created_at >= W.utc_ts(ws)) & (posts.created_at < W.utc_ts(we))].created_at
ev_h = np.sort((ev - W.utc_ts(ws)).dt.total_seconds().values / 3600.0)
pre = posts[(posts.created_at >= W.utc_ts(ws) - pd.Timedelta(hours=72)) & (posts.created_at < W.utc_ts(ws))]
ages = (W.utc_ts(ws) - pre.created_at).dt.total_seconds().values / 3600.0
Z0 = H.seed_decay_sum(ages, beta)

# ---- background rate per hour offset (tweets/h): level*(1-alpha)*shape[cell] ----
bg_rate = np.array([level * (1 - alpha) * shape[W.et_cell_of_offset(ws, k)] for k in range(int(np.ceil(T)) + 1)])


def bg_integral(t, scale=1.0):
    k = int(t)
    return scale * (bg_rate[:k].sum() + bg_rate[k] * (t - k))


def residuals(scale=1.0, use_hawkes=True):
    """Ogata compensator increments ξ_i for the in-window events."""
    a = alpha if use_hawkes else 0.0
    xi, prev_L, prev_t = [], 0.0, 0.0
    for ti in ev_h:
        Lbg = bg_integral(ti, scale)
        Lhk = a * (np.sum(1 - np.exp(-beta * (ti - ev_h[ev_h < ti]))) + Z0 * (1 - np.exp(-beta * ti)))
        L = Lbg + Lhk
        xi.append(L - prev_L)
        prev_L = L
    return np.array(xi)


# scale so the compensator total matches realized (pure shape/burst test)
total_L = bg_integral(T) + alpha * (np.sum(1 - np.exp(-beta * (T - ev_h))) + Z0 * (1 - np.exp(-beta * T)))
scale_real = realized / total_L if total_L > 0 else 1.0

xi_pred = residuals(1.0, True)            # predicted level + Hawkes
xi_shape = residuals(scale_real, True)    # level rescaled to realized + Hawkes (pure shape/burst)
xi_seas = residuals(scale_real, False)    # seasonal only (no Hawkes), level rescaled

def ks(xi):
    return float(stats.kstest(xi, "expon").statistic), float(stats.kstest(xi, "expon").pvalue)

ks_pred, ks_shape, ks_seas = ks(xi_pred), ks(xi_shape), ks(xi_seas)
print(f"\n[2+3] TIMING+BURSTS (résidus d'Ogata, doivent ~Exp(1) si le modèle tient) :")
print(f"  modèle complet (niveau prévu)        : moyenne ξ={xi_pred.mean():.2f} (≈1 si niveau OK), "
      f"KS={ks_pred[0]:.3f} p={ks_pred[1]:.2f}")
print(f"  forme seule (niveau recalé au réel)   : KS={ks_shape[0]:.3f} p={ks_shape[1]:.2f} "
      f"({'FORME OK' if ks_shape[1] > 0.05 else 'forme s écarte'})")
print(f"  saisonnier SEUL (sans Hawkes)         : KS={ks_seas[0]:.3f} p={ks_seas[1]:.2f}")
verdict_burst = ("Hawkes AMÉLIORE → bursts réels et bien modélisés" if ks_shape[0] < ks_seas[0] - 0.02
                 else "Hawkes n'aide pas ici (peu/pas de clustering ce coup-ci)")
print(f"  → {verdict_burst}")

# ---- timing: realized ET-hour-of-day vs seasonal expectation ----
et_hours = pd.to_datetime(ev).dt.tz_convert(W.ET).dt.hour.values
exp_by_h = np.zeros(24)
for k in range(int(np.ceil(T))):
    cell = W.et_cell_of_offset(ws, k)
    exp_by_h[cell % 24] += shape[cell]
exp_by_h = exp_by_h / exp_by_h.sum() * len(et_hours)
obs_by_h = np.bincount(et_hours, minlength=24).astype(float)
chi = float(np.sum((obs_by_h - exp_by_h) ** 2 / np.maximum(exp_by_h, 0.5)))
corr_h = float(np.corrcoef(obs_by_h, exp_by_h)[0, 1])
print(f"\n[2] TIMING horaire (heure ET) : corr(réel, profil saisonnier)={corr_h:+.2f}, χ²={chi:.0f} "
      f"({'EN LIGNE' if corr_h > 0.5 else 'écart au profil'})")

# ---- burst duration: intra-burst inter-arrivals vs 1/beta ----
ia = np.diff(ev_h) * 60.0  # minutes
short = ia[ia < (1 / beta * 60) * 3]  # likely intra-burst
print(f"\n[4] DURÉE DES BURSTS : 1/β modèle={1/beta*60:.0f} min | "
      f"inter-arrivées courtes observées (médiane)={np.median(short) if len(short) else float('nan'):.0f} min "
      f"sur {len(short)}/{len(ia)} intervalles "
      f"({'cohérent' if len(short) and abs(np.median(short)-1/beta*60) < 1/beta*60 else 'à regarder'})")

# ---- figure ----
fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
# QQ of xi_shape vs Exp(1)
q = np.sort(xi_shape); thq = stats.expon.ppf((np.arange(1, len(q) + 1) - 0.5) / len(q))
ax[0].plot(thq, q, "o", ms=4); ax[0].plot([0, q.max()], [0, q.max()], "r--")
ax[0].set_title(f"QQ résidus vs Exp(1)\nKS={ks_shape[0]:.3f} p={ks_shape[1]:.2f}")
ax[0].set_xlabel("quantiles Exp(1)"); ax[0].set_ylabel("ξ observés")
ax[1].bar(np.arange(24) - 0.2, obs_by_h, width=0.4, label="réel", color="#1f77b4")
ax[1].bar(np.arange(24) + 0.2, exp_by_h, width=0.4, label="saisonnier", color="#E8B04C")
ax[1].set_title(f"Timing horaire ET (corr {corr_h:+.2f})"); ax[1].set_xlabel("heure ET"); ax[1].legend(fontsize=8)
ax[2].hist(ia, bins=30, color="#4C9BE8"); ax[2].axvline(1 / beta * 60, color="r", ls="--", label=f"1/β={1/beta*60:.0f}min")
ax[2].set_title("inter-arrivées (min)"); ax[2].set_xlabel("min"); ax[2].legend(fontsize=8)
plt.tight_layout(); plt.savefig(folder / "model_fit.png", dpi=110)
print(f"\nsaved: model_fit.png (dans {folder})")
