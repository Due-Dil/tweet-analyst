"""Portfolio strategy across all active Elon-tweet markets.

Turns the model's per-bracket edges into a concrete, sized betting plan and the signals to manage it:

  * **Sizing** = fractional Kelly. For a binary bet bought at price ``q`` with model win-prob ``p``,
    the Kelly fraction of bankroll is ``(p - q) / (1 - q) = edge / (1 - q)``. We scale it by
    ``kelly_fraction`` (¼ by default) to cut variance, and only bet when ``edge > edge_threshold``.
  * **Guard-rails** (all the "parameters"): skip markets still early in their window (backtest showed
    early-week edge is noise); cap exposure per market (a market's YES brackets are mutually
    exclusive); cap total deployment at the bankroll.
  * **Signals** = compare the Kelly target portfolio to the wallet's current positions -> enter / add /
    trim / exit.

Not financial advice — decision support from a statistical model whose edge is itself uncertain.
"""
from __future__ import annotations

import datetime as dt
from collections import defaultdict

import numpy as np

from . import data as D
from . import pipeline as P

_SIDE_FR = {"YES": "OUI", "NO": "NON"}


def polymarket_url(slug: str | None) -> str:
    """Public Polymarket event page for a slug (where the user places orders)."""
    return f"https://polymarket.com/event/{slug}" if slug else ""


def kelly_horserace(p: np.ndarray, q: np.ndarray, eps: float = 1e-9) -> tuple[np.ndarray, float]:
    """Joint full-Kelly allocation over mutually-exclusive bracket outcomes (Smoczynski-Tomkins).

    The brackets of a market compete for ONE outcome, so independent per-bracket Kelly double-counts
    correlated bets and over-deploys capital (the "correlation inflation"). This sizes the JOINT bet:
    given model probs ``p`` (renormalized over the partition) and YES prices ``q``, it returns ``f[i]``
    = fraction of bankroll to bet YES on bracket i, plus the cash reserve ``R``. Only positive-edge
    brackets (with a real price) are retained; the optimum keeps ``R`` in cash and bets
    ``f_i = p_i − R·q_i`` on the retained set, with Σf_i + R = 1. Betting YES across the partition
    spans the whole outcome space, so explicit NO bets are unnecessary (and would be redundant)."""
    p = np.asarray(p, float).copy()
    q = np.clip(np.asarray(q, float), eps, 1.0 - eps)
    s = p.sum()
    if s <= 0:
        return np.zeros(len(p)), 1.0
    p = p / s
    n = len(p)
    S = [i for i in range(n) if p[i] > q[i] and q[i] >= 0.01]   # positive edge AND a real price
    S.sort(key=lambda i: p[i] / q[i], reverse=True)
    R = 1.0
    while S:
        P_ = float(sum(p[i] for i in S))
        Q_ = float(sum(q[i] for i in S))
        R = (1.0 - P_) / (1.0 - Q_) if (1.0 - Q_) > eps else 0.0
        worst = S[-1]                              # lowest p_i/q_i in the retained set
        if p[worst] / q[worst] <= R + eps:
            S.pop()
        else:
            break
    f = np.zeros(n)
    for i in S:
        f[i] = max(0.0, p[i] - R * q[i])
    return f, R


def kelly_portfolio(
    p, yes_price, no_price, edge_threshold: float = 0.0, max_frac_per_bet: float = 1.0,
) -> list[tuple[int, str, float, float]]:
    """Joint Kelly over a market's mutually-exclusive brackets, allowing a YES *or* NO bet per bracket.

    Generalizes ``kelly_horserace`` to use each bracket's **real, independently-quoted** NO price (on
    Polymarket NO is not exactly 1−YES, so NO can be the cheaper way to express a view). It maximizes
    expected **log-growth over the single winning outcome**, so the allocation is **coherent by
    construction** — it never recommends contradictory bets, can stack multiple NO (compatible), and
    will spread YES across adjacent brackets only when that genuinely maximizes growth (the dual of
    NO-ing the rest). At most one side per bracket is considered (the positive-edge one).

    Returns ``[(bracket_index, side, price, full_kelly_fraction), ...]`` (fraction of bankroll, before
    applying the fractional-Kelly multiplier). Falls back to the YES-only horse-race if SciPy is
    unavailable or the optimizer fails."""
    p = np.asarray(p, float)
    n = len(p)
    s = p.sum()
    if s <= 0:
        return []
    p = p / s
    # one positive-edge instrument per bracket (the side the model favors at its real price)
    insts: list[tuple[int, str, float]] = []
    for i in range(n):
        yq, nq = yes_price[i], no_price[i]
        e_yes = (p[i] - yq) if (yq is not None and 0.0 < yq < 1.0) else -1.0
        e_no = ((1.0 - p[i]) - nq) if (nq is not None and 0.0 < nq < 1.0) else -1.0
        if e_yes >= e_no and e_yes > edge_threshold:
            insts.append((i, "OUI", float(yq)))
        elif e_no > edge_threshold:
            insts.append((i, "NON", float(nq)))
    if not insts:
        return []
    m = len(insts)
    # excess-return matrix A[k, j] = (gross return per $ on instrument k if bracket j wins) − 1
    A = np.full((m, n), -1.0)
    for k, (i, side, price) in enumerate(insts):
        if side == "OUI":
            A[k, i] = 1.0 / price - 1.0
        else:  # NO on i pays 1/price unless i wins
            A[k, :] = 1.0 / price - 1.0
            A[k, i] = -1.0
    try:
        from scipy.optimize import minimize  # lazy

        def neg(x):
            w = 1.0 + A.T @ x                     # wealth multiple if each bracket wins
            return -float(np.sum(p * np.log(np.maximum(w, 1e-9))))

        res = minimize(neg, np.full(m, min(0.5 / m, max_frac_per_bet)),
                       method="SLSQP", bounds=[(0.0, max_frac_per_bet)] * m,
                       constraints=[{"type": "ineq", "fun": lambda x: 1.0 - x.sum()}],
                       options={"maxiter": 300, "ftol": 1e-10})
        x = res.x if res.success else None
    except Exception:  # noqa: BLE001
        x = None
    if x is None:
        # fallback: YES-only horse-race (still coherent, just no NO instruments)
        q = np.array([(yes_price[i] if (yes_price[i] is not None) else 1.0) for i in range(n)])
        f, _ = kelly_horserace(p, q)
        return [(i, "OUI", float(q[i]), float(f[i])) for i in range(n) if f[i] > 1e-4]
    return [(insts[k][0], insts[k][1], insts[k][2], float(x[k]))
            for k in range(m) if x[k] > 1e-4]


def propose(
    bankroll: float = 1000.0,
    kelly_fraction: float = 0.25,
    edge_threshold: float = 0.04,
    max_sigma_ratio: float = 1.2,   # forecast must be sharp: σ < this × bracket width
    min_obs: int = 0,               # optional extra floor on observed tweets (off by default — the
                                    # σ-confidence is the real filter; a count floor wrongly blocks
                                    # confident low-volume forecasts)
    max_per_market_frac: float = 0.40,
    sizing: str = "joint",          # "joint" = correlation-aware horse-race Kelly over the bracket
                                    # partition (YES-only, realistic — backtest V1_joint); "naive" =
                                    # independent per-bracket two-sided Kelly (legacy, inflates).
    handle: str = D.DEFAULT_HANDLE,
    now: dt.datetime | None = None,
    n_sims: int = 12000,
) -> dict:
    """Propose a sized, multi-market betting plan. Returns {bets, summary, markets, skipped}.

    Two **intrinsic, non-%** reliability gates (a market must pass both):
      * **sharpness**: σ of the simulated total < ``max_sigma_ratio`` × bracket width (the forecast
        is concentrated relative to the bracket scale);
      * **information floor**: at least ``min_obs`` tweets *observed* in the window — so we never bet
        on the pure prior. In absolute count this self-adapts across durations: a 7-day market hits 20
        tweets in ~16h (when its early pace is already predictive), a 2-day market in ~20-24h (its own
        predictivity threshold). Neither gate is a % of the window.
    """
    now = now or dt.datetime.now(dt.timezone.utc)
    trackings = [t for t in D.get_trackings(handle) if t.is_active and t.market_link and t.end > now]

    cands: list[dict] = []
    markets_meta: list[dict] = []
    skipped: list[dict] = []
    for tw in trackings:
        slug = D.slug_from_url(tw.market_link)
        try:
            run = P.run_forecast(slug, now=now, n_sims=n_sims, refresh=False)
        except Exception as e:  # noqa: BLE001
            skipped.append({"market": tw.title, "reason": f"modèle indisponible ({e})"})
            continue
        total_h = (run.window_end - run.window_start).total_seconds() / 3600.0
        elapsed_h = max(0.0, (now - run.window_start).total_seconds() / 3600.0)
        tau = elapsed_h / total_h if total_h > 0 else 1.0
        widths = [b.high - b.low + 1 for b in run.market.brackets if b.high != float("inf")]
        bracket_width = float(np.median(widths)) if widths else 20.0
        sigma_ratio = float(run.forecast.samples.std()) / bracket_width
        n_obs = run.forecast.n_obs
        dur_days = total_h / 24.0
        markets_meta.append({"slug": slug, "title": run.market.title, "tau": tau,
                             "sigma_ratio": sigma_ratio, "n_obs": n_obs,
                             "window_start": run.window_start, "window_end": run.window_end,
                             "dur_days": dur_days})
        if n_obs < min_obs:
            skipped.append({"market": run.market.title,
                            "reason": f"trop peu de tweets observés ({n_obs} < {min_obs}) — pari sur "
                                      f"le pur prior, pas encore de données (τ={tau:.0%})"})
            continue
        if sigma_ratio > max_sigma_ratio:
            skipped.append({"market": run.market.title,
                            "reason": f"prévision pas assez nette (σ={sigma_ratio:.1f}×tranche > "
                                      f"{max_sigma_ratio:.1f}) — edge non fiable (τ={tau:.0%})"})
            continue
        def _add(label, side, price, p_side, edge, kelly_frac):
            cands.append({
                "slug": slug, "market": run.market.title, "tranche": label, "côté": side,
                "prix": float(price), "proba_modèle": float(p_side), "edge": float(edge),
                "kelly_frac": float(kelly_frac), "tau": tau, "window_end": run.window_end,
                "dur_days": total_h / 24.0,
            })

        if sizing == "joint":
            # Correlation-aware joint allocation over the bracket partition, allowing a YES *or* NO
            # bet per bracket (real prices). Coherent by construction (optimizes the single-winner
            # outcome): stacks NO freely, spreads YES only when growth-optimal, never contradicts.
            p_vec = [t["model_prob"] for t in run.table]
            yp = [t.get("yes_price") for t in run.table]
            nq = [t.get("no_price") for t in run.table]
            for (i, side, price, f) in kelly_portfolio(p_vec, yp, nq, edge_threshold=edge_threshold):
                t = run.table[i]
                p_side = t["model_prob"] if side == "OUI" else 1.0 - t["model_prob"]
                _add(t["label"], side, price, p_side, p_side - price, kelly_fraction * f)
        else:
            # Legacy naive: independent two-sided per-bracket Kelly (over-deploys; kept for comparison).
            for t in run.table:
                for side, price, p_side in (("OUI", t.get("yes_price"), t["model_prob"]),
                                            ("NON", t.get("no_price"), 1.0 - t["model_prob"])):
                    if price is None or price <= 0.0 or price >= 1.0:
                        continue
                    edge = p_side - price
                    if edge <= edge_threshold:
                        continue
                    _add(t["label"], side, price, p_side, edge,
                         max(0.0, kelly_fraction * edge / (1.0 - price)))

    # ---- allocation: raw fractional-Kelly stakes, then per-market cap, then global bankroll cap ----
    for c in cands:
        c["stake"] = bankroll * c["kelly_frac"]
    per_market: dict[str, float] = defaultdict(float)
    for c in cands:
        per_market[c["slug"]] += c["stake"]
    cap = bankroll * max_per_market_frac
    for c in cands:
        tot = per_market[c["slug"]]
        if tot > cap and tot > 0:
            c["stake"] *= cap / tot
    grand = sum(c["stake"] for c in cands)
    if grand > bankroll and grand > 0:
        for c in cands:
            c["stake"] *= bankroll / grand

    bets = []
    for c in cands:
        if c["stake"] < 1.0:  # drop dust allocations
            continue
        stake = c["stake"]
        payoff = stake / c["prix"]               # shares * $1 if win
        gain_max = payoff - stake
        ev = c["proba_modèle"] * payoff - stake  # expected value of the bet
        c.update({
            "stake": stake, "payoff_max": payoff, "gain_max": gain_max,
            "rendement_max": gain_max / stake, "ev": ev,
            "ev_pct": ev / stake,             # expected return per $ staked
            "proba_gain": c["proba_modèle"],  # win probability of the held side (explicit alias)
        })
        bets.append(c)
    bets.sort(key=lambda b: -b["ev"])

    summary = {
        "n_bets": len(bets),
        "n_markets_betted": len({b["slug"] for b in bets}),
        "mise_totale": sum(b["stake"] for b in bets),
        "gain_max_total": sum(b["gain_max"] for b in bets),
        "ev_total": sum(b["ev"] for b in bets),
        "bankroll": bankroll,
    }
    return {"bets": bets, "summary": summary, "markets": markets_meta, "skipped": skipped}


def reconcile(bets: list[dict], current_positions: list[dict],
              add_ratio: float = 1.3, trim_ratio: float = 0.7) -> list[dict]:
    """Compare the Kelly target portfolio to current wallet positions -> action signals.

    ``current_positions`` are rows from ``positions.analyze()``. Matching is by
    (market title, bracket, side). Emits Entrer / Renforcer / Alléger / Sortir / Conserver.
    """
    def key(market, tranche, side):
        return (str(market).strip(), str(tranche).strip(), str(side).strip().upper())

    target = {key(b["market"], b["tranche"], b["côté"]): b for b in bets}
    held = {}
    for r in current_positions:
        side = _SIDE_FR.get(str(r.get("côté", "")).upper(), str(r.get("côté", "")).upper())
        held[key(r.get("marché"), r.get("tranche"), side)] = r

    actions = []
    for k, b in target.items():
        cur = held.get(k)
        cur_stake = float(cur["valeur_actuelle"]) if cur else 0.0
        tgt = b["stake"]
        if cur is None:
            act, reason = "🟢 Entrer", f"edge +{b['edge']:.0%} non détenu"
        elif cur_stake < tgt * trim_ratio:
            act, reason = "🔵 Renforcer", "sous la cible Kelly"
        elif cur_stake > tgt * add_ratio:
            act, reason = "🟠 Alléger", "au-dessus de la cible Kelly"
        else:
            act, reason = "✅ Conserver", "proche de la cible"
        actions.append({"marché": b["market"], "tranche": b["tranche"], "côté": b["côté"],
                        "action": act, "valeur_actuelle": cur_stake, "cible": tgt,
                        "edge": b["edge"], "prix": b.get("prix"), "proba_modèle": b.get("proba_modèle"),
                        "slug": b.get("slug"), "dur_days": b.get("dur_days"),
                        "lien": polymarket_url(b.get("slug")), "raison": reason})
    # Held but no longer a target -> exit (model edge gone)
    for k, r in held.items():
        if k in target:
            continue
        edge = r.get("edge_côté")
        if edge is not None and edge < 0:
            reason = f"edge négatif ({edge:.0%}) — le modèle n'y voit plus de valeur"
        else:
            reason = "plus dans le portefeuille-cible (edge sous le seuil)"
        actions.append({"marché": r.get("marché"), "tranche": r.get("tranche"),
                        "côté": _SIDE_FR.get(str(r.get("côté")).upper(), r.get("côté")),
                        "action": "🔴 Sortir", "valeur_actuelle": float(r.get("valeur_actuelle", 0)),
                        "cible": 0.0, "edge": edge, "prix": r.get("prix_marché"),
                        "proba_modèle": r.get("proba_modèle_côté"), "slug": r.get("slug"),
                        "dur_days": None, "lien": polymarket_url(r.get("slug")), "raison": reason})
    order = {"🔴 Sortir": 0, "🟠 Alléger": 1, "🟢 Entrer": 2, "🔵 Renforcer": 3, "✅ Conserver": 4}
    actions.sort(key=lambda a: (order.get(a["action"], 9), -a["valeur_actuelle"]))
    return actions
