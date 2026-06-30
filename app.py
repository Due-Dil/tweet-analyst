"""Streamlit app: probabilités par tranche pour les marchés Polymarket '# of tweets' d'Elon Musk.

Lancer :  streamlit run app.py
"""
from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from tweetanalyst import backtest as BT  # noqa: E402
from tweetanalyst import calibration as CAL  # noqa: E402
from tweetanalyst import crowd as CR  # noqa: E402
from tweetanalyst import data as D  # noqa: E402
from tweetanalyst import execution as EXE  # noqa: E402
from tweetanalyst import history as HIST  # noqa: E402
from tweetanalyst import model as M  # noqa: E402
from tweetanalyst import pipeline as P  # noqa: E402
from tweetanalyst import positions as POS  # noqa: E402
from tweetanalyst import strategy as STR  # noqa: E402
from tweetanalyst import windows as W  # noqa: E402

try:
    from streamlit_autorefresh import st_autorefresh
    _HAS_AUTOREFRESH = True
except ImportError:  # graceful: live mode just disabled
    _HAS_AUTOREFRESH = False

DAYS = ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"]
FR_MONTHS = ["jan", "fév", "mars", "avr", "mai", "juin",
             "juil", "août", "sept", "oct", "nov", "déc"]


def _fmt_range(start_utc, end_utc) -> str:
    """'26 juin → 3 juil' in ET local dates."""
    s = W.utc_ts(start_utc).tz_convert(W.ET)
    e = W.utc_ts(end_utc).tz_convert(W.ET)
    return f"{s.day} {FR_MONTHS[s.month-1]} → {e.day} {FR_MONTHS[e.month-1]}"


def _fmt_rel(seconds: float) -> str:
    """Compact time-to-close, e.g. '6j 4h', '18h 20min', '45min', or 'clôturé'."""
    if seconds <= 0:
        return "clôturé"
    d, h, m = int(seconds // 86400), int((seconds % 86400) // 3600), int((seconds % 3600) // 60)
    if d > 0:
        return f"{d}j {h}h"
    if h > 0:
        return f"{h}h {m:02d}min"
    return f"{m}min"

st.set_page_config(page_title="Elon Tweet Tracker", layout="wide")


def _refresh_button(key: str, *cache_fns) -> None:
    """Per-page refresh: clears ONLY this page's cached data (live CLOB prices, positions…) and reruns.
    Other pages stay cached, so navigation between pages never recomputes."""
    if st.button("🔄 Rafraîchir cette page", key=key, type="primary",
                 help="Recharge les données live de cette page (prix du carnet d'ordres CLOB, "
                      "positions…). Les autres pages restent en cache."):
        for fn in cache_fns:
            try:
                fn.clear()
            except Exception:  # noqa: BLE001
                pass
        st.rerun()


@st.cache_data(show_spinner="Lecture des positions…", ttl=None)
def cached_positions(address: str, n_sims: int, token: int) -> dict:
    return POS.analyze(address, n_sims=n_sims)


@st.cache_data(show_spinner="Lecture de l'historique…", ttl=None)
def cached_history(address: str, token: int) -> dict:
    return HIST.performance_history(address)


def render_history_page() -> None:
    st.title("📈 Mon historique de performance — marchés Elon")
    _refresh_button("rf_history", cached_history)
    st.caption(
        "Toute ta performance sur les marchés « # tweets Elon » : **réalisé** (gains/pertes verrouillés "
        "sur les parts vendues ou résolues) + **latent** (P&L non réalisé sur tes positions encore "
        "ouvertes). Reconstruit du flux d'activité Polymarket (achats/ventes + redeems à la résolution) "
        "par **adresse de wallet** (lecture seule, aucune clé).")
    wallet = POS.load_wallet()
    if not wallet:
        st.info("Renseigne ton wallet (page « Mes positions ») pour voir ton historique.")
        return
    try:
        res = cached_history(wallet, st.session_state.get("pos_token", 0))
    except Exception as e:  # noqa: BLE001
        st.error(f"Historique indisponible : {e}")
        return
    rows, t = res["rows"], res["totals"]
    if not rows:
        st.info("Aucune activité trouvée sur les marchés Elon pour ce wallet.")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("P&L total", f"${t['total']:+,.0f}",
              delta=(f"{t['roi']:+.1%} sur investi" if t.get("roi") is not None else None),
              delta_color="off")
    m2.metric("Réalisé (clôturé)", f"${t['realized']:+,.0f}")
    m3.metric("Latent (ouvert)", f"${t['latent']:+,.0f}")
    m4.metric("Capital investi (cumulé)", f"${t['invested']:,.0f}")
    m5, m6, m7 = st.columns(3)
    m5.metric("Marchés joués", f"{t['n_markets']} ({t['n_closed']} clôturés, {t['n_open']} ouverts)")
    m6.metric("Win-rate (clôturés)", f"{t['win_rate_closed']:.0%}" if t.get("win_rate_closed") is not None else "—")
    m7.metric("Valeur ouverte actuelle", f"${t['current_value']:,.0f}")

    # cumulative realized P&L over time (closed markets by resolution date)
    closed = sorted([r for r in rows if not r["is_open"] and r["last_ts"]], key=lambda r: r["last_ts"])
    if closed:
        import datetime as _dt
        xs = [_dt.datetime.fromtimestamp(r["last_ts"], _dt.timezone.utc) for r in closed]
        cum = np.cumsum([r["realized"] for r in closed])
        figc = go.Figure()
        figc.add_scatter(x=xs, y=cum, mode="lines+markers", line_color="#1f77b4", name="réalisé cumulé")
        figc.add_hline(y=0, line_color="gray", line_width=0.6)
        figc.update_layout(height=300, margin=dict(l=10, r=10, t=30, b=10),
                           title="P&L réalisé cumulé (marchés clôturés)",
                           xaxis_title="date de résolution", yaxis_title="$ cumulés")
        st.plotly_chart(figc, use_container_width=True)

    df = pd.DataFrame([{
        "Marché": r["label"], "Statut": "🟢 ouvert" if r["is_open"] else "✅ clôturé",
        "Investi": f"${r['invested']:,.0f}", "Réalisé": f"${r['realized']:+,.1f}",
        "Latent": (f"${r['latent']:+,.1f}" if r["is_open"] else "—"),
        "Total": f"${r['total']:+,.1f}", "ROI": (f"{r['roi']:+.0%}" if r.get("roi") is not None else "—"),
        "Trades": r["n_trades"], "Polymarket": STR.polymarket_url(r["slug"]),
    } for r in rows])
    st.dataframe(df, use_container_width=True, hide_index=True,
                 column_config={"Polymarket": st.column_config.LinkColumn("Marché ↗", display_text="Ouvrir ↗")},
                 height=min(640, 40 * (len(df) + 1)))
    st.caption(
        "**Réalisé** = profit/perte déjà encaissé (ventes + redeems − coût des parts correspondantes). "
        "**Latent** = P&L non réalisé sur les positions encore ouvertes (mark-to-market). **Total** = "
        "réalisé + latent. **ROI** = total ÷ capital investi sur ce marché.")


def render_positions_page() -> None:
    st.title("💼 Mes positions — alignement avec le modèle")
    _refresh_button("rf_positions", cached_positions)
    st.caption(
        "Lecture seule de tes positions Polymarket par **adresse de wallet** (publique, aucune clé). "
        "Pour chaque pari Elon ouvert : montant en jeu, P&L, et l'**edge du côté que tu détiens** "
        "selon le modèle le plus récent → es-tu encore aligné, ou faut-il réajuster ?"
    )
    addr = st.text_input("Adresse du wallet Polymarket (0x…)",
                         value=st.session_state.get("wallet", POS.load_wallet()),
                         key="wallet").strip()
    if addr and addr != POS.load_wallet():
        POS.save_wallet(addr)  # remember locally for next time (git-ignored)
    hide_dust = st.checkbox("Masquer les positions négligeables (< $5)", value=True)
    if not addr:
        st.info("Entre ton adresse de wallet (visible sur ton profil Polymarket) pour voir tes positions.")
        return
    try:
        res = cached_positions(addr, 12000, st.session_state.get("pos_token", 0))
    except Exception as e:  # noqa: BLE001
        st.error(f"Impossible de lire les positions: {e}")
        return
    pos, s = res["positions"], res["summary"]
    if not pos:
        st.info("Aucune position Elon ouverte trouvée pour ce wallet.")
        return

    m1, m2, m3, m4, m5, m6 = st.columns(6)
    m1.metric("Positions ouvertes", s["n_positions"])
    m2.metric("Valeur actuelle", f"${s['valeur_actuelle']:,.0f}")
    m3.metric("P&L total", f"${s['pnl_total']:,.0f}")
    _rdt = s.get("rendement_max_pct")
    m4.metric("Gain max potentiel", f"${s['gain_max_total']:,.0f}",
              delta=(f"+{_rdt:.0%} sur la mise" if _rdt is not None else None), delta_color="off",
              help="Somme des gains si chaque pari gagne, et rendement max sur la mise totale. "
                   "Plafond théorique : les tranches OUI d'un même marché s'excluent (une seule peut "
                   "gagner), donc non réalisable en entier.")
    m5.metric("À réajuster", s["n_misaligned"])
    m6.metric("Exposition à revoir", f"${s['exposition_a_revoir']:,.0f}")

    df = pd.DataFrame(pos)
    if hide_dust:
        df = df[df["valeur_actuelle"] >= 5.0]
    df = df.sort_values("valeur_actuelle", ascending=False)
    disp = pd.DataFrame({
        "Marché": df["marché"].str.replace("Elon Musk # tweets ", "", regex=False),
        "Tranche": df["tranche"], "Côté": df["côté"],
        "Mise": df["mise"], "Valeur": df["valeur_actuelle"], "P&L": df["pnl"],
        "Gain max": df["gain_max"], "Rendement max": df["rendement_max"],
        "Proba modèle (côté)": df["proba_modèle_côté"], "Edge": df["edge_côté"],
        "Statut": df["statut"],
    })

    def _hl(row):
        if "Réajuster" in str(row["Statut"]):
            return ["background-color: rgba(220,0,0,0.18)"] * len(row)
        if "Aligné" in str(row["Statut"]):
            return ["background-color: rgba(0,170,0,0.16)"] * len(row)
        return [""] * len(row)

    sty = (disp.style
           .format({"Mise": "${:,.0f}", "Valeur": "${:,.0f}", "P&L": "${:,.0f}",
                    "Gain max": "${:,.0f}", "Rendement max": "{:.0%}",
                    "Proba modèle (côté)": "{:.1%}", "Edge": "{:+.1%}"}, na_rep="—")
           .apply(_hl, axis=1))
    st.dataframe(sty, use_container_width=True, hide_index=True,
                 height=min(720, 40 * (len(disp) + 1)))
    st.caption(
        "**Statut** : compare la proba du modèle pour ton côté au prix de marché de ce côté. "
        "🟢 **Aligné** = le modèle te donne encore un edge (> +3 pts) → garde. "
        "🔴 **Réajuster** = le modèle est désormais en-dessous du prix (< −3 pts) → ta position est "
        "richement valorisée, envisage d'alléger/sortir. ≈ Neutre = pas de signal net."
    )


@st.cache_data(show_spinner="Construction de la stratégie…", ttl=None)
def cached_strategy(bankroll: float, kelly: float, edge_thr: float, max_sigma: float,
                    min_obs: int, max_mkt: float, sizing: str, token: int) -> dict:
    return STR.propose(bankroll=bankroll, kelly_fraction=kelly, edge_threshold=edge_thr,
                       max_sigma_ratio=max_sigma, min_obs=min_obs,
                       max_per_market_frac=max_mkt, sizing=sizing, n_sims=12000)


def render_strategy_page() -> None:
    st.title("🎯 Stratégie multi-marchés")
    _refresh_button("rf_strategy", cached_strategy, cached_positions)
    st.caption(
        "Plan de paris **dimensionné par Kelly** sur tous les marchés Elon actifs. En mode **joint "
        "(recommandé)**, les tranches d'un marché sont traitées comme mutuellement exclusives → "
        "allocation conjointe (course de chevaux) : on mise les tranches sous-cotées et on garde le "
        "reste en cash, au lieu d'arroser des paris NON corrélés. C'est le sizing validé par le "
        "backtest (V1_joint). **Aide à la décision, pas un conseil financier.**"
    )
    cz1, cz2 = st.columns([2, 3])
    sizing_label = cz1.radio(
        "Dimensionnement", ["Kelly joint (réaliste)", "Indépendant (legacy, deux côtés)"],
        index=0, horizontal=False,
        help="Joint = corrige la corrélation entre tranches (déploie moins, ROI/$ réaliste, YES "
             "uniquement). Indépendant = ancien comportement deux côtés, qui sur-déploie et gonfle.")
    sizing = "joint" if sizing_label.startswith("Kelly joint") else "naive"
    cz2.caption("ℹ️ Le mode **joint** propose **OUI et NON** au meilleur prix réel, mais en allouant "
                "tout **conjointement** sur l'issue unique (une seule tranche gagne) : paris cohérents "
                "(plusieurs NON OK, OUI groupés seulement si optimal), corrélation prise en compte. "
                "Le mode **indépendant** dimensionne chaque pari isolément (sur-déploie).")
    c1, c2, c3 = st.columns(3)
    bankroll = c1.number_input("Capital à déployer ($)", 50, 10_000_000, 1000, step=100)
    kelly = c2.slider("Fraction de Kelly", 0.05, 1.0, 0.25, 0.05,
                      help="¼ Kelly = prudent (moins de variance). 1.0 = Kelly plein (agressif).")
    edge_pts = c3.slider("Edge minimum (points)", 0, 15, 4,
                         help="Ne parie que si l'avantage modèle dépasse ce seuil.")
    c4, c5, c6 = st.columns(3)
    max_sigma = c4.slider("Netteté min (σ ÷ tranche)", 0.6, 3.0, 1.2, 0.1,
                          help="Ne trade que si l'incertitude de la prévision (σ) est sous ce multiple "
                               "de la largeur de tranche. Plus bas = plus sélectif. S'adapte à la durée "
                               "et à la taille des tranches.")
    min_obs = c5.slider("Plancher tweets observés (optionnel)", 0, 60, 0,
                        help="Filtre OPTIONNEL (0 = off). Le vrai filtre est la confiance du modèle (σ "
                             "à gauche). Un plancher sur le comptage bloquerait à tort les prévisions "
                             "confiantes de marchés calmes — laisse 0 sauf besoin spécifique.")
    max_mkt = c6.slider("Expo max / marché (%)", 10, 100, 40) / 100

    res = cached_strategy(bankroll, kelly, edge_pts / 100, max_sigma, min_obs, max_mkt, sizing,
                          st.session_state.get("calib_token", 0))
    s, bets = res["summary"], res["bets"]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Paris proposés", f"{s['n_bets']} ({s['n_markets_betted']} marché·s)")
    m2.metric("Mise totale", f"${s['mise_totale']:,.0f}",
              delta=f"{s['mise_totale']/s['bankroll']:.0%} du capital", delta_color="off")
    m3.metric("Gain max potentiel", f"${s['gain_max_total']:,.0f}")
    m4.metric("Valeur attendue (EV)", f"${s['ev_total']:,.0f}",
              help="Somme des espérances de gain (proba modèle × payoff − mise).")

    if not bets:
        st.info("Aucun pari proposé : edges sous le seuil, ou tous les marchés sont trop tôt dans "
                "leur fenêtre. Baisse le seuil d'edge ou le % de fenêtre minimum pour être plus agressif.")
    else:
        st.markdown("### Plan proposé")
        bdf = pd.DataFrame(bets)
        bdisp = pd.DataFrame({
            "Marché": bdf["market"].str.replace("Elon Musk # tweets ", "", regex=False),
            "Tranche": bdf["tranche"], "Côté": bdf["côté"],
            "Prix": bdf["prix"], "Proba gain": bdf["proba_gain"], "Edge": bdf["edge"],
            "Mise": bdf["stake"], "Gain max": bdf["gain_max"],
            "Rendement max": bdf["rendement_max"], "EV": bdf["ev"], "EV %": bdf["ev_pct"],
        })
        st.dataframe(
            bdisp.style.format({"Prix": "{:.2f}", "Proba gain": "{:.0%}", "Edge": "{:+.1%}",
                                "Mise": "${:,.0f}", "Gain max": "${:,.0f}",
                                "Rendement max": "{:.0%}", "EV": "${:,.0f}", "EV %": "{:+.1%}"}),
            use_container_width=True, hide_index=True, height=min(560, 40 * (len(bdisp) + 1)))
        st.caption(
            "**Proba gain** = chance que ton côté gagne (les NON à forte proba sont sûrs mais à faible "
            "rendement). **EV %** = rendement *attendu* par $ misé (>0 seulement s'il y a un edge réel : "
            "à prix juste, un pari à 95% de réussite a une EV nulle). Mise = ¼-Kelly, qui pondère déjà "
            "plus les paris à forte proba à edge égal. Trié par EV."
        )

    if res["skipped"]:
        with st.expander(f"Marchés écartés ({len(res['skipped'])})"):
            for x in res["skipped"]:
                st.write(f"• **{x['market'].replace('Elon Musk # tweets ', '')}** — {x['reason']}")

    # ---- Signals vs current wallet positions ----
    st.markdown("### 🎬 Actions à passer (selon le modèle + la stratégie)")
    wallet = POS.load_wallet()
    if not wallet:
        st.info("Renseigne ton wallet (page « Mes positions ») pour comparer ce plan à tes positions "
                "et obtenir les signaux Entrer / Renforcer / Alléger / Sortir.")
        return
    try:
        posres = cached_positions(wallet, 12000, st.session_state.get("pos_token", 0))
    except Exception as e:  # noqa: BLE001
        st.warning(f"Positions indisponibles: {e}")
        return
    cur = posres["positions"]
    acts = STR.reconcile(bets, cur)
    meta_by_slug = {m["slug"]: m for m in res["markets"]}

    def _clean(name: str) -> str:
        return str(name).replace("Elon Musk # tweets ", "").replace("Elon Musk # of tweets ", "")

    if not acts:
        st.info("Aucun signal : pas de position actuelle ni de cible.")
    else:
        st.caption("🔴 Sortir · 🟠 Alléger · 🟢 Entrer · 🔵 Renforcer · ✅ Conserver. "
                   "**Prix** et **Proba modèle** sont du côté recommandé (OUI/NON). Clique **Ouvrir ↗** "
                   "pour passer l'ordre sur Polymarket. Regroupé par marché (durée indiquée).")
        # group by market (slug), ordered by duration then closing date -> 2-day & 7-day separated
        def _order_key(slug):
            m = meta_by_slug.get(slug, {})
            return (m.get("dur_days") or 99, str(m.get("window_end") or ""))
        slugs = sorted({a.get("slug") for a in acts if a.get("slug")}, key=_order_key)
        no_slug = [a for a in acts if not a.get("slug")]
        for slug in slugs:
            g = [a for a in acts if a.get("slug") == slug]
            m = meta_by_slug.get(slug, {})
            title = _clean(g[0]["marché"] or slug)
            dur = m.get("dur_days") or g[0].get("dur_days")
            dur_lbl = f"⏱ {dur:.0f} j" if dur else ""
            url = STR.polymarket_url(slug)
            st.markdown(f"**{title}** &nbsp; {dur_lbl} &nbsp;·&nbsp; [↗ Ouvrir sur Polymarket]({url})")
            gdf = pd.DataFrame([{
                "Action": a["action"], "Tranche": a["tranche"], "Côté": a["côté"],
                "Prix": (f"{a['prix']:.2f}" if a.get("prix") is not None else "—"),
                "Proba modèle": (f"{a['proba_modèle']:.0%}" if a.get("proba_modèle") is not None else "—"),
                "Edge": (f"{a['edge']:+.1%}" if a.get("edge") is not None else "—"),
                "Actuel": f"${a['valeur_actuelle']:,.0f}", "Cible": f"${a['cible']:,.0f}",
                "Pourquoi": a["raison"], "Polymarket": a.get("lien") or "",
            } for a in g])
            st.dataframe(gdf, use_container_width=True, hide_index=True,
                         column_config={"Polymarket": st.column_config.LinkColumn(
                             "Ordre", display_text="Ouvrir ↗")})
        if no_slug:
            st.caption(f"({len(no_slug)} action·s sans marché identifié, ignorées de l'affichage groupé.)")

    # ---- Performance tracking on current positions ----
    st.markdown("### 📊 Suivi de ma performance")
    psum = posres["summary"]
    if not cur:
        st.info("Aucune position Elon ouverte sur ce wallet pour l'instant.")
    else:
        mise = psum["mise_totale"]
        pnl = psum["pnl_total"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Positions", psum["n_positions"])
        c2.metric("Mise totale (entrée)", f"${mise:,.0f}")
        c3.metric("Valeur actuelle", f"${psum['valeur_actuelle']:,.0f}")
        c4.metric("P&L total", f"${pnl:,.0f}",
                  delta=(f"{pnl/mise:+.1%}" if mise > 0 else None))
        pdf = pd.DataFrame(cur)
        perf = pd.DataFrame({
            "Marché": pdf["marché"].map(_clean), "Tranche": pdf["tranche"], "Côté": pdf["côté"],
            "Prix entrée": pdf["prix_entrée"].map(lambda x: f"{x:.2f}"),
            "Prix marché": pdf["prix_marché"].map(lambda x: f"{x:.2f}"),
            "Mise": pdf["mise"].map(lambda x: f"${x:,.0f}"),
            "Valeur": pdf["valeur_actuelle"].map(lambda x: f"${x:,.0f}"),
            "P&L": pdf["pnl"].map(lambda x: f"${x:,.0f}"),
            "Rendement": (pdf["pnl"] / pdf["mise"].replace(0, np.nan)).map(
                lambda x: f"{x:+.0%}" if pd.notna(x) else "—"),
            "Proba modèle (côté)": pdf["proba_modèle_côté"].map(
                lambda x: f"{x:.0%}" if pd.notna(x) else "—"),
            "Statut": pdf["statut"],
            "Polymarket": pdf["slug"].map(STR.polymarket_url),
        }).sort_values("P&L", ascending=False)
        st.dataframe(perf, use_container_width=True, hide_index=True,
                     column_config={"Polymarket": st.column_config.LinkColumn(
                         "Marché ↗", display_text="Ouvrir ↗")},
                     height=min(560, 40 * (len(perf) + 1)))
        st.caption("**Prix entrée** = ton prix d'achat moyen. **P&L** et **Rendement** = gain/perte "
                   "latent vs ta mise. **Statut** : ✅ Aligné (le modèle te donne encore un edge) · "
                   "⚠️ Réajuster (le modèle est passé sous ton prix) · ≈ Neutre.")

    # ---- Value-at-risk: current positions vs the strategy's target portfolio ----
    st.markdown("### ⚠️ Montants à risque")
    st.caption(
        "Tout est en **capital investi** (prix d'entrée × parts pour tes positions, mise pour la "
        "stratégie). Chaque marché ne fait gagner qu'**une** tranche → P&L discret. On montre les **deux "
        "côtés** : **Perte probable** = perte non dépassée dans 95% des cas (queue basse) ; **Gain "
        "probable** = gain atteint dans les 5% meilleurs cas (queue haute) ; **P&L espéré** = moyenne. "
        "Le **ratio gain/risque** compare les deux queues. Total combiné par simulation des marchés.")
    brackets_by_slug = {m["slug"]: m.get("brackets", []) for m in res["markets"]}

    def _render_risk(title, risk):
        t = risk["total"]
        st.markdown(f"**{title}**")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Capital investi", f"${t['value']:,.0f}")
        k2.metric("Perte probable (95%)", f"−${t['var95_loss']:,.0f}")
        k3.metric("Gain probable (95%)", f"+${t['var95_gain']:,.0f}")
        _e = t["expected_pnl"]
        k4.metric("P&L espéré", f"${_e:+,.0f}" if _e == _e else "—")  # noqa: PLR0124 (NaN check)
        _ratio = (t["var95_gain"] / t["var95_loss"]) if t["var95_loss"] > 1e-9 else float("inf")
        st.caption(
            f"Risque/récompense : tu risques **−${t['var95_loss']:,.0f}** pour viser **+${t['var95_gain']:,.0f}** "
            f"(ratio **{_ratio:.1f}×**), espérance **${_e:+,.0f}**. "
            f"Extrêmes : pire −${t['max_loss']:,.0f} / meilleur +${t['max_gain']:,.0f}.")
        rows = []
        for m in risk["per_market"]:
            _pe = m["expected_pnl"]
            rows.append({
                "Marché": _clean((meta_by_slug.get(m["slug"], {}) or {}).get("title", m["slug"])),
                "Capital investi": f"${m['value']:,.0f}",
                "Perte (95%)": f"−${m['var95_loss']:,.0f}", "Gain (95%)": f"+${m['var95_gain']:,.0f}",
                "P&L espéré": (f"${_pe:+,.0f}" if _pe == _pe else "—")})
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Reason in CAPITAL INVESTED (cost basis = entry price × shares = "mise"), not current value:
    # if a position loses, the shares go to $0 → you lose what you put in, not today's mark.
    cur_holdings = [{"slug": r.get("slug"), "tranche": r.get("tranche"), "side": r.get("côté"),
                     "value": float(r.get("mise", 0) or 0),
                     "payoff": float(r.get("parts", 0) or 0)} for r in cur]
    strat_holdings = [{"slug": b["slug"], "tranche": b["tranche"], "side": b["côté"],
                       "value": float(b["stake"]),
                       "payoff": (float(b["stake"]) / float(b["prix"]) if b["prix"] else 0.0)}
                      for b in bets]
    rc1, rc2 = st.columns(2)
    with rc1:
        _render_risk("📍 Mes positions actuelles", STR.portfolio_risk(cur_holdings, brackets_by_slug))
    with rc2:
        _render_risk("🎯 Si j'applique la stratégie", STR.portfolio_risk(strat_holdings, brackets_by_slug))

    # ---- Auto-sell preview (Phase 2 — dry-run; real execution stays disabled) ----
    st.markdown("### 🤖 Ordres de vente automatiques (aperçu)")
    live_on = EXE.EXECUTION_ENABLED
    orders = EXE.build_sell_orders(acts, cur)
    if not orders:
        st.info("Aucun ordre de vente : aucune position à **Sortir** ou **Alléger**.")
    else:
        ex = EXE.get_executor(live=False)  # always preview here; live runs only via run_autosell(confirm=True)
        rows = []
        for o in orders:
            ex.submit(o)  # dry-run validation/log
            title = _clean((meta_by_slug.get(o.market_slug, {}) or {}).get("title", o.market_slug))
            rows.append({"Marché": title, "Tranche": o.bracket_label, "Vendre (parts)": f"{o.shares:g}",
                         "Prix limite": f"{o.limit_price:.3f}", "Raison": o.reason,
                         "Polymarket": STR.polymarket_url(o.market_slug)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True,
                     column_config={"Polymarket": st.column_config.LinkColumn(
                         "Ordre", display_text="Ouvrir ↗")})
        st.caption(
            f"🔒 **Dry-run** — aperçu uniquement, aucun ordre envoyé. Exécution réelle "
            f"{'**ACTIVÉE**' if live_on else 'désactivée'} (Phase 2 : ventes uniquement, clé dans le "
            f"trousseau, confirmation par ordre). Voir `PHASE2_ACTIVATION.md`.")


# --------------------------------------------------------------------------- #
# Cached heavy computations (ttl=None → persist across reruns; each page's "🔄 Rafraîchir" button
# clears ONLY its own cache, so navigating between pages never recomputes).
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, ttl=None)
def market_duration(slug: str, handle: str) -> float:
    m = D.get_market(slug)
    ws, we = D.resolve_window(slug, m, handle)
    return (we - ws).total_seconds() / 86400.0


@st.cache_data(show_spinner=False, ttl=None)
def list_active_markets(handle: str) -> list[dict]:
    """Open markets, sorted by close date (soonest first), with readable labels."""
    now = dt.datetime.now(dt.timezone.utc)
    out = []
    for tw in D.get_trackings(handle):
        if tw.is_active and tw.market_link and tw.end > now:
            dur = (tw.end - tw.start).total_seconds() / 86400.0
            out.append({
                "slug": D.slug_from_url(tw.market_link),
                "range": _fmt_range(tw.start, tw.end),
                "duration": dur,
                "end": tw.end,
            })
    return sorted(out, key=lambda m: m["end"])  # closest close first


@st.cache_data(show_spinner=True, ttl=None)
def cached_run(slug: str, handle: str, now_iso: str | None, n_sims: int,
               half_life: float, fit_days: float, gamma: float | None,
               refresh_token: int = 0, calib_token: int = 0) -> dict:
    # refresh_token busts the cache on each live tick; calib_token busts it after a recalibration.
    now = dt.datetime.fromisoformat(now_iso) if now_iso else None
    run = P.run_forecast(slug, handle=handle, now=now, n_sims=n_sims, gamma=gamma,
                         half_life_days=half_life, fit_days=fit_days)
    conf = M.confidence_report(run.table, run.forecast.samples,
                               run.forecast.summary()["hours_remaining"])
    daily = M.daily_forecast(run.fit, run.window_start, run.window_end)
    duration_days = (run.window_end - run.window_start).total_seconds() / 86400.0
    # repackage to a cacheable dict (avoid caching heavy objects with live handles)
    return {
        "confidence": conf,
        "daily": daily,
        "gamma_applied": run.gamma,
        "duration_days": duration_days,
        "title": run.market.title,
        "window_start": run.window_start,
        "window_end": run.window_end,
        "now": run.now,
        "table": run.table,
        "samples": run.forecast.samples,
        "summary": run.forecast.summary(),
        "heatmap": run.fit.intensity.heatmap(),
        "alpha": run.fit.hawkes.alpha,
        "beta": run.fit.hawkes.beta,
        "burst_h": run.fit.hawkes.burst_timescale_h,
        "mean_level": run.fit.intensity.mean_level,
        "weekly_totals": run.fit.intensity.weekly_totals,
    }


@st.cache_data(show_spinner=False, ttl=None)
def cached_fade_signals(slug: str, handle: str, refresh_token: int = 0) -> list:
    # Live overreaction alerts: brackets that just spiked up into the mid-price fade zone.
    try:
        return CR.detect_fade_signals(slug, handle=handle)
    except Exception:  # noqa: BLE001
        return []


page = st.sidebar.radio("📄 Page", ["📊 Analyse marché", "💼 Mes positions", "🎯 Stratégie",
                                    "📈 Historique"], index=0)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Chaque page a son bouton **🔄 Rafraîchir cette page** (en haut). Les données restent en cache "
    "entre les visites → navigation instantanée, **aucun recalcul** en changeant de page. Rafraîchis "
    "une page pour des prix du carnet d'ordres **live**.")

if page == "💼 Mes positions":
    render_positions_page()
    st.stop()
if page == "🎯 Stratégie":
    render_strategy_page()
    st.stop()
if page == "📈 Historique":
    render_history_page()
    st.stop()

st.title("📊 Elon Musk — probabilités par tranche (Polymarket)")
_refresh_button("rf_market", cached_run, list_active_markets, cached_fade_signals)
st.caption(
    "Modèle: intensité saisonnière jour×heure (ET) + processus auto-excitant de Hawkes "
    "(bursts) + Monte-Carlo de la fin de semaine. Données: xtracker.polymarket.com (source "
    "de résolution) + prix du **carnet d'ordres CLOB live** (midpoint, source de vérité)."
)


# --------------------------------------------------------------------------- #
# Sidebar controls
# --------------------------------------------------------------------------- #
st.sidebar.header("Paramètres")
handle = st.sidebar.text_input("Compte (handle)", value="elonmusk")

try:
    markets = list_active_markets(handle)
except Exception as e:  # noqa: BLE001
    markets = []
    st.sidebar.warning(f"Marchés actifs indisponibles: {e}")

mode = st.sidebar.radio("Marché", ["Marchés actifs", "URL / slug manuel"], index=0)
if mode == "Marchés actifs" and markets:
    _now = dt.datetime.now(dt.timezone.utc)
    # Select by SLUG (stable identity) with a persistent key — NOT by positional index, so refreshing
    # / reordering the list (sorted by close date) never remaps the selection to a different market.
    label_by_slug = {
        m["slug"]: f"⏳ {_fmt_rel((m['end'] - _now).total_seconds())}  ·  {m['range']}  ·  {m['duration']:.0f}j"
        for m in markets
    }
    slug_options = [m["slug"] for m in markets]
    slug = st.sidebar.selectbox(
        "Choisir un marché (clôture la plus proche en premier)", slug_options,
        format_func=lambda s: label_by_slug.get(s, s), key="sel_market_slug")
else:
    url = st.sidebar.text_input(
        "URL Polymarket ou slug",
        value="elon-musk-of-tweets-june-26-july-3",
    )
    slug = D.slug_from_url(url)

n_sims = st.sidebar.select_slider(
    "Simulations Monte-Carlo", [4000, 8000, 20000, 40000], value=8000,
    help="8 000 = rapide pour naviguer entre marchés (probas quasi identiques). Monte à 20-40k pour "
         "une précision maximale sur un marché donné.")
half_life = st.sidebar.slider(
    "Demi-vie récence (jours)", 7, 60, 28,
    help="Vitesse d'oubli des vieux tweets pour le profil jour×heure. Plus court = colle au "
         "comportement très récent (réactif mais bruité) ; plus long = profil lisse et stable mais "
         "lent à s'adapter. Défaut 28 j (générique, non optimisé par backtest).")
fit_days = st.sidebar.slider(
    "Fenêtre fit Hawkes (jours)", 30, 180, 90,
    help="Historique utilisé pour estimer les paramètres de burst (α taille, β durée). Plus court = "
         "réactif au comportement récent mais bruité ; plus long = noyau de burst plus stable. "
         "Défaut 90 j (générique, non optimisé par backtest).")
st.sidebar.caption("ℹ️ Laisse les valeurs par défaut sauf pour expérimenter — le reste du modèle "
                   "(γ, niveau, tranches) est déjà calibré et adapté au marché.")
auto_gamma = st.sidebar.checkbox(
    "γ auto (calibré par durée de marché)", value=True,
    help="Choisit le sharpening γ calibré pour la durée du marché (2/3/7 j). γ corrige la "
         "sous-confiance ; il dépend surtout du régime récent, peu de la durée.",
)
if auto_gamma:
    gamma = None
else:
    gamma = st.sidebar.slider("Recalibrage manuel (sharpening γ)", 1.0, 3.0,
                              float(P.CALIBRATED_GAMMA), 0.05)
if "calib_token" not in st.session_state:
    st.session_state.calib_token = 0

override = st.sidebar.checkbox("Forcer une date 'as of' (backtest manuel)")
now_iso = None
if override:
    d = st.sidebar.date_input("Date", value=dt.date(2026, 6, 23))
    t = st.sidebar.time_input("Heure (UTC)", value=dt.time(16, 0))
    now_iso = dt.datetime.combine(d, t).replace(tzinfo=dt.timezone.utc).isoformat()

# ---- Live mode: auto re-pull (incremental) + recompute on an interval ----
st.sidebar.markdown("---")
live = st.sidebar.checkbox(
    "🔴 Mode live (auto-refresh)", value=False, disabled=(override or not _HAS_AUTOREFRESH),
    help="Re-tire les nouveaux tweets (incrémental, ~5 min de latence XTracker) et recalcule "
         "automatiquement. À activer surtout en fin de semaine près de la clôture.",
)
if override:
    st.sidebar.caption("Mode live indisponible avec une date forcée.")
elif not _HAS_AUTOREFRESH:
    st.sidebar.caption("Installer `streamlit-autorefresh` pour activer le mode live.")

refresh_token = 0
if live:
    interval = st.sidebar.select_slider("Intervalle live", [30, 60, 120, 300], value=60,
                                        format_func=lambda x: f"{x}s")
    tick = st_autorefresh(interval=interval * 1000, key="live_tick")
    refresh_token = int(tick)  # changes each tick -> busts cached_run -> fresh pull + recompute

# --------------------------------------------------------------------------- #
# Run
# --------------------------------------------------------------------------- #
try:
    R = cached_run(slug, handle, now_iso, n_sims, half_life, fit_days, gamma, refresh_token,
                   st.session_state.calib_token)
except Exception as e:  # noqa: BLE001
    st.error(f"Erreur: {e}")
    st.stop()

if live:
    st.caption(f"🔴 Live — dernière mise à jour {dt.datetime.now().strftime('%H:%M:%S')} "
               f"(re-tirage incrémental toutes les {interval}s)")

s = R["summary"]
remaining_h = s["hours_remaining"]
settled = remaining_h <= 0

st.subheader(R["title"])
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Tweets observés", s["n_obs"])
c2.metric("Heures restantes", f"{remaining_h:.0f}" if not settled else "réglé")
c3.metric("Total final (médiane)", f"{s['median']:.0f}")
c4.metric("Intervalle 90%", f"{s['p5']:.0f}–{s['p95']:.0f}")
c5.metric("Niveau hebdo moyen", f"{R['mean_level']:.0f}")

# ---- Applied γ + per-duration recalibration ----
dur = R["duration_days"]
_cinfo = CAL.load_calibration().get(int(round(dur)))
_age = CAL._age_days(_cinfo["calibrated_at"]) if _cinfo else None

# Tiered recommendation banner — you recalibrate manually via the button when prompted.
if _cinfo is None:
    st.warning(f"🟡 **Recalibrage recommandé** — aucune calibration enregistrée pour ~{dur:.0f} j "
               "(valeur par défaut utilisée). Clique sur **🎯 Recalibrer γ** ci-dessous.")
elif _age is not None and _age > 21:
    st.error(f"🔴 **Recalibrage nécessaire** — dernière calibration il y a {_age:.0f} jours. "
             "Le régime a probablement dérivé. Clique sur **🎯 Recalibrer γ**.")
elif _age is not None and _age > 7:
    st.warning(f"🟡 **Recalibrage recommandé** — dernière calibration il y a {_age:.0f} jours. "
               "Clique sur **🎯 Recalibrer γ** pour la rafraîchir.")

_cwhen = f" — calibré il y a {_age:.0f} j" if _cinfo else " — valeur par défaut"
gcol1, gcol2 = st.columns([3, 1])
gcol1.caption(
    f"γ appliqué = **{R['gamma_applied']:.2f}** "
    f"({'auto, calibré pour ~%.0f j' % dur if auto_gamma else 'manuel'}{_cwhen}). "
    f"Durée du marché ≈ **{dur:.1f} j**. "
    + ("⚠️ Marché court : variance plus élevée, prudence." if dur < 6 else "")
)
if gcol2.button("🎯 Recalibrer γ (régime actuel)", help="Rejoue des fenêtres synthétiques de "
                "cette durée sur le régime récent et réajuste γ, puis le persiste. ~1-3 min."):
    with st.spinner(f"Calibration de γ pour ~{dur:.0f} j…"):
        posts_c = D.load_posts(handle, start=R["now"] - dt.timedelta(days=130), end=R["now"])
        mkt_c = D.get_market(slug)  # calibrate against THIS market's actual brackets (width varies)
        brs_c = [(b.low, b.high, b.label) for b in mkt_c.brackets]
        res_c = CAL.calibrate_gamma(posts_c, R["now"], dur, n_sims=3000, brackets=brs_c)
        CAL.store_calibration(dur, res_c)  # persist so the banner clears and γ is reused
    st.session_state.calib_token += 1
    st.success(f"γ recalibré pour ~{dur:.0f} j = {res_c['gamma']:.2f} "
               f"(sur {res_c['n_windows']} fenêtres, log-loss {res_c['ll_before']:.2f}→{res_c['ll_after']:.2f}). "
               "Relancé.")
    st.rerun()

# --------------------------------------------------------------------------- #
# Bracket table with edge
# --------------------------------------------------------------------------- #
st.markdown("### Probabilités par tranche")
# ---- Confidence panel (built on the distance-to-edge insight) ----
conf = R["confidence"]
saf = conf["safety"]
if saf >= 1.5:
    badge, color = "🟢 CONFIANT", "rgba(0,160,0,0.18)"
elif conf["young_week"]:
    badge, color = "⚪️ TROP TÔT", "rgba(150,150,150,0.18)"
elif saf <= 0.7:
    badge, color = "🟠 BORD DE TRANCHE", "rgba(230,150,0,0.20)"
else:
    badge, color = "🟡 INTERMÉDIAIRE", "rgba(220,200,0,0.15)"

st.markdown(
    f"<div style='background:{color};padding:10px 14px;border-radius:8px'>"
    f"<b>Indice de confiance — {badge}</b> &nbsp;|&nbsp; "
    f"Tranche la plus probable : <b>{conf['top_label']}</b> à <b>{conf['top_prob']:.0%}</b> "
    f"&nbsp;|&nbsp; Total projeté ≈ <b>{conf['proj_total']:.0f}</b>, "
    f"marge au bord de tranche : <b>{conf['margin']:.0f} tweets</b> "
    f"(soit <b>{conf['safety']:.1f}×</b> l'incertitude restante)<br>"
    f"<span style='font-size:0.9em'>→ {conf['regime']}</span></div>",
    unsafe_allow_html=True,
)
st.caption(
    "La confiance = proba sur la tranche leader. La *marge au bord* est le levier clé : un total "
    "loin d'un bord de 20 = modèle sûr (peu d'edge) ; collé à un bord = un burst peut tout faire "
    "basculer (incertitude = opportunité)."
)

# ---- ⚡ Live overreaction fade alert (validated crowd pattern) ----
if not override and not settled:
    _fsig = cached_fade_signals(slug, handle, refresh_token)
    if _fsig:
        _lines = " · ".join(
            f"**{s['tranche']}** (YES {s['prix_yes']:.2f}, pic +{s['saut']:.0%}) → **NON @ {s['prix_no']:.2f}**"
            for s in _fsig)
        st.warning(
            f"⚡ **Surréaction détectée — fade candidate(s)** : {_lines}\n\n"
            "La foule a poussé ces tranches en zone mi-prix (0.25–0.75) sur un pic ; historiquement le "
            "prix **revient** (backtest : +16–19 pts vs baseline, robuste en walk-forward). Piste : "
            "**acheter NON** et sortir sous ~6 h. *Pas au-dessus de 0.75 (là le pic est mérité).*")

df = pd.DataFrame(R["table"])
df_disp = pd.DataFrame(
    {
        "Tranche": df["label"],
        "Proba modèle": df["model_prob"],
        "Prix OUI": df["yes_price"],
        "Prix NON": df["no_price"],
        "Action": df["best_side"],
        "Edge du pari": df["best_edge"],
    }
)


def _hl_action(row):
    side = row["Action"]
    e = row["Edge du pari"]
    if side == "OUI" and pd.notna(e) and e > 0.03:
        return ["background-color: rgba(0,180,0,0.22)"] * len(row)
    if side == "NON" and pd.notna(e) and e > 0.03:
        return ["background-color: rgba(80,140,230,0.22)"] * len(row)
    return [""] * len(row)


styled = (
    df_disp.style.format(
        {"Proba modèle": "{:.1%}", "Prix OUI": "{:.2f}", "Prix NON": "{:.2f}",
         "Edge du pari": "{:+.1%}"},
        na_rep="—",
    )
    .apply(_hl_action, axis=1)
)
st.dataframe(styled, use_container_width=True, hide_index=True, height=min(680, 38 * (len(df) + 1)))
st.caption(
    "**Action** = côté recommandé. 🟢 Vert = acheter **OUI** (tranche sous-cotée). "
    "🔵 Bleu = acheter **NON** (tranche surcotée). *Edge du pari* = avantage estimé du côté "
    "recommandé, après recalibrage γ. Seuls les paris à edge > 3 pts sont surlignés."
)

# --------------------------------------------------------------------------- #
# Charts
# --------------------------------------------------------------------------- #
col_l, col_r = st.columns(2)

with col_l:
    st.markdown("#### Distribution simulée du total final")
    samples = R["samples"]
    fig = go.Figure()
    fig.add_histogram(x=samples, nbinsx=60, marker_color="#4C9BE8", name="simulé")
    fig.add_vline(x=s["n_obs"], line_dash="dot", line_color="gray",
                  annotation_text=f"observés ({s['n_obs']})")
    fig.add_vline(x=s["median"], line_color="#E8704C", annotation_text="médiane")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="total tweets sur la semaine", yaxis_title="fréquence")
    st.plotly_chart(fig, use_container_width=True)

with col_r:
    st.markdown("#### Modèle vs marché par tranche")
    fig2 = go.Figure()
    fig2.add_bar(x=df["label"], y=df["model_prob"], name="Modèle", marker_color="#4C9BE8")
    if df["market_price"].notna().any():
        fig2.add_bar(x=df["label"], y=df["market_price"], name="Marché", marker_color="#E8B04C")
    fig2.update_layout(height=360, barmode="group", margin=dict(l=10, r=10, t=10, b=10),
                       xaxis_title="tranche", yaxis_title="probabilité",
                       legend=dict(orientation="h", y=1.1))
    st.plotly_chart(fig2, use_container_width=True)

# ---- Per-day forecast: actuals + estimated, over the selected market window ----
st.markdown("#### Prévision du nombre de tweets par jour (réels + estimés)")
dd = R["daily"]
labels = [
    f"{DAYS[d['date'].weekday()]} {d['date'].day}" + (" ½j" if d.get("half_day") else "")
    for d in dd
]
actual = [d["actual"] for d in dd]
est_rem = [max(d["est_median"] - d["actual"], 0) if d["status"] != "passé" else 0 for d in dd]
fut_mask = [d["status"] != "passé" for d in dd]
fut_x = [labels[i] for i, m in enumerate(fut_mask) if m]
fut_med = [dd[i]["est_median"] for i, m in enumerate(fut_mask) if m]
err_up = [dd[i]["est_p90"] - dd[i]["est_median"] for i, m in enumerate(fut_mask) if m]
err_dn = [dd[i]["est_median"] - dd[i]["est_p10"] for i, m in enumerate(fut_mask) if m]

figd = go.Figure()
figd.add_bar(x=labels, y=actual, name="Réels (déjà postés)", marker_color="#1f77b4")
figd.add_bar(x=labels, y=est_rem, name="Estimés (à venir)", marker_color="#E8B04C")
if fut_x:
    figd.add_scatter(
        x=fut_x, y=fut_med, mode="markers", marker=dict(opacity=0),
        error_y=dict(type="data", symmetric=False, array=err_up, arrayminus=err_dn,
                     color="rgba(80,80,80,0.7)", thickness=1.5, width=5),
        name="plage 10–90 %", showlegend=True,
    )
figd.update_layout(height=340, barmode="stack", margin=dict(l=10, r=10, t=10, b=10),
                   xaxis_title="jour (ET)", yaxis_title="tweets / jour",
                   legend=dict(orientation="h", y=1.12))
st.plotly_chart(figd, use_container_width=True)
st.caption(
    "**Bleu** = tweets déjà postés (exacts). **Orange** = estimation médiane du reste de la journée / "
    "des jours à venir. **Barre noire = plage 10–90 %** : dans 80 % des scénarios simulés, le total du "
    "jour tombe dans cet intervalle (capte la variabilité et le risque de burst). "
    "**½j** = demi-journée : le marché ouvre et clôture le vendredi **midi** ET, donc les vendredis aux "
    "extrémités ne couvrent qu'environ 12 h → total attendu ~moitié d'un jour plein (ce n'est pas une anomalie)."
)

st.markdown("#### Rythme de tweets — intensité par jour × heure (heure ET, pondérée récence)")
hm = R["heatmap"]  # (7, 24) tweets/hour
figh = go.Figure(
    go.Heatmap(z=hm, x=[f"{h:02d}h" for h in range(24)], y=DAYS,
               colorscale="YlOrRd", colorbar=dict(title="tw/h"))
)
figh.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10))
st.plotly_chart(figh, use_container_width=True)
st.caption(
    f"Hawkes: α={R['alpha']:.2f} (part de tweets déclenchés par 'burst'), "
    f"échelle de burst ≈ {R['burst_h']*60:.0f} min. "
    f"Totaux hebdo récents: {[int(x) for x in R['weekly_totals'][-8:]]}"
)

# --------------------------------------------------------------------------- #
# Backtest (optional, heavier)
# --------------------------------------------------------------------------- #
st.markdown("---")
with st.expander("🔬 Backtest de calibration (replay des semaines passées)"):
    n_weeks = st.slider("Semaines à rejouer", 4, 24, 10)
    bt_sims = st.select_slider("Sims/backtest", [2000, 4000, 8000], value=4000)
    if st.button("Lancer le backtest"):
        with st.spinner("Replay en cours…"):
            posts = D.load_posts(handle, start=R["now"] - dt.timedelta(days=260), end=R["now"])
            res = BT.run_backtest(posts, R["window_end"], n_weeks=n_weeks, n_sims=bt_sims)
        g_opt, ll0, ll1 = BT.fit_sharpening(res.prob_matrix, res.true_idx)
        m1, m2, m3 = st.columns(3)
        m1.metric("Log-loss (↓ mieux)", f"{ll1:.3f}", delta=f"{ll1 - ll0:+.3f} vs brut")
        m2.metric("Brier multiclasse (↓ mieux)", f"{res.brier:.3f}")
        m3.metric("γ optimal (sharpening)", f"{g_opt:.2f}")
        st.caption(
            f"γ ajusté sur ces {n_weeks} semaines = **{g_opt:.2f}** (in-sample, donc légèrement "
            f"optimiste). Reporte cette valeur dans le curseur 'Recalibrage' à gauche si tu veux "
            f"l'appliquer aux prévisions."
        )
        rel = res.reliability
        figr = go.Figure()
        figr.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line_dash="dash",
                         line_color="gray", name="parfait")
        figr.add_scatter(x=rel["mean_pred"], y=rel["hit_rate"], mode="markers+lines",
                         marker_size=8, name="modèle")
        figr.update_layout(height=380, xaxis_title="probabilité prédite",
                           yaxis_title="fréquence réalisée", title="Courbe de fiabilité")
        st.plotly_chart(figr, use_container_width=True)
        st.dataframe(res.records, use_container_width=True, hide_index=True)
