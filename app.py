"""Streamlit app: probabilités par tranche pour les marchés Polymarket '# of tweets' d'Elon Musk.

Lancer :  streamlit run app.py
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import sys
import threading
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from tweetanalyst import archive as ARCH  # noqa: E402
from tweetanalyst import backtest as BT  # noqa: E402
from tweetanalyst import calibration as CAL  # noqa: E402
from tweetanalyst import crowd as CR  # noqa: E402
from tweetanalyst import data as D  # noqa: E402
from tweetanalyst import diagnostics as DG  # noqa: E402
from tweetanalyst import execution as EXE  # noqa: E402
from tweetanalyst import history as HIST  # noqa: E402
from tweetanalyst import model as M  # noqa: E402
from tweetanalyst import pathbacktest as PB  # noqa: E402
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
# Diagnostic page — "marché vs modèle" goodness-of-fit
# --------------------------------------------------------------------------- #
_VERDICT_EMO = {"normal": "🟢", "limite": "🟡", "extrême": "🔴"}


@st.cache_data(show_spinner=False, ttl=None)
def list_diagnostic_markets(handle: str, token: int = 0) -> list[dict]:
    """Recent Elon markets (closed + ongoing) with parsed counting windows, newest first.

    Pulls both resolved events (``closed=true``) and the currently open ones (``closed=false``) so an
    in-progress market shows up immediately, not only once it settles.
    """
    now = dt.datetime.now(dt.timezone.utc)
    out, seen = [], set()
    for e in PB._series_events(closed=True) + PB._series_events(closed=False):
        slug = e.get("slug", "")
        if not slug or slug in seen or slug.startswith("arch-"):
            continue
        yhint = None
        if e.get("endDate"):
            try:
                yhint = pd.to_datetime(e["endDate"], utc=True).year
            except Exception:  # noqa: BLE001
                yhint = None
        win = PB.parse_window(e.get("description"), year_hint=yhint)
        if not win:
            continue
        ws, we = win
        dur = (W.utc_ts(we) - W.utc_ts(ws)).total_seconds() / 86400.0
        if dur <= 0 or dur > 9:
            continue
        if W.utc_ts(ws) > pd.Timestamp(now):
            continue  # not yet open — nothing to diagnose
        closed = W.utc_ts(we) <= pd.Timestamp(now)
        state = "terminé" if closed else "🔴 en cours"
        seen.add(slug)
        dur_label = "7j" if dur >= 4 else f"{round(dur)}j"
        out.append({
            "slug": slug, "ws_iso": W.utc_ts(ws).isoformat(), "we_iso": W.utc_ts(we).isoformat(),
            "dur": round(dur, 1), "dur_label": dur_label, "closed": closed,
            "label": f"{_fmt_range(ws, we)}  ·  {state}",
        })
    # newest first within each duration bucket; bucket order handled by the page (group by dur_label)
    return sorted(out, key=lambda m: m["we_iso"], reverse=True)


@st.cache_data(show_spinner=True, ttl=None)
def cached_diagnose(slug: str, ws_iso: str, we_iso: str, handle: str,
                    eval_iso: str | None, n_sims: int, token: int = 0) -> dict:
    """Fit the model as of the window open and diagnose the realized stream vs the model."""
    ws = dt.datetime.fromisoformat(ws_iso)
    we = dt.datetime.fromisoformat(we_iso)
    eval_end = dt.datetime.fromisoformat(eval_iso) if eval_iso else None
    if eval_end is not None:  # ongoing market: pull the latest tweets before diagnosing
        D.ensure_history(handle)
    posts = D.load_posts(handle, end=we)
    fit = M.fit_model(posts, ws)
    res = DG.diagnose(fit, ws, we, n_sims=n_sims, rng=np.random.default_rng(7), eval_end=eval_end)
    return res


def _band_fig(x, band: dict, title: str, xlabel: str, x_is_hour: bool = False):
    """Filled model 5–95 band + median line, realized as bars coloured by tail percentile."""
    pct = band["pct"]
    colors = ["#D1495B" if (p < 0.05 or p > 0.95) else "#E8B04C" if (p < 0.15 or p > 0.85)
              else "#4C9BE8" for p in pct]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(x) + list(x)[::-1],
                             y=list(band["p95"]) + list(band["p5"])[::-1],
                             fill="toself", fillcolor="rgba(120,120,120,0.18)",
                             line=dict(width=0), hoverinfo="skip", name="modèle 5–95%"))
    fig.add_trace(go.Scatter(x=list(x), y=list(band["p50"]), mode="lines",
                             line=dict(color="#888", dash="dash"), name="médiane modèle"))
    fig.add_trace(go.Bar(x=list(x), y=list(band["realized"]), marker_color=colors, name="réalisé",
                         customdata=[f"{p*100:.0f}" for p in pct],
                         hovertemplate="%{x}<br>réalisé=%{y}<br>percentile=%{customdata}%<extra></extra>"))
    fig.update_layout(title=title, height=300, margin=dict(l=10, r=10, t=40, b=10),
                      xaxis_title=xlabel, legend=dict(orientation="h", y=-0.2),
                      bargap=0.25)
    if x_is_hour:
        fig.update_xaxes(dtick=2)
    return fig


def render_diagnostic_page() -> None:
    st.title("🔬 Diagnostic marché vs modèle")
    st.caption(
        "Pour un marché **terminé ou en cours** : on ajuste le modèle à l'ouverture de la fenêtre "
        "(données d'avant seulement), on simule la fenêtre des milliers de fois, puis on situe **chaque "
        "caractéristique réalisée** du flux de tweets d'Elon dans la distribution du modèle — "
        "🟢 normal · 🟡 limite · 🔴 extrême (queue de distribution).")
    _refresh_button("refresh_diag", list_diagnostic_markets, cached_diagnose)

    token = st.session_state.get("refresh_diag_token", 0)
    try:
        mkts = list_diagnostic_markets("elonmusk", token)
    except Exception as e:  # noqa: BLE001
        st.error(f"Liste des marchés indisponible : {e}")
        return
    if not mkts:
        st.warning("Aucun marché Elon trouvé.")
        return

    # ---- group by duration (7j, 2j, …), newest first within each group; default to the open market ----
    dur_order = sorted({m["dur_label"] for m in mkts}, key=lambda d: -int(d.rstrip("j")))
    c0, c1, c2 = st.columns([1, 3, 1])
    dur_label = c0.radio("Durée", dur_order, horizontal=True)
    group = [m for m in mkts if m["dur_label"] == dur_label]
    labels = [m["label"] for m in group]
    default_idx = next((i for i, m in enumerate(group) if not m["closed"]), 0)
    idx = c1.selectbox("Marché", range(len(group)), format_func=lambda i: labels[i], index=default_idx)
    n_sims = c2.select_slider("Simulations", [4000, 8000, 20000, 40000], value=20000,
                              help="Nombre de trajectoires Monte-Carlo tirées du modèle pour estimer "
                                   "sa distribution (médiane, bande 5–95%) sur cette fenêtre. Plus haut "
                                   "= bandes plus lisses et percentiles plus précis, mais plus lent. "
                                   "Même échelle que la page Analyse marché.")
    chosen = group[idx]

    eval_iso = None
    if not chosen["closed"]:
        eval_iso = dt.datetime.now(dt.timezone.utc).isoformat()
        st.info("Marché **en cours** : la comparaison ne porte que sur la portion déjà écoulée "
                "(« Elon suit-il le modèle jusqu'ici ? »).")

    res = cached_diagnose(chosen["slug"], chosen["ws_iso"], chosen["we_iso"], "elonmusk",
                          eval_iso, n_sims, token)
    m = res["meta"]

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Tweets réalisés", m["realized_total"])
    k2.metric("Fenêtre évaluée", f"{m['eval_hours']:.0f} h" + (" (partiel)" if m["partial"] else ""))
    k3.metric("Burstiness (α)", f"{m['alpha']:.2f}")
    k4.metric("Échelle de burst (1/β)", f"{m['burst_timescale_min']:.0f} min")

    # ---- scalar verdicts table ----
    st.subheader("Synthèse — chaque paramètre dans la distribution du modèle")
    rows = []
    for r in res["scalars"]:
        rows.append({
            "": _VERDICT_EMO.get(r["verdict"], ""),
            "Paramètre": r["param"],
            "Réalisé": round(r["realized"], 2),
            "Médiane modèle": round(r["p50"], 1),
            "Intervalle 5–95%": f"[{r['p5']:.1f} – {r['p95']:.1f}]",
            "Percentile": f"{r['pct']*100:.0f}%",
            "Verdict": f"{r['verdict']} ({r['direction']})",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("Percentile = position du réalisé parmi les simulations. ~50% = au cœur du modèle ; "
               "<5% ou >95% = comportement extrême que le modèle n'attendait quasiment pas.")

    # ---- vector diagnostics ----
    st.subheader("Détail temporel — réalisé vs bande modèle (5–95%)")
    st.caption(
        "Chaque barre = nombre réalisé sur ce créneau (heure ou jour) ; la zone grise = bande 5–95% "
        "**pour ce créneau précis** simulée par le modèle, la ligne pointillée = sa médiane. Couleur de "
        "la barre = percentile du réalisé *dans la distribution simulée de ce seul créneau* (pas du "
        "total semaine) : 🔵 normal (15–85%) · 🟡 limite (5–15% ou 85–95%) · 🔴 extrême (<5% ou >95%, "
        "le modèle n'attendait quasiment pas ce niveau à ce moment-là).")
    pday = res["per_day"]
    day_x = [str(d) for d in pday["dates"]]
    st.plotly_chart(_band_fig(day_x, pday, "Tweets par jour (ET)", "jour ET"),
                    use_container_width=True)
    g1, g2 = st.columns(2)
    with g1:
        st.plotly_chart(_band_fig(list(range(24)), res["per_hour"],
                                  "Tweets par heure du jour (ET)", "heure ET", x_is_hour=True),
                        use_container_width=True)
    with g2:
        st.plotly_chart(_band_fig(list(range(24)), res["burst_hour"],
                                  "Démarrages de burst par heure (ET)", "heure ET", x_is_hour=True),
                        use_container_width=True)
    st.caption(f"Un *burst* = salve d'au moins 2 tweets séparés de moins de "
               f"{m['gap_threshold_min']:.0f} min (≈ quelques fois l'échelle 1/β).")

    render_playback_section(chosen, "elonmusk", token)


# --------------------------------------------------------------------------- #
# Playback — replay a resolved market minute-by-minute against the user's OWN executed trades, with
# the model's live view + reliability-vs-history at every scrubbed instant.
# --------------------------------------------------------------------------- #
def _ff(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


@st.cache_data(show_spinner=False, ttl=300)
def fetch_user_trades(wallet: str, slug: str, token: int = 0) -> list[dict]:
    """The wallet's executed trades on this market's event, from Polymarket's public data-api."""
    if not wallet:
        return []
    out = []
    for page in range(20):
        try:
            r = requests.get("https://data-api.polymarket.com/trades",
                             params={"user": wallet, "limit": 500, "offset": page * 500}, timeout=20)
            d = r.json() if r.status_code == 200 else []
        except Exception:  # noqa: BLE001
            break
        if not isinstance(d, list) or not d:
            break
        for t in d:
            if t.get("eventSlug") != slug:
                continue
            title = t.get("title", "") or ""
            brk = title.split("post ", 1)[1].split(" tweets", 1)[0].strip() if "post " in title and " tweets" in title else None
            out.append({"ts": int(t["timestamp"]), "side": (t.get("side") or "").upper(),
                        "size": _ff(t.get("size")) or 0.0, "price": _ff(t.get("price")) or 0.0,
                        "outcome": (t.get("outcome") or "").capitalize(), "bracket": brk})
        if len(d) < 500:
            break
    return sorted(out, key=lambda x: x["ts"])


def _playback_market(slug: str):
    """Build a ResolvedMarket for playback directly from the Gamma event by slug — lenient (works for
    just-closed or ongoing markets not yet in the `closed=true` series). Winner=None if unresolved."""
    import json as _json
    try:
        e = requests.get("https://gamma-api.polymarket.com/events", params={"slug": slug}, timeout=20).json()
    except Exception:  # noqa: BLE001
        return None
    ev = e[0] if isinstance(e, list) and e else (e if isinstance(e, dict) else None)
    if not ev:
        return None
    yhint = pd.to_datetime(ev["endDate"], utc=True).year if ev.get("endDate") else None
    win = PB.parse_window(ev.get("description"), year_hint=yhint)
    if not win:
        return None
    ws, we = win
    brackets, winner = [], None
    for mk in ev.get("markets", []):
        label = mk.get("groupItemTitle") or ""
        lo, hi = D._parse_bracket_bounds(label)
        toks = mk.get("clobTokenIds")
        toks = _json.loads(toks) if isinstance(toks, str) else toks
        oc = mk.get("outcomes")
        oc = _json.loads(oc) if isinstance(oc, str) else oc
        op = mk.get("outcomePrices")
        op = _json.loads(op) if isinstance(op, str) else op
        yi = ([o.lower() for o in oc].index("yes") if oc and "yes" in [o.lower() for o in oc] else 0)
        brackets.append((lo, hi, label, toks[yi] if toks else None))
        if op and float(op[yi]) > 0.5:
            winner = label
    if not brackets or any(b[3] is None for b in brackets):
        return None
    brackets.sort(key=lambda b: b[0])
    return PB.HB.ResolvedMarket(slug, ws, we, 0, winner, brackets)


@st.cache_data(show_spinner="Calcul de la trajectoire du modèle sur le marché…", ttl=None)
def playback_trajectory(slug: str, handle: str, n_grid: int = 48, token: int = 0) -> dict | None:
    """Fit the model once, then re-forecast on a time grid over the window: per-bracket model prob +
    real CLOB market price at each grid instant (the deployed model's live view minute-by-minute)."""
    posts = D.load_posts(handle)
    mkt = _playback_market(slug)
    if mkt is None:
        return None
    ws, we = W.utc_ts(mkt.window_start), W.utc_ts(mkt.window_end)
    labels = [b[2] for b in mkt.brackets]
    winner_idx = next((i for i, b in enumerate(mkt.brackets) if b[2] == mkt.winner), None)
    widths = [hi - lo + 1 for (lo, hi, _, _) in mkt.brackets if np.isfinite(hi)]
    bw = float(np.median(widths)) if widths else 20.0
    dur = (we - ws).total_seconds() / 86400.0
    gamma = CAL.gamma_for_duration(dur)
    fit = M.fit_model(posts, mkt.window_start)
    brs = [D.Bracket(lab, lo, hi, None) for (lo, hi, lab, _) in mkt.brackets]
    pc = {tok: PB.HB.fetch_prices(tok, mkt.window_start, mkt.window_end) for (_, _, _, tok) in mkt.brackets}
    rng = np.random.default_rng(7)
    grid = pd.date_range(ws, we, periods=n_grid)
    probs, prices = [], []
    for T in grid:
        yes = PB.HB.market_probs_at(mkt, T.to_pydatetime(), pc)
        fc = M.forecast(dataclasses.replace(fit, now=T.to_pydatetime()), mkt.window_start,
                        mkt.window_end, n_sims=2000, rng=rng)
        tbl = M.bracket_probabilities(brs, fc.samples, gamma=gamma)
        probs.append([float(t["model_prob"]) for t in tbl])
        prices.append([float(v) for v in yes])
    return {"grid": [t.isoformat() for t in grid], "labels": labels, "winner_idx": winner_idx,
            "probs": probs, "prices": prices, "ws": ws.isoformat(), "we": we.isoformat(), "dur": dur}


def _position_at(trades: list[dict], labels: list[str], yes_prices: list[float], t_cut: int) -> dict:
    """Net position + running P&L (marked at the grid's YES prices) from trades up to ``t_cut`` (unix)."""
    shares = {(lab, oc): 0.0 for lab in labels for oc in ("Yes", "No")}
    cash = 0.0
    for tr in trades:
        if tr["ts"] > t_cut or tr["bracket"] not in labels:
            continue
        key = (tr["bracket"], tr["outcome"] if tr["outcome"] in ("Yes", "No") else "Yes")
        sgn = 1.0 if tr["side"] == "BUY" else -1.0
        shares[key] += sgn * tr["size"]
        cash -= sgn * tr["size"] * tr["price"]     # BUY spends cash, SELL gains cash
    rows, mark = [], 0.0
    for i, lab in enumerate(labels):
        sy, sn = shares[(lab, "Yes")], shares[(lab, "No")]
        if abs(sy) < 1e-6 and abs(sn) < 1e-6:
            continue
        val = sy * yes_prices[i] + sn * (1.0 - yes_prices[i])
        mark += val
        rows.append({"Tranche": lab, "OUI (parts)": round(sy, 1), "NON (parts)": round(sn, 1),
                     "Valeur @ maintenant": round(val, 2)})
    return {"rows": rows, "pnl": cash + mark, "cash": cash, "mark": mark}


@st.cache_data(show_spinner=False, ttl=None)
def _reliability_at_tau(dur: float, tau: float, top_prob: float, mtime: float) -> dict | None:
    """Compact reliability lookup at an arbitrary τ, for the playback panel (2-day history only)."""
    key, path = _hist_file_for_duration(dur)
    if path is None:
        return None
    hist = _load_hist_grid(str(path), path.stat().st_mtime)
    step = 0.05
    g = round(tau / step) * step
    pts = [t for t in {round(g - step, 3), round(g, 3)} if t > 0]
    sub = hist[np.isclose(hist["tau"].values[:, None], pts, atol=1e-6).any(axis=1)]
    if sub["slug"].nunique() < 5:
        return None
    lead = sub[sub["model_rank"] == 1]
    cond, cn = _bracket_hit_rate(lead, top_prob)
    return {"lead_hit": float(lead["is_winner"].mean()), "lead_n": len(lead),
            "cond_hit": cond, "cond_n": cn, "checkpoints": pts, "key": key}


def render_playback_section(chosen: dict, handle: str, token: int = 0) -> None:
    st.markdown("---")
    st.subheader("🎬 Playback — mes trades vs le modèle")
    if not chosen["closed"]:
        st.info("Le playback est disponible sur un marché **terminé** (rejeu complet). "
                "Sélectionne un marché clôturé ci-dessus.")
        return
    st.caption("Rejoue le marché dans le temps. **Déplace le curseur** : à cet instant précis tu vois "
               "ce que le modèle affichait, le prix du marché, **tes trades exécutés**, ta position/P&L "
               "à ce moment, et la fiabilité du modèle à ce stade (vs historique).")

    wallet = POS.load_wallet()
    tj = playback_trajectory(chosen["slug"], handle, token=token)
    if tj is None:
        st.warning("Marché introuvable dans la série résolue (prix indisponibles).")
        return
    trades = fetch_user_trades(wallet, chosen["slug"], token)
    # pd.Timestamp parses nanosecond-precision ISO strings (datetime.fromisoformat can't, on 3.9)
    grid = [pd.Timestamp(t).to_pydatetime() for t in tj["grid"]]
    labels, prices, probs = tj["labels"], tj["prices"], tj["probs"]
    win_idx = tj["winner_idx"]
    ws = pd.Timestamp(tj["ws"]).to_pydatetime()
    span = (pd.Timestamp(tj["we"]).to_pydatetime() - ws).total_seconds()

    if not trades:
        st.info(f"Aucun trade trouvé pour ton wallet (`{wallet[:6]}…{wallet[-4:]}`) sur ce marché. "
                "Le playback fonctionne quand même — il montre juste le modèle vs le marché sans tes trades.")

    # ---- time scrubber (snap to nearest grid instant) ----
    et = [(g - dt.timedelta(hours=4)).strftime("%d/%m %H:%M") for g in grid]
    gi = st.select_slider("⏱️ Instant du marché (ET)", options=list(range(len(grid))),
                          value=len(grid) - 1, format_func=lambda i: et[i], key="pb_slider")
    T = grid[gi]
    tau = float(np.clip((T - ws).total_seconds() / span, 0, 1))

    # ---- price/trade chart with a vertical 'now' line ----
    fig = go.Figure()
    for i, lab in enumerate(labels):
        is_win = i == win_idx
        fig.add_trace(go.Scatter(x=grid, y=[p[i] for p in prices], mode="lines", name=lab,
                                 line=dict(width=3 if is_win else 1,
                                           color="#1f77b4" if is_win else None),
                                 opacity=1.0 if is_win else 0.30, legendgroup=lab))
    for tr in trades:
        tt = dt.datetime.fromtimestamp(tr["ts"], dt.timezone.utc)
        if tt < ws:
            continue
        buy = tr["side"] == "BUY"
        fig.add_trace(go.Scatter(
            x=[tt], y=[tr["price"]], mode="markers", showlegend=False,
            marker=dict(symbol="triangle-up" if buy else "triangle-down", size=13,
                        color="#2ca02c" if buy else "#d62728", line=dict(width=1, color="white")),
            hovertemplate=f"{tr['side']} {tr['outcome']} {tr['bracket']}<br>"
                          f"{tr['size']:.0f} @ {tr['price']:.3f}<extra></extra>"))
    fig.add_vline(x=T, line=dict(color="#E8B04C", width=2, dash="dash"))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title="temps", yaxis=dict(title="prix OUI", range=[0, 1]),
                      legend=dict(orientation="h", y=-0.2, font=dict(size=9)))
    st.plotly_chart(fig, use_container_width=True, key="pb_chart")
    st.caption("Lignes = prix OUI par tranche (gagnante en gras). ▲ vert = tes achats · ▼ rouge = tes "
               "ventes. Trait jaune = instant sélectionné.")

    # ---- three panels at the scrubbed instant ----
    st.markdown(f"#### 📍 À **{et[gi]} ET**  ·  τ = {tau:.2f}")
    p_now, price_now = probs[gi], prices[gi]
    order = np.argsort(p_now)[::-1]
    colM, colP, colR = st.columns(3)

    with colM:
        st.markdown("**🧠 Le modèle à cet instant**")
        rows = [{"Tranche": labels[i], "Modèle": f"{p_now[i]:.0%}", "Marché": f"{price_now[i]:.2f}",
                 "Edge": f"{p_now[i]-price_now[i]:+.0%}"} for i in order[:5]]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    with colP:
        st.markdown("**💼 Ma position ici**")
        pos = _position_at(trades, labels, price_now, int(T.timestamp()))
        if pos["rows"]:
            st.dataframe(pd.DataFrame(pos["rows"]), use_container_width=True, hide_index=True)
            st.metric("P&L marqué à cet instant", f"${pos['pnl']:+.2f}")
        else:
            st.caption("Aucune position ouverte à cet instant.")

    with colR:
        st.markdown("**🎯 Fiabilité à ce stade**")
        top_i = int(order[0])
        rel = None
        _, hp = _hist_file_for_duration(tj["dur"])
        if hp is not None:
            rel = _reliability_at_tau(tj["dur"], tau, float(p_now[top_i]), hp.stat().st_mtime)
        if rel is None:
            st.caption("Historique indisponible pour cette durée de marché.")
        else:
            st.metric(f"Favori ({labels[top_i]}) — gagne à ce τ", f"{rel['lead_hit']:.0%}",
                      help=f"Tous favoris confondus, sur {rel['lead_n']} cas aux checkpoints "
                           f"{rel['checkpoints']}.")
            if not np.isnan(rel["cond_hit"]):
                st.caption(f"Conditionné à sa confiance (~{p_now[top_i]:.0%}) : "
                           f"**{rel['cond_hit']:.0%}** (n={rel['cond_n']}).")
    if win_idx is not None:
        st.caption(f"🏁 Résultat final du marché : **{labels[win_idx]}** a gagné.")


# --------------------------------------------------------------------------- #
# Tau-grid backtest page — interactive deep-dive on run_tau_backtest.py output
# --------------------------------------------------------------------------- #
_TAU_FILES = {"2 jours": Path("backtest_data/tau_backtest_2d.csv"),
              "7 jours": Path("backtest_data/tau_backtest_7d.csv")}


@st.cache_data(show_spinner=False, ttl=None)
def load_tau_backtest(path_str: str, mtime: float) -> pd.DataFrame:
    df = pd.read_csv(path_str)
    df["pos_bucket"] = df["rel_rank"].clip(-3, 3).map(
        lambda r: "0 (leader)" if r == 0 else (f"{r:+d}" if abs(r) < 3 else (f"≤{r}" if r < 0 else f"≥+{r}")))
    return df


def _kelly_roi(sub: pd.DataFrame, spread: float = 0.03, kelly: float = 0.25,
              bankroll: float = 1000.0, side: str = "OUI") -> dict:
    """Flat-$1 and quarter-Kelly ROI of buying YES (or NO, fading) at price+spread/2, held to close."""
    if side == "OUI":
        pe = np.minimum(sub["yes_price"].values + spread / 2.0, 0.999)
        win = sub["is_winner"].values.astype(float)
        p = sub["model_prob"].values
    else:
        pe = np.minimum(1.0 - sub["yes_price"].values + spread / 2.0, 0.999)
        win = (~sub["is_winner"].values).astype(float)
        p = 1.0 - sub["model_prob"].values
    flat_pnl = (win - pe) / np.maximum(pe, 1e-6)
    b = (1.0 - pe) / np.maximum(pe, 1e-6)
    f = np.clip((p * b - (1 - p)) / np.maximum(b, 1e-9), 0.0, 1.0) * kelly
    stake = bankroll * f
    shares = np.divide(stake, pe, out=np.zeros_like(stake), where=stake > 0)
    pnl = shares * win - stake
    staked = float(stake.sum())
    n = len(sub)
    return {"n": n, "win_rate": float(win.mean()) if n else float("nan"),
            "roi_flat": float(flat_pnl.mean()) if n else float("nan"),
            "roi_kelly": float(pnl.sum() / staked) if staked > 0 else float("nan")}


def _heatmap_fig(piv: pd.DataFrame, title: str, zlabel: str, zmid: float = 0.0):
    """px.imshow handles DataFrames natively and renders correctly inside Streamlit tabs."""
    piv_disp = piv.copy()
    piv_disp.columns = [f"{c:.2f}" for c in piv_disp.columns]
    piv_disp.index = [str(i) for i in piv_disp.index]
    absmax = float(np.abs(piv_disp.values).max()) or 0.1
    fig = px.imshow(piv_disp, color_continuous_scale="RdBu_r",
                    color_continuous_midpoint=zmid, zmin=-absmax, zmax=absmax,
                    labels={"x": "τ", "y": "", "color": zlabel},
                    text_auto=".2f", aspect="auto")
    fig.update_layout(title=title, height=340, margin=dict(l=10, r=10, t=40, b=10))
    fig.update_coloraxes(colorbar_title_text=zlabel)
    return fig


def render_taubacktest_page() -> None:
    st.title("📐 Backtest τ — deep-dive interactif")
    st.caption(
        "Marche chaque marché résolu sur une grille de τ régulière (modèle vs prix CLOB réel vs "
        "vainqueur final). Génère les données avec `python run_tau_backtest.py` (2j) — voir "
        "[run_tau_backtest.py](run_tau_backtest.py).")

    dur_label = st.radio("Durée des marchés", list(_TAU_FILES.keys()), horizontal=True)
    path = _TAU_FILES[dur_label]
    if not path.exists():
        st.warning(f"Pas encore généré : `{path}`. Lance `python run_tau_backtest.py` "
                   f"(adapter `DURATIONS` pour le 7j) puis rafraîchis cette page.")
        return
    df = load_tau_backtest(str(path), path.stat().st_mtime)
    n_mkts = df["slug"].nunique()
    st.caption(f"{len(df):,} lignes · {n_mkts} marchés résolus · généré "
              f"{dt.datetime.fromtimestamp(path.stat().st_mtime):%d/%m %H:%M}".replace(",", " "))

    tab_a, tab_b, tab_c, tab_d, tab_e = st.tabs([
        "🎯 Calibration du leader", "📊 Biais modèle", "💹 Biais marché", "🔍 Explorateur ROI",
        "🔬 Trajectoire d'un marché"])

    # ---- [A] calibration of the model's own leader pick, over tau ----
    with tab_a:
        st.caption("Tranche favorite du modèle (`model_rank==1`) à chaque τ : sa probabilité annoncée "
                  "tient-elle, et acheter au prix marché (+ spread) rapporte-t-il ?")
        leader = df[df["model_rank"] == 1]
        rows = []
        for tau, g in leader.groupby("tau"):
            r = _kelly_roi(g)
            rows.append({"tau": tau, "prob_modele": g["model_prob"].mean(), "win_rate": r["win_rate"],
                        "roi_flat": r["roi_flat"], "roi_kelly": r["roi_kelly"], "n": r["n"]})
        cal = pd.DataFrame(rows)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=cal["tau"], y=cal["prob_modele"], name="prob. annoncée (modèle)",
                                 mode="lines+markers", line=dict(color="#888", dash="dash")))
        fig.add_trace(go.Scatter(x=cal["tau"], y=cal["win_rate"], name="taux de victoire réel",
                                 mode="lines+markers", line=dict(color="#1f77b4", width=3)))
        fig.add_trace(go.Bar(x=cal["tau"], y=cal["roi_flat"], name="ROI flat (axe droit)",
                             marker_color="#4CAF50", opacity=0.45, yaxis="y2"))
        fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                          xaxis_title="τ (avancement de la fenêtre)",
                          yaxis=dict(title="probabilité / taux de victoire", range=[0, 1]),
                          yaxis2=dict(title="ROI flat", overlaying="y", side="right", tickformat=".0%"),
                          legend=dict(orientation="h", y=-0.18), hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True, key="tau_calib_chart")
        st.dataframe(cal.round(3), use_container_width=True, hide_index=True)

    # ---- [B] structural bias: model_prob - realized, by (tau, position relative to model leader) ----
    with tab_b:
        st.caption("Biais = probabilité moyenne du modèle − taux de victoire réel, par position relative "
                  "au leader du modèle (0) et par τ. **Rouge = le modèle SURévalue** cette position ; "
                  "**bleu = il la SOUSévalue**. Les chiffres dans chaque case = biais en points de proba.")
        order = ["≤-3", "-2", "-1", "0 (leader)", "+1", "+2", "≥+3"]
        grp = df.groupby(["tau", "pos_bucket"])
        bias = (grp["model_prob"].mean() - grp["is_winner"].mean()).rename("biais").reset_index()
        piv = bias.pivot(index="pos_bucket", columns="tau", values="biais").reindex(
            [o for o in order if o in bias["pos_bucket"].unique()])
        st.plotly_chart(_heatmap_fig(piv, "Biais du modèle (prob − réel) par position et τ", "biais"),
                        use_container_width=True, key="tau_bias_model")
        ncount = grp.size().rename("n").reset_index().pivot(index="pos_bucket", columns="tau", values="n")
        with st.expander("Effectifs (n) par case"):
            st.dataframe(ncount.reindex(piv.index), use_container_width=True)

    # ---- [C] market mispricing: yes_price - realized, by (tau, market rank) ----
    with tab_c:
        st.caption("Biais marché = prix CLOB moyen − taux de victoire réel, par rang de prix du marché "
                  "(#1 = favori du marché) et par τ. **Rouge = le marché SURpaie** (vendre/NO rentable) ; "
                  "**bleu = le marché SOUSpaie** (acheter/OUI rentable).")
        max_rank = st.slider("Rangs affichés (1 = favori marché)", 2, 8, 5, key="tau_c_rank")
        sub = df[df["market_rank"] <= max_rank]
        grp_c = sub.groupby(["tau", "market_rank"])
        bias_c = (grp_c["yes_price"].mean() - grp_c["is_winner"].mean()).rename("biais").reset_index()
        piv_c = bias_c.pivot(index="market_rank", columns="tau", values="biais").sort_index()
        st.plotly_chart(_heatmap_fig(piv_c, "Biais du marché (prix − réel) par rang et τ", "biais"),
                        use_container_width=True, key="tau_bias_market")

    # ---- [D] interactive ROI explorer ----
    with tab_d:
        st.caption("Filtre la grille complète et recalcule le ROI en direct : choisis un côté "
                  "(acheter OUI là où le modèle voit un edge positif, ou acheter NON là où il voit un "
                  "edge négatif = le marché surpaie), une position, une plage de τ, un seuil d'edge.")
        c1, c2 = st.columns(2)
        side = c1.radio("Côté", ["OUI (modèle > marché)", "NON (marché surpaie)"],
                        horizontal=True, key="tau_d_side")
        side_key = "OUI" if side.startswith("OUI") else "NON"
        edge_min = c2.slider("Edge minimum (points de proba)", 1, 20, 5,
                             format="%d pts", key="tau_d_edge") / 100.0
        tau_range = st.slider("Plage de τ", 0.05, 0.95, (0.05, 0.95), step=0.05, key="tau_d_range")
        positions = st.multiselect("Positions (vide = toutes)",
                                   ["0 (leader)", "-1", "+1", "-2", "+2", "≤-3", "≥+3"],
                                   default=[], key="tau_d_pos")

        sel = df[(df["tau"] >= tau_range[0]) & (df["tau"] <= tau_range[1])]
        if positions:
            sel = sel[sel["pos_bucket"].isin(positions)]
        sel = sel[sel["edge"] >= edge_min] if side_key == "OUI" else sel[sel["edge"] <= -edge_min]

        r = _kelly_roi(sel, side=side_key)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Trades", r["n"])
        k2.metric("Taux de victoire", f"{r['win_rate']*100:.0f}%" if r["n"] else "—")
        k3.metric("ROI flat", f"{r['roi_flat']*100:+.0f}%" if r["n"] else "—")
        k4.metric("ROI ¼-Kelly", f"{r['roi_kelly']*100:+.0f}%" if r["n"] else "—")

        if r["n"] >= 5:
            byt = []
            for tau_val, g_tau in sel.groupby("tau"):
                rr = _kelly_roi(g_tau, side=side_key)
                if rr["n"] >= 3:
                    byt.append({"tau": tau_val, **rr})
            if byt:
                bt = pd.DataFrame(byt)
                fig_d = go.Figure()
                fig_d.add_trace(go.Bar(x=bt["tau"], y=bt["roi_flat"], name="ROI flat",
                                       marker_color="#4CAF50", text=bt["n"], textposition="outside"))
                fig_d.update_layout(height=320, margin=dict(l=10, r=10, t=20, b=10),
                                    xaxis_title="τ", yaxis=dict(title="ROI flat", tickformat=".0%"),
                                    showlegend=False)
                st.plotly_chart(fig_d, use_container_width=True, key="tau_d_roi_chart")
                st.caption("Étiquette = nombre de trades dans le bucket τ.")
        else:
            st.info("Moins de 5 trades avec ces filtres — élargis la plage ou baisse le seuil d'edge.")

    # ---- [E] single-market trajectory drill-down ----
    with tab_e:
        st.caption("Pour un marché donné : trajectoire de la probabilité modèle et du prix marché pour "
                  "chaque tranche, au fil de τ. La tranche gagnante est mise en évidence.")
        slugs = sorted(df["slug"].unique(), reverse=True)
        slug_e = st.selectbox("Marché", slugs, key="tau_e_slug")
        mdf = df[df["slug"] == slug_e]
        winner = mdf.loc[mdf["is_winner"], "bracket"].iloc[0] if mdf["is_winner"].any() else None
        fig_e = go.Figure()
        for b, g_b in mdf.groupby("bracket"):
            g_b = g_b.sort_values("tau")
            is_win = b == winner
            fig_e.add_trace(go.Scatter(x=g_b["tau"], y=g_b["model_prob"], name=f"{b} · modèle",
                                       mode="lines", line=dict(width=3 if is_win else 1,
                                                               color="#1f77b4" if is_win else None),
                                       legendgroup=b, opacity=1.0 if is_win else 0.35))
            fig_e.add_trace(go.Scatter(x=g_b["tau"], y=g_b["yes_price"], name=f"{b} · marché",
                                       mode="lines", line=dict(width=3 if is_win else 1, dash="dot",
                                                               color="#E8B04C" if is_win else None),
                                       legendgroup=b, opacity=1.0 if is_win else 0.35))
        fig_e.update_layout(title=f"{slug_e} — gagnant : {winner}", height=480,
                            margin=dict(l=10, r=10, t=40, b=10), xaxis_title="τ",
                            yaxis=dict(title="probabilité", range=[0, 1]),
                            legend=dict(orientation="h", y=-0.25, font=dict(size=9)))
        st.plotly_chart(fig_e, use_container_width=True, key="tau_e_traj")
        st.caption("Trait plein = probabilité modèle · pointillé = prix marché (YES). Bleu/jaune épais "
                  "= tranche gagnante. Les autres tranches sont grisées en transparence.")


# --------------------------------------------------------------------------- #
# Live model reliability vs history: at the current market's τ, how often did the model's per-bracket
# probabilities actually come true on past *resolved* markets of the same duration? (calibration lookup)
# --------------------------------------------------------------------------- #
@st.cache_data(show_spinner=False, ttl=None)
def _load_hist_grid(path_str: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path_str)


def _hist_file_for_duration(dur: float):
    key = "2 jours" if dur < 4 else "7 jours"
    p = _TAU_FILES.get(key)
    return key, (p if (p and p.exists()) else None)


def _bracket_hit_rate(sub: pd.DataFrame, p: float, min_n: int = 15):
    """Historical realized win-rate for brackets the model rated ≈p at this τ (widen tol until min_n)."""
    m = sub
    for tol in (0.07, 0.10, 0.15, 0.25):
        m = sub[(sub["model_prob"] >= p - tol) & (sub["model_prob"] <= p + tol)]
        if len(m) >= min_n:
            return float(m["is_winner"].mean()), len(m)
    return (float(m["is_winner"].mean()) if len(m) else float("nan")), len(m)


def render_reliability_section(R: dict) -> None:
    """For the current (ongoing) market at its live τ, cross the model's per-bracket probability with
    the realized hit-rate observed at a comparable τ on resolved markets of the same duration."""
    st.markdown("### 🎯 Fiabilité du modèle à ce stade (vs historique)")
    dur = R["duration_days"]
    span = max((W.utc_ts(R["window_end"]) - W.utc_ts(R["window_start"])).total_seconds(), 1.0)
    tau = float(np.clip((W.utc_ts(R["now"]) - W.utc_ts(R["window_start"])).total_seconds() / span, 0, 1))
    key, path = _hist_file_for_duration(dur)
    if path is None:
        st.info(f"Pas encore de backtest historique pour les marchés **{key}**. "
                f"Lance `run_tau_backtest.py` (avec `DURATIONS` adapté) pour activer cette lecture.")
        return
    hist = _load_hist_grid(str(path), path.stat().st_mtime)
    # Strict, BACKWARD-looking window: the nearest grid checkpoint and the one just before it (two
    # snapshots, e.g. τ=0.70 & 0.75), never a later one — we don't compare to a more advanced stage
    # than the live market has actually reached.
    step = 0.05
    g = round(tau / step) * step                      # nearest grid checkpoint
    tau_points = sorted({round(g - step, 3), round(g, 3)})
    tau_points = [t for t in tau_points if t > 0]     # drop τ≤0 (no data at open)
    sub = hist[np.isclose(hist["tau"].values[:, None], tau_points, atol=1e-6).any(axis=1)]
    ckpt_lbl = " & ".join(f"τ={t:.2f}" for t in tau_points)
    n_mkts = sub["slug"].nunique()
    if n_mkts < 5:
        st.info(f"Trop peu de marchés {key} historiques aux checkpoints {ckpt_lbl} pour une lecture fiable.")
        return

    lead_all = sub[sub["model_rank"] == 1]
    lead_hit = float(lead_all["is_winner"].mean())
    top2 = float(sub[sub["model_rank"] <= 2].groupby("slug")["is_winner"].max().mean())

    # current market's own top bracket, and the leader hit-rate CONDITIONED on that confidence level
    top = max(R["table"], key=lambda r: r["model_prob"])
    top_p = float(top["model_prob"])
    cond_hit, cond_n = _bracket_hit_rate(lead_all, top_p, min_n=15)  # leaders rated ≈ top_p at this τ

    c1, c2, c3 = st.columns(3)
    c1.metric("τ actuel", f"{tau:.2f}",
              help="Avancement du marché en cours (0 = ouverture, 1 = clôture).")
    c2.metric("Favori du modèle gagne (à ce τ)", f"{lead_hit:.0%}",
              help=f"Tous favoris confondus : part des {len(lead_all)} cas (sur {n_mkts} marchés {key}) "
                   "où, à ce même stade, la tranche n°1 du modèle a effectivement gagné.")
    c3.metric("Top-2 contient le gagnant", f"{top2:.0%}",
              help="Part des marchés où le gagnant final était dans les 2 tranches les plus probables "
                   "du modèle à ce stade.")

    # ---- headline callout, tied to the CURRENT top bracket ----
    cond_txt = (f"a gagné <b>{cond_hit:.0%}</b> du temps (n={cond_n})" if not np.isnan(cond_hit)
                else "n'a pas assez de comparables historiques")
    st.markdown(
        f"<div style='background:rgba(80,140,230,0.16);padding:10px 14px;border-radius:8px'>"
        f"🏆 <b>Tranche la plus probable maintenant : {top['label']} ({top_p:.0%})</b><br>"
        f"<span style='font-size:0.95em'>"
        f"① <b>En ignorant sa confiance</b> — simplement « le favori désigné à ce stade, gagnant ou "
        f"non » — le favori du modèle à ce τ a été le vrai gagnant <b>{lead_hit:.0%}</b> du temps "
        f"(n={len(lead_all)}).<br>"
        f"② <b>En tenant compte de sa confiance</b> (~{top_p:.0%}) : quand le favori était aussi sûr "
        f"qu'aujourd'hui, il {cond_txt}.<br>"
        f"<i>Méthode : on rejoue le modèle au même τ sur chaque marché {key} clôturé et on compare à "
        f"la résolution réelle.</i></span></div>",
        unsafe_allow_html=True)

    # ---- how the leader behaves by its own confidence band, at this τ (leader calibration) ----
    bands = pd.cut(lead_all["model_prob"], [0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01],
                   labels=["<40%", "40–50%", "50–60%", "60–70%", "70–80%", ">80%"])
    cal_rows = []
    for b, gg in lead_all.groupby(bands, observed=True):
        if len(gg):
            cal_rows.append({"Confiance du favori": b, "A réellement gagné": float(gg["is_winner"].mean()),
                             "n": len(gg)})
    with st.expander(f"📊 Fiabilité du favori selon sa confiance, à ce stade (τ≈{tau:.2f})"):
        st.dataframe(
            pd.DataFrame(cal_rows).style.format({"A réellement gagné": "{:.0%}"}),
            use_container_width=True, hide_index=True)
        st.caption("Plus le modèle est confiant sur son favori, plus il a raison — et ces taux valident "
                   "(ou non) le niveau de confiance affiché ci-dessus.")

    rows = []
    for r in R["table"]:
        p = float(r["model_prob"])
        hit, n = _bracket_hit_rate(sub, p)
        if np.isnan(hit):
            verdict = "—"
        elif p - hit > 0.12:
            verdict = "🟠 modèle sur-confiant"
        elif hit - p > 0.12:
            verdict = "🔵 modèle sous-confiant"
        else:
            verdict = "🟢 fiable"
        rows.append({"Tranche": r["label"], "Proba modèle (maintenant)": p,
                     "Réussite historique (à ce τ)": hit, "n comparables": n, "Lecture": verdict})
    rel = pd.DataFrame(rows)
    st.dataframe(
        rel.style.format({"Proba modèle (maintenant)": "{:.0%}",
                          "Réussite historique (à ce τ)": "{:.0%}"}, na_rep="—"),
        use_container_width=True, hide_index=True, height=min(680, 38 * (len(rel) + 1)))
    st.caption(
        f"Croisé avec **{n_mkts} marchés {key} résolus**, aux checkpoints **{ckpt_lbl}** (les 2 stades les "
        f"plus proches du τ actuel de {tau:.2f}, sans jamais regarder un stade plus avancé). *Réussite "
        "historique* = parmi les tranches que le modèle notait à un niveau de proba comparable **à ce "
        "stade**, la part qui a réellement gagné. 🟢 la proba tient · 🟠 le modèle promet plus que la "
        "réalité (surévalue) · 🔵 il est plus prudent que nécessaire (sous-évalue).")


# --------------------------------------------------------------------------- #
# Decision panel — the "leader confirmé" strategy wired live (see docs/STRATEGY.md).
# 2-day: model prob + directional filter, τ∈[0.55,0.70]. 7-day: ENSEMBLE prob
# (0.5·model + 0.5·normalized price), τ∈[0.85,0.95], reduced stake.
# --------------------------------------------------------------------------- #
_BANKROLL_FILE = Path("data/bankroll.txt")


def _load_bankroll() -> float:
    try:
        return float(_BANKROLL_FILE.read_text().strip())
    except Exception:  # noqa: BLE001
        return 1000.0


def _lo_of_label(lab: str) -> float:
    lab = (lab or "").strip()
    if lab.startswith("<"):
        return 0.0
    if lab.endswith("+"):
        return float(lab[:-1])
    try:
        return float(lab.split("-")[0])
    except Exception:  # noqa: BLE001
        return 1e9


def render_decision_section(R: dict) -> None:
    st.markdown("### 🧭 Décision — stratégie « leader confirmé »")
    if R["summary"]["hours_remaining"] <= 0:
        st.caption("Marché réglé — plus de décision à prendre.")
        return
    dur = R["duration_days"]
    is2d = dur < 4
    span = (W.utc_ts(R["window_end"]) - W.utc_ts(R["window_start"])).total_seconds()
    tau = float(np.clip((W.utc_ts(R["now"]) - W.utc_ts(R["window_start"])).total_seconds() / max(span, 1), 0, 1))
    lo_t, hi_t = (0.55, 0.70) if is2d else (0.85, 0.95)

    tbl = [r for r in R["table"] if r.get("yes_price") is not None]
    if len(tbl) < 3:
        st.info("Prix marché indisponibles — décision impossible.")
        return
    psum = sum(r["yes_price"] for r in tbl) or 1.0
    dec = []
    for r in tbl:
        q = r["yes_price"] / max(psum, 0.2)
        dec.append({**r, "p_dec": r["model_prob"] if is2d else 0.5 * r["model_prob"] + 0.5 * q})
    dec.sort(key=lambda r: -r["p_dec"])
    lead, second = dec[0], dec[1]
    mkt_fav = max(dec, key=lambda r: r["yes_price"])
    edge = lead["p_dec"] - lead["yes_price"]
    conf_min, edge_min = (0.45, 0.05) if is2d else (0.45, 0.025)
    plabel = "proba modèle" if is2d else "proba ensemble (½ modèle + ½ prix)"

    gates = [
        (f"Fenêtre de décision τ∈[{lo_t:.2f}, {hi_t:.2f}]", lo_t <= tau <= hi_t, f"τ = {tau:.2f}"),
        (f"{plabel} du favori ≥ {conf_min:.0%}", lead["p_dec"] >= conf_min, f"{lead['p_dec']:.0%} ({lead['label']})"),
        (f"Edge ≥ {edge_min*100:.1f} pts", edge >= edge_min, f"{edge:+.1%}"),
        ("Prix OUI ≤ 0.80", lead["yes_price"] <= 0.80, f"{lead['yes_price']:.2f}"),
        ("Direction : favori modèle = ou > favori marché",
         _lo_of_label(lead["label"]) >= _lo_of_label(mkt_fav["label"]),
         f"{lead['label']} vs {mkt_fav['label']} (marché)"),
    ]
    all_ok = all(ok for _, ok, _ in gates)

    gdf = pd.DataFrame([{"": "✅" if ok else "❌", "Condition": name, "Valeur": val}
                        for name, ok, val in gates])
    st.dataframe(gdf, use_container_width=True, hide_index=True)

    c1, c2 = st.columns(2)
    bankroll = c1.number_input("Bankroll ($)", min_value=50.0, max_value=1e7,
                               value=_load_bankroll(), step=100.0, key="dec_bankroll")
    try:  # persist across sessions
        if abs(bankroll - _load_bankroll()) > 1e-9:
            _BANKROLL_FILE.write_text(str(bankroll))
    except Exception:  # noqa: BLE001
        pass
    frac_lab = c2.radio("Mise (protocole : 5% les 20 premiers trades)",
                        ["5%", "10%"], horizontal=True, key="dec_frac")
    frac = 0.05 if frac_lab == "5%" or not is2d else 0.10   # 7j = ligne secondaire, 5% max

    if not all_ok:
        if tau < lo_t:
            t_open = (W.utc_ts(R["window_start"]) + pd.Timedelta(seconds=span * lo_t))
            t_et = t_open.tz_convert(W.ET)
            st.info(f"⏳ **Trop tôt** — la fenêtre de décision ouvre à **{t_et:%a %d %b %H:%M} ET** "
                    f"(τ={lo_t}). Reviens à ce moment-là ; d'ici là, ne rien faire.")
        elif tau > hi_t:
            st.warning("⌛ **Fenêtre de décision passée** — on n'entre plus sur ce marché "
                       "(entrer tard dégrade fortement l'edge mesuré). Attendre le prochain.")
        else:
            failed = " · ".join(name for name, ok, _ in gates if not ok)
            st.error(f"❌ **PASSER ce marché** — condition(s) non satisfaite(s) : {failed}. "
                     "Ne pas forcer : ~60% des marchés ne qualifient pas, c'est normal.")
        return

    # qualified → build the order ticket(s)
    stake = bankroll * frac
    if is2d and lead["p_dec"] < 0.75:
        wtot = lead["p_dec"] + second["p_dec"]
        legs = [(lead, stake * lead["p_dec"] / wtot), (second, stake * second["p_dec"] / wtot)]
        mode = f"PANIER TOP-2 (leader à {lead['p_dec']:.0%} < 75% → on assure la tranche n°2)"
    else:
        legs = [(lead, stake)]
        mode = ("TOP-1" if is2d else "TOP-1 (7j, ensemble)") + f" — favori à {lead['p_dec']:.0%}"
    rows = []
    for r, amt in legs:
        pe = min(r["yes_price"] + 0.015, 0.999)
        rows.append({"Ordre": "ACHETER OUI", "Tranche": r["label"],
                     "Prix limite ≈": f"{min(r['yes_price'] + 0.01, 0.99):.2f}",
                     "Montant": f"${amt:,.0f}".replace(",", " "),
                     "Parts ≈": f"{amt / pe:,.0f}".replace(",", " ")})
    st.success(f"✅ **ACHETER — {mode}** · mise totale ${stake:,.0f} ({frac:.0%} du bankroll)".replace(",", " "))
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    st.caption("Puis **tenir jusqu'à la résolution** — pas de moyennage, pas de sortie anticipée "
               "(option : si ta tranche cote ≥0.95 à τ0.90, tu peux vendre pour éliminer le risque "
               "de bord). Règles et backtests : docs/STRATEGY.md.")


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


# --------------------------------------------------------------------------- #
# Auto-archive: on app launch, capture any freshly-closed market at 1-min fidelity in a background
# daemon thread (non-blocking). Runs once per session; near-instant after the first pass since already
# archived markets are skipped. Protects the backtest DB against Polymarket's limited price retention.
# --------------------------------------------------------------------------- #
@st.cache_resource
def _archive_singleton() -> dict:
    # process-global (survives reruns & sessions) so the archiver fires once, not per rerun
    return {"started": False, "result": None, "shown": False}


def _auto_archive_bg(state: dict) -> None:
    try:
        state["result"] = ARCH.archive_recent(handle="elonmusk", lookback=12)
    except Exception as e:  # noqa: BLE001
        state["result"] = {"error": str(e)}


_astate = _archive_singleton()
if not _astate["started"]:
    _astate["started"] = True
    threading.Thread(target=_auto_archive_bg, args=(_astate,), daemon=True).start()

_ares = _astate["result"]
if _ares and not _astate["shown"]:
    _astate["shown"] = True
    if _ares.get("new"):
        st.toast(f"🗄️ {len(_ares['new'])} marché(s) fraîchement clôturé(s) archivé(s) en 1-min "
                 f"({_ares['points_added']:,} points).".replace(",", " "))

page = st.sidebar.radio("📄 Page", ["📊 Analyse marché", "💼 Mes positions", "🎯 Stratégie",
                                    "📈 Historique", "🔬 Diagnostic modèle",
                                    "📐 Backtest τ"], index=0)
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
if page == "🔬 Diagnostic modèle":
    render_diagnostic_page()
    st.stop()
if page == "📐 Backtest τ":
    render_taubacktest_page()
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

render_decision_section(R)
render_reliability_section(R)

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
