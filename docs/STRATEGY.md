# Stratégie de trading — marchés Polymarket « Elon # of tweets »

*Synthèse de l'analyse approfondie du 2 juillet 2026 — 140 marchés résolus archivés en 1-min
(76×2j + 64×7j), 4,8M trades, vérité-terrain validée à 140/140.*

---

## 1. Ce que l'analyse a établi

**Fondations (Phase 0)**
- Le compte XTracker tombe dans la tranche gagnante Polymarket sur **140/140 marchés** → nos
  backtests reposent sur une vérité-terrain exacte.
- ⚠️ Risque de bord structurel : la résolution finit à **≤2 tweets d'un bord de tranche dans 26%
  des cas** (médiane 5). Aucune confiance modèle ne protège d'un tweet de dernière heure.

**Le modèle vs le marché (Phase 2)**
- Le modèle est un prévisionniste **légèrement** meilleur que le prix (Δ Brier ≈ −0.01 à −0.02 à
  la plupart des τ) — l'avantage moyen est mince ; **le profit est concentré dans les gros edges** :
  ROI réalisé +6% (edge 5-10pts) → +20% (10-20pts) → +79% (>20pts). Les micro-edges (0-5pts) perdent.
- **Mode d'échec unique : 95% des pertes = Elon a surgi AU-DESSUS de la tranche pariée.** Le modèle
  ne se trompe quasiment jamais dans l'autre sens. Les perdants ont un edge *plus gros* sur des
  tranches *moins chères* : quand le modèle contredit fortement le marché vers le bas, c'est
  généralement le marché qui a raison (il price un risque de surge que le modèle ne voit pas).
- Les edges d'**ouverture** (favori surcoté, etc.) étaient réels en janvier-février (+14pts) mais
  ont **décayé** (+3pts en juin) — le marché est devenu efficient à l'ouverture. Pas de stratégie
  d'ouverture durable.

**Coûts réels (Phase 3)**
- Écart mesuré vrai-NON vs (1−YES) : **+2,6c** → le haircut de 3c utilisé partout est réaliste.
- Fade tardif (NON sur tranche surpayée, τ0.70-0.85) au vrai prix NON : **77% win, +7,8% ROI** —
  positif mais mince. Stratégie d'appoint seulement.

---

## 2. LA STRATÉGIE — « Leader confirmé » (marchés 2 jours)

### Règles d'entrée (une seule décision par marché)

1. **QUAND** : un seul check à **τ ∈ [0.55, 0.70]** — en pratique le soir du 2ᵉ jour (~15h-22h ET
   pour une fenêtre midi→midi). Premier instant qualifiant dans cette plage.
2. **QUOI** : la tranche favorite du modèle, achetée en OUI, si TOUTES les conditions tiennent :
   - probabilité modèle **≥ 0.45** ;
   - edge (proba modèle − prix OUI) **≥ +5 pts** ;
   - prix OUI **≤ 0.80** ;
   - **FILTRE DIRECTIONNEL** : la tranche favorite du modèle est **la même ou une tranche PLUS
     HAUTE** que le favori du marché. *Jamais* parier que le marché surestime le nombre de tweets.
3. **COMBIEN** : mise fixe **10% du bankroll** (5% en version prudente). Une jambe, un marché.
4. **ENSUITE** : tenir jusqu'à la résolution. Pas de moyennage, pas de sortie anticipée.
5. **SI RIEN NE QUALIFIE** : passer. ~40% des marchés seulement sont tradés (~1,2 trade/semaine).

### Pourquoi le filtre directionnel (la découverte clé)

Sur le régime récent (mars→juin), la même règle **sans** le filtre fait 56% win / +16% ROI ;
les paris « modèle en-dessous du marché » font **21% win / −44% ROI** ; les paris « = ou au-dessus »
font **75% win / +47% ROI**. Le filtre élimine exactement le mode d'échec du modèle (sous-prévision
des surges) et rend l'edge **stable dans le temps** (77% complet → 75% récent).

### Performance attendue (base : mars→juin 2026, 24 trades)

| Métrique | Valeur |
|---|---|
| Taux de réussite | ~75% |
| ROI par trade (sur la mise) | ~+45-50% en moyenne |
| Cadence | ~1,2 trade/semaine |
| Pire mois observé | +2% (juin) — aucun mois négatif sur 6 |

### Sizing (bootstrap 20k tirages, 24 trades ≈ 4 mois, base récente)

| Mise/trade | Gain médian 4 mois | p5 | P(perte) | Drawdown médian |
|---|---|---|---|---|
| 5% | +63% | +4% | 3% | 10% |
| **10% (recommandé)** | **+145%** | **+3%** | **4%** | **19%** |
| 15% | +240% | −4% | 6% | 28% |
| 25% | +475% | −25% | 8% | 46% |

**Recommandation : 10% flat.** Au-delà, le drawdown croît plus vite que le gain médian et la
liquidité par tranche (~2M$ de volume par marché mais concentré) devient une contrainte.

### Stratégie d'appoint (optionnelle, petite taille)

Fade tardif : à τ ∈ [0.70, 0.85], acheter **NON** (au vrai prix NON) sur une tranche non-leader
que le modèle dit surpayée de ≥5pts → 77% win, +7,8% ROI/trade. À n'utiliser qu'en complément,
mise ≤5%, car l'edge est mince après coûts réels.

### 2bis. Le « billet \<40 » — jeu d'optionalité pré-ouverture (validé le 2 juil., 40 marchés)

Découverte issue d'un trade réel de l'utilisateur. La tranche `<40` achetée **avant l'ouverture**
de la fenêtre (~-30h à -12h, prix moyen 0.11, médian 0.077) n'est PAS sous-évaluée pour la
*résolution* (elle ne gagne que 10% des cas → tenir = **EV négative −11%**). Mais son
**optionalité intra-fenêtre est systématiquement sous-payée** : dans le 1er jour, une accalmie
d'Elon (nuit, pause) fait presque toujours flamber son prix — max J1 ≥ 2× l'achat dans 78% des
cas, ≥ 3× dans 55%.

**La règle** : acheter `<40` en pré-ouverture **seulement si prix ≤ 0.10** ; poser **immédiatement
un ordre limite de vente à ~2.5-3× le prix d'achat** ; ne JAMAIS tenir à la clôture (si la limite
n'est pas touchée, la position se règle presque toujours à 0).

| Cible de vente | Touchée | ROI moyen/trade | ROI médian |
|---|---|---|---|
| 2× | 78% | +55% | +100% |
| **2.5-3×** | **55-62%** | **+56-65%** | **+150-200%** |
| 4× | 42% | +70% | −100% (loterie) |

Mécanisme : c'est **l'autre face du pattern de sur-réaction de foule** déjà validé (les pics
mi-prix retombent) — ici on est le *vendeur* dans le pic. Taille : **≤1-2% du bankroll** (petit
échantillon, un seul régime couvert — avr-juin ; les 4 seules victoires à la clôture datent
toutes de mai, période calme ; en régime hyper-actif le billet peut coter 0.02 et ne jamais
spiker). L'archiveur capture désormais 48h de pré-ouverture pour suivre ce pattern dans le temps.

**Scan systématique d'optionalité (2 juil., 2 581 tranches × 140 marchés) — le billet `<40` est
UNIQUE :**
- **L'edge exige l'entrée PRÉ-ouverture** : au prix d'ouverture (~30-40% plus cher que la veille),
  le même trade tombe à EV ≈ 0. Acheter la veille (−30h à −12h), jamais après l'ouverture.
- **Pas de symétrique en queue haute** : les tranches hautes bon marché (165-189, 115-139…)
  spikent aussi (36-47% touchent 2×) mais pas assez souvent → EV −13% à −29%. Cause : l'accalmie
  nocturne est quasi certaine chaque jour (le carburant du `<40`), un burst assez gros pour
  réveiller les queues hautes ne l'est pas — et le marché price déjà la prime de surge.
- **Aucun équivalent 7 jours** : toutes les cases du scan 7j sont à EV ≤ 0 (une nuit calme ne
  change presque rien à la proba d'un total sur 7 jours ; les repricings intra-fenêtre sont trop
  faibles par rapport au spread sur des tickets à 0.03-0.08).
- Sensibilité honnête : l'EV moyen du `<40` varie selon la bande de prix d'entrée et la période
  (+9% à +65% selon l'échantillon) ; la **médiane** à cible 2-2.5× reste robustement positive.
  Le pattern reste classé "petit billet" (≤1-2%), à re-mesurer quand l'archive pré-ouverture
  (désormais capturée) aura accumulé quelques mois.

---

## 3. Garde-fous & honnêteté intellectuelle

1. **Échantillon petit** : la validation récente = 24 trades. Les intervalles sont larges ;
   juger la stratégie sur ≥20 trades, pas sur 5.
2. **Le filtre directionnel a été découvert sur ces données** (risque d'overfit résiduel), mais il
   a une histoire causale solide (faiblesse connue du timing horaire du modèle) et tient sur
   chaque sous-période. À re-vérifier tous les ~20 trades.
3. **L'edge décroît** : le marché s'améliore (les edges d'ouverture sont morts, le ROI mid-window a
   baissé de +70% → +47%). S'attendre à une érosion continue ; re-mesurer mensuellement
   (`scripts/analysis/analyze_tau_backtest.py` + re-run du τ-backtest).
4. **Risque de bord** : 26% des marchés finissent à ≤2 tweets d'un bord. Le 10% de mise doit être
   pensé comme perdable à chaque trade, quelle que soit la confiance affichée.
5. **7 jours — verdict du τ-backtest (64 marchés)** : le favori du modèle n'est fiable que très
   tard (avant τ0.85 le pari PERD : −36% à τ0.70-0.85 même filtré). **Jouable uniquement le dernier
   jour (τ ∈ [0.85, 0.95])**, et avec la **proba d'ENSEMBLE** plutôt que la proba modèle brute :
   `p_dec = 0.5·proba_modèle + 0.5·prix_normalisé`, gates conf≥0.45, edge_ensemble≥2.5pts,
   prix≤0.80, + filtre directionnel → **88% win / +52% complet, 80% / +40% récent** (vs 76%/+39%
   et 67%/+26% avec le modèle seul). Ligne secondaire à mise réduite (5%). NB : sur le **2j**, le
   test d'ensemble n'apporte rien — le filtre directionnel binaire est déjà optimal, la règle cœur
   reste inchangée.

6. **Fiabilité statistique** : IC95% de la règle cœur (récent, n=24) = win [56–89%], ROI [+14–+80%]
   — même le bas des intervalles est positif, mais attente réaliste en live ≈ +20-30%/trade (biais
   de sélection). Protocole : démarrer à **5% de mise** sur les 15-20 premiers trades ; passer à 10%
   si confirmé ; **arrêt et ré-analyse si win<55% après 20 trades**.

7. **Révision du modèle — testée et REJETÉE (2 juil. 2026)** : l'analyse PIT sur les 140 marchés
   montre que la calibration marginale est déjà bonne (queue haute 2j : 8-10% observé vs 10%
   attendu — PAS de biais de surge *global* ; le déficit de surge est *conditionnel* au désaccord
   modèle/marché, déjà traité par le filtre directionnel). Les petits défauts marginaux (queue
   basse 2j trop grasse, 7j sous-dispersé) ont été corrigés par remapping PIT et testés en
   **walk-forward strict** (appris jan-avr → testé mai-juin) : **la correction DÉGRADE** (Brier
   2j 0.467→0.478 ; stratégie 7j 71%/+26% → 50%/−20%). Conclusion : **le cœur du modèle reste
   inchangé** ; l'information de marché s'intègre au niveau de la couche de décision (filtre
   directionnel 2j, ensemble 7j), pas dans le modèle. Pistes futures non testées : prior horaire
   adaptatif, Hawkes multi-échelles — chantiers lourds, gain incertain.

   **Corollaire — sur-cote de la queue basse (`<40`) : correction testée & REJETÉE (2 juil.)** :
   le modèle sur-estime `<40` (12,8% vs 6,7% réel ; le marché est juste à 8,4%) car sa queue gauche
   lisse ignore le plancher comportemental d'Elon (min 27, <40 seulement 5/76 fois) et la
   quasi-certitude d'un burst sur 48h. Un shrink de la tranche la plus basse (1 paramètre, appris
   sur le Brier) **paraissait** marcher sur un split unique (+64% ROI, Brier 0,467→0,454) mais
   **échoue en validation roulante mensuelle** : `s` instable (0,4→0,8→0,6→1,0), gain Brier positif
   1 mois sur 4, stratégie dégradée (66% vs 75%). Cause : événement rare (7%, n=76) + dépendance au
   régime → incorrigible avec un paramètre fixe. Le **filtre directionnel** (0 paramètre) neutralise
   déjà ce défaut de façon robuste. Modèle inchangé, définitivement.

## 5. Pipeline de données (tout est local, rien n'est re-téléchargé)

- **Tweets** : `data/cache.db::posts` — lecture locale pure dans tous les backtests ; refresh
  **incrémental** (`ensure_history`, comptabilité `fetched_ranges`) uniquement quand l'app tourne.
- **Marchés résolus** : archivés automatiquement en 1-min (YES+NON+trades+scalaires) au lancement
  de l'app, et par le **sync hebdomadaire**.
- **Sync hebdomadaire** : agent launchd `com.tweetanalyst.sync` (lundi 10h, rattrapé au réveil) →
  `scripts/tools/sync_data.py` (tweets incrémentaux + marchés fraîchement résolus). Log :
  `data/sync.log`. Désactiver : `launchctl unload ~/Library/LaunchAgents/com.tweetanalyst.sync.plist`.

## 4. Outillage

- Check du soir : page **📊 Analyse marché** (proba, edge, fiabilité historique au τ courant).
- Revue post-marché : page **🔬 Diagnostic** (fit du modèle) + **🎬 Playback** (tes trades vs modèle).
- Re-mesure de l'edge : `python scripts/backtests/run_tau_backtest.py [2|7]` puis
  `python scripts/analysis/analyze_tau_backtest.py`.
- Données : tout est archivé en local (1-min, trades, NON) via `scripts/tools/archive_markets.py`.
