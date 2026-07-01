# Elon Tweet Tracker — probabilités par tranche (Polymarket)

Outil qui estime, pour chaque **tranche** des marchés Polymarket *« Elon Musk # of tweets »*,
la **probabilité** que le total final tombe dans cette tranche — et la compare au prix du marché
pour faire ressortir l'**edge**.

Le cœur est un modèle statistique du processus de tweets qui prend en compte les trois effets
demandés :

| Effet | Où c'est modélisé |
|-------|-------------------|
| **Jour de la semaine** (Elon poste différemment selon le jour) | Intensité saisonnière `λ₀(jour, heure)` |
| **Heure de la journée** (rythme intra-journalier, sommeil, pics) | Idem, en heure **ET** |
| **Bursts / batches** (quand il commence, c'est une salve, pas un tweet isolé) | Processus auto-excitant de **Hawkes** |

## Principe du modèle

À l'instant `t` dans la fenêtre du marché on connaît **exactement** `N_obs` (tweets déjà postés).
Il reste à simuler `N_restant` sur le temps restant, puis :

```
total_final = N_obs + N_restant      →     P(total_final ∈ tranche)
```

`N_restant` est simulé par **Monte-Carlo** :

1. **Intensité saisonnière** `λ₀(jour, heure)` — histogramme jour×heure (ET), lissé
   circulairement et **pondéré par la récence** (demi-vie 28 j par défaut) pour refléter le
   rythme *actuel* d'Elon, pas celui d'il y a 6 mois. (`intensity.py`)
2. **Niveau hebdomadaire** bootstrappé sur les semaines récentes → capture l'incertitude sur
   « combien au total » (évite la sur-confiance).
3. **Hawkes** `λ(t) = μ(t) + Σ α·β·e^(−β(t−tⱼ))` — chaque tweet augmente temporairement
   l'intensité (bursts). `α` = part de tweets « déclenchés », `1/β` = durée typique d'une salve.
   Ajusté par **EM** sur l'historique récent. (`hawkes.py`)
4. La simulation est **amorcée** par les vrais tweets récents (momentum de burst en cours).

## Données

- **xtracker.polymarket.com** — historique horodaté des posts (c'est la **source de
  résolution officielle** du marché : posts du fil + quotes + reposts, replies exclus).
  Vérifié : 240 posts sur la fenêtre June 19–26 → tranche résolue 240-259. ✅
- **Gamma API** (`gamma-api.polymarket.com`) — tranches du marché + prix live.
- Tout est mis en cache localement dans `data/cache.db` (SQLite).

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Lancer l'app

```bash
streamlit run app.py
```

Dans l'app : choisir un marché actif (ou coller une URL/slug Polymarket), et lire le tableau
**Tranche → Proba modèle → Prix marché → Edge**, plus les graphiques (distribution simulée,
modèle vs marché, heatmap jour×heure) et un **backtest de calibration**.

## Utilisation en script

```python
import sys; sys.path.insert(0, "src")
from tweetanalyst import pipeline as P

run = P.run_forecast("elon-musk-of-tweets-june-26-july-3")
print(P.table_dataframe(run))          # tranche / proba / prix / edge
print(run.forecast.summary())          # médiane, p5–p95, n_obs, heures restantes
```

## Structure

```
app.py                     # interface Streamlit (point d'entrée)
README.md, requirements*.txt

src/tweetanalyst/          # la bibliothèque (le cœur)
  data.py        # XTracker + Gamma, cache SQLite, fenêtre de comptage (DST-safe)
  windows.py     # ET, jour/heure, grille hebdo (backtest)
  intensity.py   # intensité saisonnière jour×heure + niveau bootstrap (récence)
  hawkes.py      # fit EM + simulateur du processus auto-excitant (bursts)
  model.py       # fit + forecast Monte-Carlo + proba par tranche
  calibration.py # recalibrage γ (sharpening) par durée de marché
  backtest.py / histbacktest.py / pathbacktest.py   # replays & backtests
  strategy.py    # moteur de stratégie multi-marchés (Kelly, gates)
  crowd.py       # analyse de sur-réaction / fade
  diagnostics.py # goodness-of-fit marché vs modèle (page Diagnostic)
  archive.py     # archivage 1-min des marchés résolus (prix, trades, méta)
  positions.py / history.py / execution.py / sources.py / pipeline.py

scripts/                   # exécutables (à lancer depuis n'importe où)
  backtests/     # run_*.py  → écrivent dans backtest_data/<catégorie>/
  analysis/      # analyze_*.py  (post-mortems, dynamique de prix, τ-grid)
  tools/         # archive_markets, capture_book, snapshot_market, recalibrate, setup_credentials

data/                      # local, git-ignoré : cache.db (posts/prix/trades), wallet.txt
backtest_data/             # local, git-ignoré : tous les outputs (crowd/, historical/,
                           #   manual/, path_strategy/, postmortem/, orderbook/, …)
docs/                      # PHASE2_ACTIVATION.md, notes
```

Les scripts s'ancrent automatiquement sur la racine du repo (`os.chdir`) : on peut les lancer
de partout, p.ex. `python scripts/backtests/run_tau_backtest.py`.

## Limites / notes

- Le modèle suppose que le rythme récent est informatif du futur proche ; un changement brutal
  de comportement d'Elon (événement, voyage, polémique) n'est pas anticipé.
- `α` est borné < 0.95 pour la stabilité du processus.
- Le backtest est calibré sur les semaines présentes dans l'historique XTracker (depuis ~nov. 2025).
