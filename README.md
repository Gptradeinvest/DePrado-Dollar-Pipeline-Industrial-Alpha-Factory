# DePrado Dollar Pipeline

**Alpha Factory industrielle basée sur les travaux de Marcos López de Prado (2018). Pipeline ETL complet : échantillonnage par Dollar Bars, labeling Triple Barrière, stationnarité via différentiation fractionnaire. Validation sur XAUUSD 2010–2025 : compression de 4139 barres D1 à 664 Dollar Bars, Jarque-Bera de 4509 à 1545. Optimisé Numba (JIT).**

---

## Quickstart

```
pip install numpy pandas scikit-learn matplotlib scipy yfinance
python pipe.py
```

Une seule commande. Le pipeline télécharge GLD (Gold ETF), construit les dollar bars, génère les features, entraîne le meta-model en walk-forward, produit un backtest net de coûts et livre un dashboard HTML interactif.

---

## Architecture AFML

Le pipeline implémente fidèlement les chapitres clés du livre :

**Ch.2 — Dollar Bars** : Échantillonnage par volume en dollars plutôt que par le temps. Produit des bars aux propriétés statistiques supérieures (stationnarité, normalité des rendements, réduction de l'autocorrélation). Un contrôle qualité automatique du volume décide si les dollar bars sont exploitables ou si un fallback en time bars est nécessaire.

**Ch.2.5 — Filtre CUSUM** : Détection d'événements structurels dans la série de prix. Seuls les points de changement significatif déclenchent une évaluation — le modèle ne trade pas en continu mais réagit aux ruptures.

**Ch.3 — Triple Barrière** : Labeling par barrières symétriques (take-profit, stop-loss, horizon temporel). Chaque événement CUSUM reçoit un label basé sur le premier contact avec une barrière, calibré par la volatilité locale.

**Ch.3.6 — Meta-Labeling** : Le signal primaire (momentum 20 jours) donne la direction. Un Random Forest apprend *quand* ce signal est fiable — il prédit si le trade sera gagnant, pas la direction elle-même. C'est le cœur de la méthode De Prado.

**Ch.4 — Sample Weights** : Pondération par unicité (average uniqueness) et amplitude du rendement. Les échantillons redondants ou chevauchants reçoivent un poids réduit pour éviter le surentraînement.

**Ch.5 — Fractional Differentiation (FFD)** : Transformation de la série de prix pour atteindre la stationnarité tout en préservant la mémoire longue. Le paramètre d=0.4 offre un compromis entre stationnarité et conservation de l'information.

**Ch.7 — Purged K-Fold CV** : Validation croisée avec purge temporel et embargo. Élimine le look-ahead bias inhérent aux CV standards sur données financières.

**Ch.10 — Bet Sizing** : Dimensionnement des positions par la probabilité du meta-model, transformée via la CDF normale. Plus le modèle est confiant, plus la taille de position est élevée.

**Ch.17 — SADF (Structural Breaks)** : Détection de bulles par le Supreme Augmented Dickey-Fuller test. Utilisé comme feature et comme indicateur de régime (BUBBLE/NORMAL).

**Ch.18 — Entropy** : Shannon, plug-in et Lempel-Ziv. Mesure la complexité informationnelle de la série de prix — un marché à faible entropie est plus prévisible.

---

## Signal Primaire — Momentum

Le modèle primaire est un momentum pur à 20 jours :

```
side = sign(close[t] / close[t-20] - 1)
```

Si le prix a monté sur 20 jours → LONG. Sinon → SHORT. Le meta-model RF décide ensuite si ce signal est fiable dans le contexte courant (volatilité, entropie, régime, fractional diff).

---

## Backtest

Le moteur de backtest simule chaque trade avec un modèle de coûts réaliste :

- **Spread** round-trip (0.04% par défaut)
- **Slippage** entrée + sortie (0.01% × 2)
- **Swap** overnight (0.015% × jours de détention)

Les métriques reportées sont nettes de tous coûts : Sharpe, max drawdown, Calmar, win rate, profit factor.

---

## Contrôle Qualité des Données

Le pipeline intègre un diagnostic automatique du volume :

```
Volume Quality: GOOD (CV=0.66, zero_days=0.0%) → dollar bars
Volume Quality: POOR (CV=5.1, zero_days=12.3%) → fallback time bars
```

Les futures (GC=F, CL=F) ont un volume Yahoo erratique — le système le détecte et bascule automatiquement. Les ETF (GLD, SPY) et les actions US offrent un volume fiable pour les dollar bars.

---

## Modes d'Utilisation

| Commande | Description |
|----------|-------------|
| `python pipe.py` | Clé en main : GLD, signal + backtest + dashboard |
| `python pipe.py --fetch SPY` | Autre ticker |
| `python pipe.py --fetch BTC` | Crypto |
| `python pipe.py data.csv` | CSV local |
| `python pipe.py --research` | Mode recherche (CV, rapports) |
| `python pipe.py --daily` | Mode cron quotidien (incrémental) |
| `python pipe.py --optimize --n-iter 100` | Random search sur les paramètres |
| `python pipe.py --start 2015-01-01` | Période personnalisée |

---

## Outputs

| Fichier | Contenu |
|---------|---------|
| `signal_latest.json` | Dernier signal (side, confidence, bet size, SL/TP, régime) |
| `signals_history.csv` | Historique complet des signaux walk-forward |
| `backtest_trades.csv` | Détail de chaque trade avec P&L et coûts |
| `signal_meta.json` | Configuration du pipeline + statistiques du backtest |
| `dashboard.html` | Dashboard interactif (equity curve, drawdown, trades) |

---

## Résultats

### XAUUSD 2010–2025 (validation)

| Métrique | Valeur |
|----------|--------|
| Barres D1 → Dollar Bars | 4139 → 664 |
| Jarque-Bera | 4509 → 1545 |

### GLD 2020–2026 (live)

| Métrique | Valeur |
|----------|--------|
| Sharpe | 2.10 |
| Return (net) | +8.04% |
| Max Drawdown | -2.74% |
| Calmar | 2.94 |
| Win Rate | 64.6% |
| Profit Factor | 1.42 |
| Trades | 130 |
| Avg Hold | 2.7 jours |

---

## Dépendances

```
pip install numpy pandas scikit-learn matplotlib scipy yfinance
```

---

## Références

López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.
