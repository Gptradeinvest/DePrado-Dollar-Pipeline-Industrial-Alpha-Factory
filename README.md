# DePrado Dollar Pipeline

**Alpha Factory industrielle basée sur les travaux de Marcos López de Prado (2018). Pipeline ETL complet : échantillonnage par Dollar Bars, labeling Triple Barrière, stationnarité via différentiation fractionnaire. Validation sur XAUUSD 2010–2025 : compression de 4139 barres D1 à 664 Dollar Bars, Jarque-Bera de 4509 à 1545. Optimisé Numba (JIT).**

```
pip install numpy pandas scikit-learn matplotlib scipy yfinance
python pipe.py
```

Une commande. Le pipeline télécharge les données, construit les dollar bars, génère les features, entraîne un meta-model en walk-forward, produit un backtest net de coûts et livre un dashboard HTML.

---

## Architecture AFML

| Chapitre | Module | Description |
|----------|--------|-------------|
| Ch.2 | `dollar_bars` | Échantillonnage par volume en dollars. Contrôle qualité automatique (CV, % jours zéro). Fallback time bars si volume insuffisant |
| Ch.2.5 | `cusum_filter` | Filtre CUSUM symétrique. Détection d'événements structurels, threshold auto par écart-type des log-rendements |
| Ch.3 | `triple_barrier` | Labeling TP/SL/Time calibré par volatilité locale. Barrières asymétriques paramétrables |
| Ch.3.6 | `meta_label` | Le signal primaire donne la direction, le RF apprend *quand* ce signal est fiable |
| Ch.4 | `sample_weights` | Pondération par unicité (average uniqueness) × amplitude du rendement |
| Ch.5 | `frac_diff_ffd` | Fixed-width window FFD. Stationnarité avec mémoire longue (d=0.4) |
| Ch.7 | `PurgedKFoldCV` | K-Fold avec purge temporel + embargo. Élimine le look-ahead bias |
| Ch.8 | `mda_importance` | Mean Decrease Accuracy par permutation. Élagage automatique des features bruit |
| Ch.10 | `bet_size` | Dimensionnement par transformation CDF de la probabilité du meta-model |
| Ch.11 | `CombPurgedCV` | C(N,k) chemins de backtest combinatoires avec purge |
| Ch.14 | `deflated_sharpe` | Test de significativité du Sharpe ajusté pour le multiple testing |
| Ch.17 | `sadf` | Supreme ADF — détection de régimes de bulle |
| Ch.18 | `entropy_features` | Shannon, plug-in, Lempel-Ziv sur log-rendements |

---

## Signal primaire

Momentum pur à 20 jours :

```
side = sign(close[t] / close[t-20] - 1)
```

Le meta-model Random Forest décide si ce signal est fiable dans le contexte courant (volatilité, entropie, régime SADF, fractional diff). Le MDA pruning (Ch.8) élague automatiquement les features non contributives — typiquement 4 features retenues sur 8.

---

## Backtest

Modèle de coûts réaliste appliqué à chaque trade :

| Composante | Valeur par défaut |
|------------|-------------------|
| Spread (round-trip) | 0.04% |
| Slippage (entrée + sortie) | 0.02% |
| Swap overnight | 0.015% × jours |

Métriques nettes de coûts : Sharpe, max drawdown, Calmar, win rate, profit factor, skew, kurtosis des rendements.

---

## Contrôle qualité des données

```
Volume : GOOD (CV=0.66, zéros=0.0%) → dollar bars
Volume : POOR (CV=5.1, zéros=12.3%) → fallback time bars
```

Les futures (GC=F, CL=F) ont un volume Yahoo Finance erratique. Les ETF (GLD, SPY) et actions US offrent un volume fiable. Le pipeline détecte et bascule automatiquement.

---

## Modes

| Commande | Description |
|----------|-------------|
| `python pipe.py` | Clé en main : GLD, signal + backtest + dashboard |
| `python pipe.py --fetch SPY` | Autre ticker |
| `python pipe.py --fetch BTC` | Crypto |
| `python pipe.py data.csv` | CSV local (OHLCV) |
| `python pipe.py --research` | CPCV, MDA, rapports de distribution |
| `python pipe.py --daily` | Mode cron incrémental |
| `python pipe.py --optimize --n-iter 100` | Random search + Deflated Sharpe |
| `python pipe.py --start 2015-01-01` | Période personnalisée |
| `python pipe.py --tf h4 d1 w1` | Multi-timeframe séquentiel |

---

## Outputs

| Fichier | Contenu |
|---------|---------|
| `signal_latest.json` | Dernier signal : direction, confiance, taille, SL/TP, régime |
| `signals_history.csv` | Historique walk-forward complet |
| `backtest_trades.csv` | P&L par trade avec décomposition des coûts |
| `signal_meta.json` | Configuration pipeline + statistiques backtest |
| `dashboard.html` | Dashboard interactif (equity, drawdown, P&L, trades) |
| `distribution_report.png` | Distribution des rendements, labels, SADF, weights |
| `optimize_results.json` | Meilleurs paramètres + DSR (mode optimize) |

---

## Résultats

### XAUUSD 2010–2025 (validation ETL)

| Métrique | Time Bars | Dollar Bars |
|----------|-----------|-------------|
| Barres | 4139 | 664 |
| Jarque-Bera | 4509 | 1545 |

### GLD 2020–2026 (walk-forward, net de coûts)

| Métrique | Valeur |
|----------|--------|
| Sharpe | 2.26 |
| Return | +11.64% |
| Max Drawdown | -4.35% |
| Calmar | 2.68 |
| Win Rate | 61.8% |
| Profit Factor | 1.46 |
| Trades | 131 |
| Avg Hold | 3 jours |
| Features retenues (MDA) | 4 / 8 |

---

## Statut

**Paper trading / Pédagogie** — prêt.

**Production (capital réel)** — manquant :

| Brique | Statut |
|--------|--------|
| Pipeline AFML complet | ✓ |
| Backtest réaliste (coûts) | ✓ |
| MDA pruning automatique | ✓ |
| Volume quality check | ✓ |
| Daily cron mode | ✓ |
| Dashboard interactif | ✓ |
| Deflated Sharpe Ratio | ✓ |
| Connexion broker (IBKR/MT5) | ✗ |
| Risk management live (kill switch, max DD) | ✗ |
| Monitoring / alertes (crash, data stale) | ✗ |
| Validation OOS de l'optimizer | ✗ |
| Logging persistant (fichier, rotation) | ✗ |

---

## Paramètres avancés

```bash
python pipe.py data.csv \
  --tf d1 \
  --horizon 10 \
  --momentum 20 \
  --upper 1.0 \
  --lower 1.0 \
  --frac-d 0.4 \
  --spread 0.0004 \
  --swap 0.00015 \
  --min-train 50 \
  --n-splits 5 \
  --embargo 0.01
```

---

## Dépendances

```
numpy
pandas
scikit-learn
matplotlib
scipy
yfinance
```

---

## Références

López de Prado, M. (2018). *Advances in Financial Machine Learning*. Wiley.

---

Gaëtan PRUVOT - 2026
