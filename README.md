DePrado-Dollar-Pipeline 🚀

Ce projet implémente une Alpha Factory industrielle basée sur les travaux de **Marcos López de Prado** (2018) pour transformer des données financières brutes en datasets optimisés pour le Machine Learning.

Le Processus Industriel
Contrairement à l'approche standard qui échoue souvent par overfitting, ce pipeline traite le ML financier comme un flux de production rigoureux :
1. Échantillonnage par Dollar Bars : Le temps est une mesure arbitraire ; nous créons une barre dès qu'un montant fixe est échangé pour restaurer la normalité statistique.
2. Labeling par Triple Barrière : Capture la réalité du trading (Profit Target, Stop Loss, Time Limit) en tenant compte de la dépendance au chemin.
3. Stationnarité via FracDiff : Utilise la différenciation fractionnaire ($d \approx 0.4$) pour rendre les données stationnaires tout en préservant la mémoire historique indispensable à la prédiction.

Validation Statistique (Résultats Réels)
Sur un dataset XAUUSD (2010-2025), le pipeline a démontré sa capacité à "nettoyer" le signal :
- Compression du bruit : Réduction de 4139 barres (D1) à 664 Dollar Bars.
- Restauration de la normalité : Le score Jarque-Bera est passé de 4509.22 (Time Bars) à 1545.43 (Dollar Bars).
- Maxime respectée : Stabilité de l'Information > Stabilité du Temps.

Utilisation
Lancez simplement `python DePrado-Dollar-Pipeline.py`, sélectionnez votre CSV, et obtenez instantanément votre dataset prêt pour l'entraînement.
