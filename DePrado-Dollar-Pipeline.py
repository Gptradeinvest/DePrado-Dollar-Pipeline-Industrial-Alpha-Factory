import pandas as pd
import numpy as np
import os
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt
from scipy.stats import jarque_bera, kurtosis

class AlphaPipeline:
    def __init__(self, multiplier=100.0):
        self.multiplier = multiplier

    def run(self):
        # 1. ACQUISITION (Productif : pas de lignes de commande)
        path = self._get_path()
        if not path: return
        
        raw_df = self._load(path)
        
        # 2. TRANSFORMATION (Dollar Bars : Stabilité de l'info > Temps)
        # Le seuil est calculé automatiquement à 5x la moyenne journalière
        daily_vol_avg = (raw_df["High"] + raw_df["Low"] + raw_df["Close"]) / 3.0 * raw_df["Volume"] * self.multiplier
        threshold = daily_vol_avg.mean() * 5
        
        db_df = self._to_dollar_bars(raw_df, threshold)
        
        # 3. LABELING (Triple Barrière : Capturer la réalité du trading)
        db_df = self._add_labels(db_df)
        
        # 4. FEATURES (FracDiff d=0.4 : Stationnarité + Mémoire)
        db_df['Close_FracDiff'] = self._frac_diff(db_df['Close'], d=0.4)
        
        # 5. ANALYSE IMMÉDIATE (Processus visuel et CLI)
        self._report(raw_df, db_df)
        
        # Sauvegarde automatique du dataset final "prêt à l'IA"
        output = f"PROD_READY_{os.path.basename(path)}"
        db_df.dropna().to_csv(output)
        print(f"\n[SUCCÈS] Dataset productif exporté : {output}")

    def _get_path(self):
        root = tk.Tk(); root.withdraw(); root.attributes("-topmost", True)
        return filedialog.askopenfilename(title="Importer CSV D1")

    def _load(self, path):
        df = pd.read_csv(path, sep=None, engine='python')
        df.columns = [c.strip().lower() for c in df.columns]
        mapping = {'date': ['date', 'time'], 'open': ['open'], 'high': ['high'], 
                   'low': ['low'], 'close': ['close'], 'volume': ['volume']}
        df = df.rename(columns={alias: std.capitalize() for std, aliases in mapping.items() for alias in aliases if alias in df.columns})
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
        return df[["Open", "High", "Low", "Close", "Volume"]].dropna().sort_index()

    def _to_dollar_bars(self, df, threshold):
        dv = ((df["High"] + df["Low"] + df["Close"]) / 3.0) * df["Volume"] * self.multiplier
        bars, cum = [], 0.0
        o, h, l, v = df["Open"].iloc[0], -np.inf, np.inf, 0.0
        for i in range(len(df)):
            cum += dv.iloc[i]; h = max(h, df["High"].iloc[i]); l = min(l, df["Low"].iloc[i]); v += df["Volume"].iloc[i]
            if cum >= threshold:
                bars.append({'Date': df.index[i], 'Open': o, 'High': h, 'Low': l, 'Close': df["Close"].iloc[i], 'Volume': v, 'DollarVolume': cum})
                cum = 0.0
                if i+1 < len(df): o, h, l, v = df["Open"].iloc[i+1], -np.inf, np.inf, 0.0
        return pd.DataFrame(bars).set_index('Date')

    def _add_labels(self, df):
        # Triple Barrière : Profit (+1), Loss (-1), Time (0)
        df['ret'] = df['Close'].pct_change()
        df['vol'] = df['ret'].rolling(20).std()
        labels = []
        for i in range(len(df)-5):
            p, v = df['Close'].iloc[i], df['vol'].iloc[i]
            if np.isnan(v): labels.append(np.nan); continue
            up, lo, target = p*(1+v), p*(1-v), 0
            for j in range(1, 6):
                if df['Close'].iloc[i+j] >= up: target = 1; break
                if df['Close'].iloc[i+j] <= lo: target = -1; break
            labels.append(target)
        df['Label'] = pd.Series(labels, index=df.index[:len(labels)])
        return df

    def _frac_diff(self, series, d):
        weights = [1.0]
        for k in range(1, 100): weights.append(-weights[-1] * (d - k + 1) / k)
        weights = np.array(weights[::-1]).reshape(-1, 1)
        res = {}
        vals = series.values.reshape(-1, 1)
        for i in range(len(weights), len(vals)):
            res[series.index[i]] = np.dot(weights.T, vals[i-len(weights):i])[0,0]
        return pd.Series(res)

    def _report(self, raw, db):
        r_t = np.log(raw['Close']/raw['Close'].shift(1)).dropna()
        r_d = np.log(db['Close']/db['Close'].shift(1)).dropna()
        print("\n" + "="*50 + "\n RAPPORT ALPHA FACTORY\n" + "="*50)
        print(f"Time Bars (D1) Kurtosis   : {kurtosis(r_t):.4f}")
        print(f"Dollar Bars Kurtosis       : {kurtosis(r_d):.4f}")
        print(f"Jarque-Bera Dollar Bars    : {jarque_bera(r_d)[0]:.2f}")
        print("-" * 50)
        plt.figure(figsize=(10,4))
        plt.hist(r_t, bins=50, alpha=0.5, label='Time', density=True)
        plt.hist(r_d, bins=50, alpha=0.5, label='Dollar', density=True)
        plt.title("Restauration de la Normalité"); plt.legend(); plt.show()

if __name__ == "__main__":
    AlphaPipeline().run()