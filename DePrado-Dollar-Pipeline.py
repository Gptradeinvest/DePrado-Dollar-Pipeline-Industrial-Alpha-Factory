import os
import sys
import argparse
import logging
import json
import tkinter as tk
from tkinter import filedialog
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import jarque_bera, kurtosis
from numba import jit

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("AlphaFactory")

# ==========================================
# NUMBA OPTIMIZED FUNCTIONS
# ==========================================

@jit(nopython=True, cache=True)
def _numba_dollar_bars(
    dates: np.ndarray,
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    threshold: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    High-performance Just-In-Time compiled generation of Dollar Bars.
    """
    n = len(closes)
    
    # Micro-optimization: np.empty avoids O(N) zero-initialization
    out_dates = np.empty(n, dtype=np.int64)
    out_open = np.empty(n, dtype=np.float64)
    out_high = np.empty(n, dtype=np.float64)
    out_low = np.empty(n, dtype=np.float64)
    out_close = np.empty(n, dtype=np.float64)
    out_vol = np.empty(n, dtype=np.float64)
    
    cum_dv = 0.0
    cur_high = -1.0e9
    cur_low = 1.0e9
    cur_vol = 0.0
    cur_open = opens[0]
    
    idx = 0
    
    for i in range(n):
        dv = (highs[i] + lows[i] + closes[i]) / 3.0 * volumes[i]
        
        cum_dv += dv
        cur_high = max(cur_high, highs[i])
        cur_low = min(cur_low, lows[i])
        cur_vol += volumes[i]
        
        if cum_dv >= threshold:
            out_dates[idx] = dates[i]
            out_open[idx] = cur_open
            out_high[idx] = cur_high
            out_low[idx] = cur_low
            out_close[idx] = closes[i]
            out_vol[idx] = cur_vol
            
            idx += 1
            cum_dv = 0.0
            cur_high = -1.0e9
            cur_low = 1.0e9
            cur_vol = 0.0
            if i < n - 1:
                cur_open = opens[i+1]
                
    return out_dates[:idx], out_open[:idx], out_high[:idx], out_low[:idx], out_close[:idx], out_vol[:idx]

@jit(nopython=True, cache=True)
def _numba_triple_barrier(
    prices: np.ndarray,
    vols: np.ndarray,
    horizon: int,
    upper_width: float,
    lower_width: float
) -> np.ndarray:
    """
    Vectorized Triple Barrier Method using Numba.
    Returns: Array of labels (1: Take Profit, -1: Stop Loss, 0: Time Limit).
    """
    n = len(prices)
    labels = np.zeros(n, dtype=np.int64) 
    limit = n - horizon
    
    for i in range(limit):
        p0 = prices[i]
        vol = vols[i]
        
        if np.isnan(vol) or vol <= 1e-8:
            continue
            
        up_barrier = p0 * (1.0 + vol * upper_width)
        lo_barrier = p0 * (1.0 - vol * lower_width)
        
        for j in range(1, horizon + 1):
            curr_p = prices[i + j]
            if curr_p >= up_barrier:
                labels[i] = 1
                break
            if curr_p <= lo_barrier:
                labels[i] = -1
                break
                
    return labels

# ==========================================
# PIPELINE CLASS
# ==========================================

class AlphaPipeline:
    """
    Financial data processing pipeline for quantitative analysis.
    Handles data ingestion, dollar bar transformation, labeling, and feature engineering.
    """

    REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]

    def __init__(
        self, 
        multiplier: float = 50.0, 
        frac_d: float = 0.4,
        horizon: int = 10,
        upper_width: float = 1.0,
        lower_width: float = 1.0
    ):
        """
        Args:
            multiplier (float): Multiplier for average daily dollar volume.
            frac_d (float): Differentiation order. Must be in (0, 1).
            horizon (int): Max holding period (bars).
            upper_width (float): Multiplier for volatility to set Take Profit.
            lower_width (float): Multiplier for volatility to set Stop Loss.
        """
        # Validations
        if multiplier <= 0:
            raise ValueError(f"Multiplier must be positive, got {multiplier}")
        if not 0 < frac_d < 1:
            raise ValueError(f"frac_d must be in range (0, 1), got {frac_d}")
        if horizon < 1:
            raise ValueError(f"Horizon must be >= 1, got {horizon}")
        if upper_width <= 0 or lower_width <= 0:
            raise ValueError(f"Barrier widths must be positive. Got Upper: {upper_width}, Lower: {lower_width}")

        self.multiplier = multiplier
        self.frac_d = frac_d
        self.horizon = horizon
        self.upper_width = upper_width
        self.lower_width = lower_width

    def run(self, filepath: Optional[str] = None) -> None:
        # 1. ACQUISITION
        path = filepath if filepath else self._get_path_gui()
        
        if not path:
            logger.error("No file selected or timeout reached. Aborting.")
            return

        if not os.path.exists(path):
            logger.error(f"File not found: {path}")
            return

        logger.info(f"Loading data from: {os.path.basename(path)}")
        try:
            raw_df = self._load(path)
        except Exception as e:
            logger.critical(f"Failed to load data: {e}")
            return

        logger.info(f"Raw data loaded: {len(raw_df)} rows.")

        # 2. TRANSFORMATION (Dollar Bars)
        avg_dv = (raw_df["Close"] * raw_df["Volume"]).mean()
        threshold = avg_dv * self.multiplier
        
        try:
            logger.info(f"Generating Dollar Bars (Threshold: {threshold:,.2f})...")
            db_df = self._generate_dollar_bars(raw_df, threshold)
            logger.info(f"Dollar Bars generated: {len(db_df)} bars.")
        except ValueError as e:
            logger.error(f"Transformation failed: {e}")
            return

        if db_df.empty:
            logger.warning("Dollar Bars dataframe is empty. Lower the multiplier/threshold.")
            return

        # 3. LABELING (Triple Barrier)
        try:
            logger.info(f"Computing Labels (H={self.horizon}, U={self.upper_width}, L={self.lower_width})...")
            db_df = self._add_labels(db_df)
        except ValueError as e:
            logger.error(f"Labeling failed: {e}")
            return

        # 4. FEATURES (Fractional Differentiation)
        logger.info(f"Applying Fractional Differentiation (d={self.frac_d})...")
        db_df['Close_FracDiff'] = self._frac_diff(db_df['Close'], d=self.frac_d)

        # 5. EXPORT & REPORTING
        self._generate_report(raw_df, db_df)
        
        output_filename = f"PROCESSED_{os.path.basename(path)}"
        final_df = db_df.dropna()
        
        # Adding metadata to DataFrame attributes
        config_metadata = {
            'multiplier': self.multiplier,
            'frac_d': self.frac_d,
            'horizon': self.horizon,
            'upper_width': self.upper_width,
            'lower_width': self.lower_width
        }
        final_df.attrs['config'] = config_metadata
        
        # Save to CSV
        final_df.to_csv(output_filename)
        
        # Save metadata separately (JSON)
        meta_filename = output_filename.replace('.csv', '_meta.json')
        with open(meta_filename, 'w') as f:
            json.dump(config_metadata, f, indent=4)
        
        logger.info(f"Pipeline completed successfully.")
        logger.info(f"Output: {output_filename} ({len(final_df)} samples)")
        logger.info(f"Metadata: {meta_filename}")

    def _get_path_gui(self) -> str:
        """
        Opens file dialog with a 30s timeout to prevent server blocking.
        Returns empty string if timeout or cancelled.
        """
        try:
            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            
            result = {'path': ''}
            
            def on_timeout():
                logger.warning("GUI selection timed out (30s).")
                root.quit()
                
            timer_id = root.after(30000, on_timeout)
            path = filedialog.askopenfilename(title="Select OHLCV CSV Data")
            root.after_cancel(timer_id)
            result['path'] = path
            
            root.destroy()
            return result['path']
        except Exception as e:
            logger.warning(f"GUI unavailable or failed: {e}")
            return ""

    def _load(self, path: str) -> pd.DataFrame:
        df = pd.read_csv(path, sep=None, engine='python')
        df.columns = [c.strip().lower() for c in df.columns]
        
        mapping = {
            'date': ['date', 'time', 'timestamp'],
            'open': ['open'],
            'high': ['high'],
            'low': ['low'],
            'close': ['close'],
            'volume': ['volume', 'vol']
        }
        
        renamed = {}
        for std, aliases in mapping.items():
            for alias in aliases:
                if alias in df.columns:
                    renamed[alias] = std.capitalize()
                    break
        
        df = df.rename(columns=renamed)
        
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
            df.set_index('Date', inplace=True)
            
        missing = [col for col in self.REQUIRED_COLUMNS if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        return df[self.REQUIRED_COLUMNS].dropna().sort_index()

    def _generate_dollar_bars(self, df: pd.DataFrame, threshold: float) -> pd.DataFrame:
        if threshold <= 0:
            raise ValueError(f"Invalid threshold: {threshold}. Must be > 0.")

        dates_arr = df.index.astype(np.int64).values
        
        res_dates, res_o, res_h, res_l, res_c, res_v = _numba_dollar_bars(
            dates_arr, 
            df['Open'].values.astype(np.float64), 
            df['High'].values.astype(np.float64), 
            df['Low'].values.astype(np.float64), 
            df['Close'].values.astype(np.float64), 
            df['Volume'].values.astype(np.float64), 
            threshold
        )
        
        return pd.DataFrame({
            'Open': res_o, 
            'High': res_h, 
            'Low': res_l, 
            'Close': res_c, 
            'Volume': res_v
        }, index=pd.to_datetime(res_dates))

    def _add_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        min_required = self.horizon + 20
        if len(df) < min_required:
            raise ValueError(f"Insufficient data: {len(df)} rows, need at least {min_required}")

        df['ret'] = df['Close'].pct_change()
        df['vol'] = df['ret'].rolling(window=20).std().bfill()
        
        prices = df['Close'].values.astype(np.float64)
        vols = df['vol'].values.astype(np.float64)
        
        labels = _numba_triple_barrier(
            prices, 
            vols, 
            horizon=self.horizon, 
            upper_width=self.upper_width, 
            lower_width=self.lower_width
        )
        
        df['Label'] = labels
        return df.iloc[:-self.horizon]

    def _frac_diff(self, series: pd.Series, d: float, window: int = 100) -> pd.Series:
        if len(series) < window:
            logger.warning(f"Series too short ({len(series)} < {window}). Adjusting window.")
            window = max(2, len(series) // 2)

        weights = [1.0]
        for k in range(1, window):
            weights.append(-weights[-1] * (d - k + 1) / k)
        
        weights = np.array(weights[::-1])
        res = np.convolve(series.values, weights, mode='valid')
        
        return pd.Series(res, index=series.index[window-1:])

    def _generate_report(self, raw: pd.DataFrame, db: pd.DataFrame) -> None:
        try:
            # 1. Normality Stats
            r_t = np.log(raw['Close'] / raw['Close'].shift(1)).dropna()
            r_d = np.log(db['Close'] / db['Close'].shift(1)).dropna()
            
            kurt_t = kurtosis(r_t)
            kurt_d = kurtosis(r_d)
            
            logger.info("--- Statistical Report ---")
            logger.info(f"Kurtosis (Time Bars): {kurt_t:.4f}")
            logger.info(f"Kurtosis (Dollar Bars): {kurt_d:.4f}")

            # 2. Label Distribution Stats
            label_dist = db['Label'].value_counts()
            n_long = label_dist.get(1, 0)
            n_short = label_dist.get(-1, 0)
            n_neutral = label_dist.get(0, 0)
            total = len(db)

            logger.info("--- Label Distribution ---")
            logger.info(f"Long (1): {n_long} | Short (-1): {n_short} | Neutral (0): {n_neutral}")
            
            if total > 0:
                signal_ratio = (n_long + n_short) / total
                logger.info(f"Signal Ratio (Active/Total): {signal_ratio:.2%}")

            # 3. Visualization
            plt.figure(figsize=(10, 5))
            plt.hist(r_t, bins=100, alpha=0.5, label='Time Bars', density=True, color='gray')
            plt.hist(r_d, bins=100, alpha=0.7, label='Dollar Bars', density=True, color='blue')
            plt.legend()
            plt.title(f"Distribution Analysis (frac_d={self.frac_d})")
            plt.grid(True, alpha=0.3)
            
            output_img = "distribution_report.png"
            plt.savefig(output_img)
            logger.info(f"Distribution plot saved to: {output_img}")
            plt.close()
            
        except Exception as e:
            logger.warning(f"Could not generate statistical report: {e}")

def main():
    parser = argparse.ArgumentParser(description='Financial Data ETL & Feature Engineering Pipeline')
    
    # Core arguments
    parser.add_argument('file', nargs='?', help='Path to input CSV file', default=None)
    parser.add_argument('--multiplier', type=float, default=50.0, help='Dollar Volume threshold multiplier')
    parser.add_argument('--frac-d', type=float, default=0.4, help='Fractional differentiation order (0-1)')
    
    # Triple Barrier Hyperparameters
    parser.add_argument('--horizon', type=int, default=10, help='Triple barrier horizon (bars)')
    parser.add_argument('--upper', type=float, default=1.0, help='Upper barrier width multiplier (Take Profit)')
    parser.add_argument('--lower', type=float, default=1.0, help='Lower barrier width multiplier (Stop Loss)')
    
    args = parser.parse_args()
    
    try:
        pipeline = AlphaPipeline(
            multiplier=args.multiplier, 
            frac_d=args.frac_d,
            horizon=args.horizon,
            upper_width=args.upper,
            lower_width=args.lower
        )
        pipeline.run(args.file)
    except ValueError as e:
        logger.error(f"Configuration Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
