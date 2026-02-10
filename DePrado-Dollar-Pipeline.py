import os
import sys
import argparse
import logging
import json
import re
import tkinter as tk
from tkinter import filedialog
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import kurtosis

# Logging Configuration
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("AlphaFactory")

# ==========================================
# OPTIMIZED FUNCTIONS (PURE NUMPY)
# ==========================================

def _dollar_bars(
    opens: np.ndarray,
    highs: np.ndarray,
    lows: np.ndarray,
    closes: np.ndarray,
    volumes: np.ndarray,
    threshold: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Generation of Dollar Bars (pure Python/NumPy).
    Returns positional indices of bar closes, OHLCV arrays, and residual dollar volume.
    """
    n = len(closes)

    out_idx = np.empty(n, dtype=np.int64)
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

    bar_count = 0

    for i in range(n):
        dv = (highs[i] + lows[i] + closes[i]) / 3.0 * volumes[i]

        cum_dv += dv
        if highs[i] > cur_high:
            cur_high = highs[i]
        if lows[i] < cur_low:
            cur_low = lows[i]
        cur_vol += volumes[i]

        if cum_dv >= threshold:
            out_idx[bar_count] = i
            out_open[bar_count] = cur_open
            out_high[bar_count] = cur_high
            out_low[bar_count] = cur_low
            out_close[bar_count] = closes[i]
            out_vol[bar_count] = cur_vol

            bar_count += 1
            cum_dv = 0.0
            cur_high = -1.0e9
            cur_low = 1.0e9
            cur_vol = 0.0
            if i < n - 1:
                cur_open = opens[i + 1]

    return (
        out_idx[:bar_count], out_open[:bar_count], out_high[:bar_count],
        out_low[:bar_count], out_close[:bar_count], out_vol[:bar_count],
        cum_dv
    )


def _triple_barrier(
    prices: np.ndarray,
    vols: np.ndarray,
    horizon: int,
    upper_width: float,
    lower_width: float
) -> np.ndarray:
    """
    Triple Barrier Method (pure Python/NumPy).
    Returns: Array of labels (1: TP, -1: SL, 0: Time Limit).
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
    VOL_WARMUP = 20

    def __init__(
        self,
        multiplier: Optional[float] = None,
        frac_d: float = 0.4,
        horizon: int = 10,
        upper_width: float = 1.0,
        lower_width: float = 1.0,
        target_bars: Optional[int] = None
    ):
        if multiplier is not None and multiplier <= 0:
            raise ValueError(f"Multiplier must be positive, got {multiplier}")
        if not 0 < frac_d < 1:
            raise ValueError(f"frac_d must be in range (0, 1), got {frac_d}")
        if horizon < 1:
            raise ValueError(f"Horizon must be >= 1, got {horizon}")
        if upper_width <= 0 or lower_width <= 0:
            raise ValueError(f"Barrier widths must be positive. Got Upper: {upper_width}, Lower: {lower_width}")
        if target_bars is not None and target_bars < 1:
            raise ValueError(f"target_bars must be >= 1, got {target_bars}")

        self.multiplier = multiplier
        self.frac_d = frac_d
        self.horizon = horizon
        self.upper_width = upper_width
        self.lower_width = lower_width
        self.target_bars = target_bars

    def run(self, filepath: Optional[str] = None) -> None:
        # 1. ACQUISITION
        path = filepath if filepath else self._get_path_gui()

        if not path:
            logger.error("No file selected or timeout reached. Aborting.")
            return

        if not os.path.exists(path):
            logger.error(f"File not found: {path}")
            return

        output_dir = os.path.dirname(os.path.abspath(path))

        logger.info(f"Loading data from: {os.path.basename(path)}")
        try:
            raw_df = self._load(path)
        except Exception as e:
            logger.critical(f"Failed to load data: {e}")
            return

        logger.info(f"Raw data loaded: {len(raw_df)} rows.")

        # 2. TRANSFORMATION (Dollar Bars)
        total_dv = ((raw_df["High"] + raw_df["Low"] + raw_df["Close"]) / 3.0 * raw_df["Volume"]).sum()
        avg_dv = total_dv / len(raw_df)

        if self.target_bars is not None:
            # Explicit target
            target = self.target_bars
            threshold = total_dv / target
            logger.info(f"User-defined target: ~{target} bars")
        elif self.multiplier is not None:
            # Explicit multiplier
            threshold = avg_dv * self.multiplier
        else:
            # Auto-detect: match source time bar count (1:1 ratio, Lopez de Prado)
            target = len(raw_df)
            threshold = total_dv / target
            logger.info(f"Auto-detected threshold to match ~{target} source time bars (1:1 ratio)")

        try:
            logger.info(f"Generating Dollar Bars (Threshold: {threshold:,.2f})...")
            db_df, residual_dv = self._generate_dollar_bars(raw_df, threshold)
            logger.info(f"Dollar Bars generated: {len(db_df)} bars.")
            if residual_dv > 0:
                pct_residual = residual_dv / threshold * 100
                logger.info(f"Residual dollar volume discarded: {residual_dv:,.2f} ({pct_residual:.1f}% of threshold)")
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
        db_df = self._add_frac_diff(db_df)

        # 5. EXPORT & REPORTING
        self._generate_report(raw_df, db_df)

        output_filename = os.path.join(output_dir, f"PROCESSED_{os.path.basename(path)}")
        final_df = db_df.dropna()

        config_metadata = {
            'multiplier': self.multiplier,
            'target_bars': self.target_bars,
            'threshold': threshold,
            'effective_multiplier': threshold / avg_dv if avg_dv > 0 else None,
            'actual_bars': len(final_df),
            'source_rows': len(raw_df),
            'frac_d': self.frac_d,
            'horizon': self.horizon,
            'upper_width': self.upper_width,
            'lower_width': self.lower_width
        }
        final_df.attrs['config'] = config_metadata

        final_df.to_csv(output_filename)

        meta_filename = output_filename.replace('.csv', '_meta.json')
        with open(meta_filename, 'w') as f:
            json.dump(config_metadata, f, indent=4)

        logger.info(f"Pipeline completed successfully.")
        logger.info(f"Output: {output_filename} ({len(final_df)} samples)")
        logger.info(f"Metadata: {meta_filename}")

    @staticmethod
    def _has_display() -> bool:
        if sys.platform == 'win32':
            return True
        return bool(os.environ.get('DISPLAY') or os.environ.get('WAYLAND_DISPLAY'))

    def _get_path_gui(self) -> str:
        if not self._has_display():
            logger.warning("No display available. Cannot open file dialog.")
            return ""

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
        decimal_sep = '.'
        try:
            with open(path, 'r') as f:
                sample = ''.join(f.readline() for _ in range(5))
            if re.search(r'\d,\d{1,3}(?:[;\t\n]|$)', sample):
                decimal_sep = ','
        except Exception:
            pass

        df = pd.read_csv(path, sep=None, engine='python', decimal=decimal_sep)
        df.columns = [c.strip().lower() for c in df.columns]

        mapping = {
            'date': ['date', 'datetime', 'time', 'timestamp', 'dt', 'period'],
            'open': ['open', 'o'],
            'high': ['high', 'h'],
            'low': ['low', 'l'],
            'close': ['close', 'c', 'adj close', 'adj_close'],
            'volume': ['volume', 'vol', 'v']
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

        for col in self.REQUIRED_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        return df[self.REQUIRED_COLUMNS].dropna().sort_index()

    def _generate_dollar_bars(self, df: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, float]:
        if threshold <= 0:
            raise ValueError(f"Invalid threshold: {threshold}. Must be > 0.")

        res_idx, res_o, res_h, res_l, res_c, res_v, residual = _dollar_bars(
            df['Open'].values.astype(np.float64),
            df['High'].values.astype(np.float64),
            df['Low'].values.astype(np.float64),
            df['Close'].values.astype(np.float64),
            df['Volume'].values.astype(np.float64),
            threshold
        )

        # Map positional indices back to original datetime index
        bar_dates = df.index[res_idx]

        result = pd.DataFrame({
            'Open': res_o,
            'High': res_h,
            'Low': res_l,
            'Close': res_c,
            'Volume': res_v
        }, index=bar_dates)
        result.index.name = df.index.name or 'Date'

        return result, residual

    def _add_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        min_required = self.horizon + self.VOL_WARMUP
        if len(df) < min_required:
            raise ValueError(f"Insufficient data: {len(df)} rows, need at least {min_required}")

        df = df.copy()
        df['ret'] = df['Close'].pct_change()
        df['vol'] = df['ret'].rolling(window=self.VOL_WARMUP).std()

        df = df.iloc[self.VOL_WARMUP:]

        prices = df['Close'].values.astype(np.float64)
        vols = df['vol'].values.astype(np.float64)

        labels = _triple_barrier(
            prices,
            vols,
            horizon=self.horizon,
            upper_width=self.upper_width,
            lower_width=self.lower_width
        )

        df['Label'] = labels
        return df.iloc[:-self.horizon]

    def _add_frac_diff(self, df: pd.DataFrame, window: int = 100) -> pd.DataFrame:
        series = df['Close']
        d = self.frac_d

        if len(series) < window:
            logger.warning(f"Series too short ({len(series)} < {window}). Adjusting window.")
            window = max(2, len(series) // 2)

        weights = [1.0]
        for k in range(1, window):
            weights.append(-weights[-1] * (d - k + 1) / k)

        weights = np.array(weights[::-1])
        res = np.convolve(series.values, weights, mode='valid')

        aligned = pd.Series(res, index=series.index[window - 1:], name='Close_FracDiff')
        df = df.copy()
        df['Close_FracDiff'] = aligned
        return df

    def _generate_report(self, raw: pd.DataFrame, db: pd.DataFrame) -> None:
        try:
            r_t = np.log(raw['Close'] / raw['Close'].shift(1)).dropna()
            r_d = np.log(db['Close'] / db['Close'].shift(1)).dropna()

            kurt_t = kurtosis(r_t)
            kurt_d = kurtosis(r_d)

            logger.info("--- Statistical Report ---")
            logger.info(f"Kurtosis (Time Bars): {kurt_t:.4f}")
            logger.info(f"Kurtosis (Dollar Bars): {kurt_d:.4f}")

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

    parser.add_argument('file', nargs='?', help='Path to input CSV file', default=None)
    parser.add_argument('--multiplier', type=float, default=None, help='Dollar Volume threshold multiplier (default: auto-detect)')
    parser.add_argument('--target-bars', type=int, default=None, help='Target number of dollar bars (auto-calibrates threshold)')
    parser.add_argument('--frac-d', type=float, default=0.4, help='Fractional differentiation order (0-1)')
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
            lower_width=args.lower,
            target_bars=args.target_bars
        )
        pipeline.run(args.file)
    except ValueError as e:
        logger.error(f"Configuration Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
