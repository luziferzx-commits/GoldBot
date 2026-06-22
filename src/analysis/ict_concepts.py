import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ICTConcepts:
    """
    Detects ICT concepts like Fair Value Gaps (FVG), Liquidity Sweeps, and Market Structure Shifts (MSS).
    """

    @staticmethod
    def detect_fvg(df: pd.DataFrame, lookback: int = 50) -> pd.DataFrame:
        """
        Detects Bullish and Bearish Fair Value Gaps in the last 'lookback' bars.
        Bullish FVG: low[i] > high[i-2]
        Bearish FVG: high[i] < low[i-2]
        """
        df = df.copy()
        df['bullish_fvg'] = False
        df['bearish_fvg'] = False
        df['fvg_top'] = np.nan
        df['fvg_bottom'] = np.nan

        for i in range(2, len(df)):
            # Bullish FVG
            if df['low'].iloc[i] > df['high'].iloc[i-2]:
                df.iloc[i, df.columns.get_loc('bullish_fvg')] = True
                df.iloc[i, df.columns.get_loc('fvg_top')] = df['low'].iloc[i]
                df.iloc[i, df.columns.get_loc('fvg_bottom')] = df['high'].iloc[i-2]
            
            # Bearish FVG
            elif df['high'].iloc[i] < df['low'].iloc[i-2]:
                df.iloc[i, df.columns.get_loc('bearish_fvg')] = True
                df.iloc[i, df.columns.get_loc('fvg_top')] = df['low'].iloc[i-2]
                df.iloc[i, df.columns.get_loc('fvg_bottom')] = df['high'].iloc[i]

        return df

    @staticmethod
    def detect_liquidity_sweep(df: pd.DataFrame, swing_lookback: int = 20) -> pd.DataFrame:
        """
        Detects Liquidity Sweeps (Sweep High / Sweep Low).
        Sweep High: Price pierces a previous swing high but closes below it.
        Sweep Low: Price pierces a previous swing low but closes above it.
        """
        df = df.copy()
        df['sweep_high'] = False
        df['sweep_low'] = False
        df['sweep_price'] = np.nan

        # Rolling max/min for previous swings
        # Exclude current bar from the rolling window by shifting
        rolling_high = df['high'].shift(1).rolling(window=swing_lookback).max()
        rolling_low = df['low'].shift(1).rolling(window=swing_lookback).min()

        # Sweep High: high > rolling_high AND close < rolling_high
        cond_sweep_high = (df['high'] > rolling_high) & (df['close'] < rolling_high)
        df.loc[cond_sweep_high, 'sweep_high'] = True
        df.loc[cond_sweep_high, 'sweep_price'] = rolling_high

        # Sweep Low: low < rolling_low AND close > rolling_low
        cond_sweep_low = (df['low'] < rolling_low) & (df['close'] > rolling_low)
        df.loc[cond_sweep_low, 'sweep_low'] = True
        df.loc[cond_sweep_low, 'sweep_price'] = rolling_low

        return df

    @staticmethod
    def detect_mss(df: pd.DataFrame, bars_after_sweep: int = 5) -> pd.DataFrame:
        """
        Detects Market Structure Shift (MSS) following a liquidity sweep.
        Bullish MSS: After Sweep Low -> Price breaks above the recent swing high.
        Bearish MSS: After Sweep High -> Price breaks below the recent swing low.
        Simplified version: We check if close breaks the high/low of the sweep candle within N bars.
        """
        df = df.copy()
        df['bullish_mss'] = False
        df['bearish_mss'] = False

        last_sweep_low_idx = -1
        last_sweep_low_candle_high = 0.0

        last_sweep_high_idx = -1
        last_sweep_high_candle_low = 0.0

        for i in range(len(df)):
            # Track Sweeps
            if df['sweep_low'].iloc[i]:
                last_sweep_low_idx = i
                last_sweep_low_candle_high = df['high'].iloc[i]
                
            if df['sweep_high'].iloc[i]:
                last_sweep_high_idx = i
                last_sweep_high_candle_low = df['low'].iloc[i]

            # Detect Bullish MSS
            if last_sweep_low_idx != -1 and (i - last_sweep_low_idx) <= bars_after_sweep and i > last_sweep_low_idx:
                if df['close'].iloc[i] > last_sweep_low_candle_high:
                    df.iloc[i, df.columns.get_loc('bullish_mss')] = True
                    last_sweep_low_idx = -1 # Reset after trigger
                    
            # Detect Bearish MSS
            if last_sweep_high_idx != -1 and (i - last_sweep_high_idx) <= bars_after_sweep and i > last_sweep_high_idx:
                if df['close'].iloc[i] < last_sweep_high_candle_low:
                    df.iloc[i, df.columns.get_loc('bearish_mss')] = True
                    last_sweep_high_idx = -1 # Reset after trigger

        return df
