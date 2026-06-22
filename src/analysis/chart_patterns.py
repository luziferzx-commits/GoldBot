import pandas as pd
import numpy as np

class ChartPatternDetector:
    """Detects chart patterns like Double Tops/Bottoms and Flags."""

    @staticmethod
    def detect_double_top(df: pd.DataFrame, lookback: int = 50) -> dict:
        """Finds two peaks with similar prices."""
        if len(df) < lookback:
            return {"pattern": "NONE", "direction": "HOLD", "strength": 0.0}

        recent = df.iloc[-lookback:]
        highs = recent['high'].values
        closes = recent['close'].values

        # Simplified local maxima detection
        peaks = []
        for i in range(1, len(highs) - 1):
            if highs[i] > highs[i - 1] and highs[i] > highs[i + 1]:
                peaks.append((i, highs[i]))

        if len(peaks) >= 2:
            # Check last two peaks
            p1, p2 = peaks[-2], peaks[-1]
            diff = abs(p1[1] - p2[1]) / p1[1]
            if diff <= 0.003: # Within 0.3%
                # Check neckline break
                neckline = min(closes[p1[0]:p2[0]])
                current_price = closes[-1]
                if current_price < neckline:
                    return {"pattern": "DOUBLE_TOP", "direction": "SELL", "neckline": neckline, "strength": 0.8}
        
        return {"pattern": "NONE", "direction": "HOLD", "strength": 0.0}

    @staticmethod
    def detect_double_bottom(df: pd.DataFrame, lookback: int = 50) -> dict:
        """Finds two troughs with similar prices."""
        if len(df) < lookback:
            return {"pattern": "NONE", "direction": "HOLD", "strength": 0.0}

        recent = df.iloc[-lookback:]
        lows = recent['low'].values
        closes = recent['close'].values

        troughs = []
        for i in range(1, len(lows) - 1):
            if lows[i] < lows[i - 1] and lows[i] < lows[i + 1]:
                troughs.append((i, lows[i]))

        if len(troughs) >= 2:
            t1, t2 = troughs[-2], troughs[-1]
            diff = abs(t1[1] - t2[1]) / t1[1]
            if diff <= 0.003: # Within 0.3%
                neckline = max(closes[t1[0]:t2[0]])
                current_price = closes[-1]
                if current_price > neckline:
                    return {"pattern": "DOUBLE_BOTTOM", "direction": "BUY", "neckline": neckline, "strength": 0.8}
        
        return {"pattern": "NONE", "direction": "HOLD", "strength": 0.0}

    @staticmethod
    def detect_bull_flag(df: pd.DataFrame, lookback: int = 30) -> dict:
        """Detects strong uptrend followed by a consolidation channel."""
        if len(df) < lookback:
            return {"pattern": "NONE", "direction": "HOLD", "strength": 0.0}
        
        recent = df.iloc[-lookback:]
        # Approximation: Check if first half is strong UP, second half is slight DOWN
        mid = lookback // 2
        p1 = recent['close'].iloc[0]
        p_mid = recent['close'].iloc[mid]
        p_last = recent['close'].iloc[-1]
        
        if p_mid > p1 * 1.005: # Strong uptrend (pole)
            if p_last < p_mid and p_last > p1: # Flag consolidation
                # Check for breakout
                if df['close'].iloc[-1] > df['high'].iloc[-5:-1].max():
                    return {"pattern": "BULL_FLAG", "direction": "BUY", "breakout_level": p_last, "strength": 0.75}

        return {"pattern": "NONE", "direction": "HOLD", "strength": 0.0}

    @staticmethod
    def detect_bear_flag(df: pd.DataFrame, lookback: int = 30) -> dict:
        """Detects strong downtrend followed by a consolidation channel."""
        if len(df) < lookback:
            return {"pattern": "NONE", "direction": "HOLD", "strength": 0.0}
        
        recent = df.iloc[-lookback:]
        mid = lookback // 2
        p1 = recent['close'].iloc[0]
        p_mid = recent['close'].iloc[mid]
        p_last = recent['close'].iloc[-1]
        
        if p_mid < p1 * 0.995: # Strong downtrend (pole)
            if p_last > p_mid and p_last < p1: # Flag consolidation
                # Check for breakout
                if df['close'].iloc[-1] < df['low'].iloc[-5:-1].min():
                    return {"pattern": "BEAR_FLAG", "direction": "SELL", "breakout_level": p_last, "strength": 0.75}

        return {"pattern": "NONE", "direction": "HOLD", "strength": 0.0}
