import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class CandlestickAnalyzer:
    """
    Analyzes candlestick patterns (Reversal and Continuation) to provide direction and strength.
    """
    
    def __init__(self):
        pass
        
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detects patterns on the given DataFrame.
        Returns the original DataFrame with new columns for pattern direction, strength, and name.
        """
        df = df.copy()
        
        o = df['open']
        h = df['high']
        l = df['low']
        c = df['close']
        
        body = (c - o).abs()
        upper_shadow = np.maximum(o, c) - h
        lower_shadow = l - np.minimum(o, c)
        range_hl = h - l
        
        # Avoid division by zero
        range_hl = range_hl.replace(0, 1e-5)
        
        df['pattern_direction'] = "NEUTRAL"
        df['pattern_strength'] = 0.0
        df['pattern_name'] = "NONE"
        
        # Reversal: Pin Bar / Hammer (Bullish)
        # Long lower shadow (> 2x body), small upper shadow, body in upper 30%
        is_hammer = (lower_shadow > 2 * body) & (upper_shadow.abs() < 0.2 * range_hl) & ((np.maximum(o, c) - l) / range_hl > 0.7)
        
        # Reversal: Shooting Star (Bearish)
        # Long upper shadow (> 2x body), small lower shadow, body in lower 30%
        is_shooting_star = (upper_shadow.abs() > 2 * body) & (lower_shadow < 0.2 * range_hl) & ((h - np.minimum(o, c)) / range_hl > 0.7)
        
        # Reversal: Engulfing
        prev_body = body.shift(1)
        prev_o = o.shift(1)
        prev_c = c.shift(1)
        is_bullish_engulfing = (prev_c < prev_o) & (c > o) & (c > prev_o) & (o < prev_c)
        is_bearish_engulfing = (prev_c > prev_o) & (c < o) & (c < prev_o) & (o > prev_c)
        
        # Continuation: Inside Bar
        prev_h = h.shift(1)
        prev_l = l.shift(1)
        is_inside_bar = (h < prev_h) & (l > prev_l)
        
        # Continuation: Marubozu (Very large body, almost no shadows)
        is_bullish_marubozu = (c > o) & (body > 0.8 * range_hl)
        is_bearish_marubozu = (c < o) & (body > 0.8 * range_hl)
        
        # Apply logic sequentially (prioritizing stronger patterns later if overlapping)
        
        # Inside Bar
        cond_inside = is_inside_bar
        df.loc[cond_inside, 'pattern_direction'] = "NEUTRAL"
        df.loc[cond_inside, 'pattern_strength'] = 0.3
        df.loc[cond_inside, 'pattern_name'] = "Inside Bar"
        
        # Hammer
        cond_hammer = is_hammer
        df.loc[cond_hammer, 'pattern_direction'] = "BUY"
        df.loc[cond_hammer, 'pattern_strength'] = 0.6 + (lower_shadow[cond_hammer] / range_hl[cond_hammer]) * 0.4
        df.loc[cond_hammer, 'pattern_name'] = "Hammer"
        
        # Shooting Star
        cond_star = is_shooting_star
        df.loc[cond_star, 'pattern_direction'] = "SELL"
        df.loc[cond_star, 'pattern_strength'] = 0.6 + (upper_shadow.abs()[cond_star] / range_hl[cond_star]) * 0.4
        df.loc[cond_star, 'pattern_name'] = "Shooting Star"
        
        # Engulfing (High Priority)
        cond_bull_eng = is_bullish_engulfing
        df.loc[cond_bull_eng, 'pattern_direction'] = "BUY"
        df.loc[cond_bull_eng, 'pattern_strength'] = 0.8
        df.loc[cond_bull_eng, 'pattern_name'] = "Bullish Engulfing"
        
        cond_bear_eng = is_bearish_engulfing
        df.loc[cond_bear_eng, 'pattern_direction'] = "SELL"
        df.loc[cond_bear_eng, 'pattern_strength'] = 0.8
        df.loc[cond_bear_eng, 'pattern_name'] = "Bearish Engulfing"
        
        # Marubozu (Very High Priority)
        cond_bull_maru = is_bullish_marubozu
        df.loc[cond_bull_maru, 'pattern_direction'] = "BUY"
        df.loc[cond_bull_maru, 'pattern_strength'] = 0.9
        df.loc[cond_bull_maru, 'pattern_name'] = "Bullish Marubozu"
        
        cond_bear_maru = is_bearish_marubozu
        df.loc[cond_bear_maru, 'pattern_direction'] = "SELL"
        df.loc[cond_bear_maru, 'pattern_strength'] = 0.9
        df.loc[cond_bear_maru, 'pattern_name'] = "Bearish Marubozu"
        
        # Ensure strength is capped at 1.0
        df['pattern_strength'] = df['pattern_strength'].clip(0.0, 1.0)
        
        return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing CandlestickAnalyzer...")
    data = {
        'open': [100, 102, 105, 104, 101],
        'high': [105, 103, 110, 106, 108],
        'low': [99, 100, 104, 98, 100],
        'close': [104, 101, 109, 100, 107]
    }
    df_test = pd.DataFrame(data)
    analyzer = CandlestickAnalyzer()
    res = analyzer.analyze(df_test)
    print(res[['open', 'high', 'low', 'close', 'pattern_direction', 'pattern_strength', 'pattern_name']])
