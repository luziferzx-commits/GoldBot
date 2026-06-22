import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class SRAnalyzer:
    """
    Support and Resistance Analyzer using Swing High/Low and Round Numbers.
    """
    
    def __init__(self, pivot_left=10, pivot_right=5, sr_tolerance=0.001):
        self.pivot_left = pivot_left
        self.pivot_right = pivot_right
        self.sr_tolerance = sr_tolerance
        
    def find_pivots(self, df: pd.DataFrame) -> pd.DataFrame:
        """Find past pivot points without lookahead bias"""
        df = df.copy()
        
        # We can only confirm a pivot 'pivot_right' bars later
        # A high is a pivot if it's the max of [i-left : i+right]
        # But we only KNOW it at i+right.
        
        highs = df['high'].values
        lows = df['low'].values
        n = len(df)
        
        pivot_highs = np.full(n, np.nan)
        pivot_lows = np.full(n, np.nan)
        
        for i in range(self.pivot_left, n - self.pivot_right):
            window_highs = highs[i - self.pivot_left : i + self.pivot_right + 1]
            if highs[i] == np.max(window_highs):
                # At bar i + pivot_right, we officially confirm this pivot
                # We record it at the confirmation index
                confirm_idx = i + self.pivot_right
                if confirm_idx < n:
                    pivot_highs[confirm_idx] = highs[i]
                    
            window_lows = lows[i - self.pivot_left : i + self.pivot_right + 1]
            if lows[i] == np.min(window_lows):
                confirm_idx = i + self.pivot_right
                if confirm_idx < n:
                    pivot_lows[confirm_idx] = lows[i]
                    
        df['pivot_high'] = pivot_highs
        df['pivot_low'] = pivot_lows
        
        # Forward fill to always have the "last known" pivot
        df['last_pivot_high'] = df['pivot_high'].ffill()
        df['last_pivot_low'] = df['pivot_low'].ffill()
        
        return df

    def extract_sr_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extracts distance to nearest support/resistance and zone strength.
        """
        df = self.find_pivots(df)
        c = df['close']
        
        nearest_res = np.full(len(df), np.nan)
        nearest_sup = np.full(len(df), np.nan)
        zone_str = np.zeros(len(df))
        
        # Round numbers (e.g. 2600, 2650)
        # We consider multiples of 50 as round numbers
        round_nums = (c // 50) * 50
        round_sup = round_nums
        round_res = round_nums + 50
        
        # Compare with last known pivots
        last_high = df['last_pivot_high']
        last_low = df['last_pivot_low']
        
        # Nearest Resistance is min(round_res, last_high IF last_high > close)
        # Nearest Support is max(round_sup, last_low IF last_low < close)
        
        for i in range(len(df)):
            price = c.iloc[i]
            res_candidates = [round_res.iloc[i]]
            sup_candidates = [round_sup.iloc[i]]
            
            if not pd.isna(last_high.iloc[i]) and last_high.iloc[i] > price:
                res_candidates.append(last_high.iloc[i])
            if not pd.isna(last_low.iloc[i]) and last_low.iloc[i] < price:
                sup_candidates.append(last_low.iloc[i])
                
            n_res = min(res_candidates)
            n_sup = max(sup_candidates)
            
            nearest_res[i] = n_res
            nearest_sup[i] = n_sup
            
            # Simple zone strength: if distance is very close to round number or pivot, it's strong
            strength = 0
            if n_res == round_res.iloc[i]: strength += 1
            if n_sup == round_sup.iloc[i]: strength += 1
            
            # Increase strength if recent pivots exist
            if not pd.isna(last_high.iloc[i]) and not pd.isna(last_low.iloc[i]):
                strength += 1
                
            zone_str[i] = strength
            
        df['nearest_resistance'] = nearest_res
        df['nearest_support'] = nearest_sup
        df['distance_to_resistance'] = (nearest_res - c) / c
        df['distance_to_support'] = (c - nearest_sup) / c
        df['zone_strength'] = zone_str
        
        return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing SRAnalyzer...")
    data = {
        'open': np.random.uniform(2600, 2620, 100),
        'high': np.random.uniform(2620, 2630, 100),
        'low': np.random.uniform(2590, 2600, 100),
        'close': np.random.uniform(2600, 2620, 100)
    }
    df_test = pd.DataFrame(data)
    analyzer = SRAnalyzer(pivot_left=2, pivot_right=2)
    res = analyzer.extract_sr_features(df_test)
    print(res[['close', 'nearest_support', 'nearest_resistance', 'distance_to_support', 'zone_strength']].tail())
