import pandas as pd
import pandas_ta_classic as ta
import numpy as np
import logging

logger = logging.getLogger(__name__)

class MarketRegime:
    """
    Analyzes current market regime using ADX, ATR, and Bollinger Bands.
    Regimes: TRENDING_UP, TRENDING_DOWN, RANGING, VOLATILE
    """
    
    def __init__(self):
        pass
        
    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculates market regime and returns dataframe with new columns.
        """
        df = df.copy()
        
        # Calculate ADX (for trend strength)
        adx = ta.adx(df['high'], df['low'], df['close'], length=14)
        if adx is not None:
            df['ADX'] = adx['ADX_14']
            df['DMP'] = adx['DMP_14']
            df['DMN'] = adx['DMN_14']
        else:
            df['ADX'] = 0.0
            df['DMP'] = 0.0
            df['DMN'] = 0.0
            
        # Calculate ATR and ATR Average
        df['ATR_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        df['ATR_50_avg'] = df['ATR_14'].rolling(50).mean()
        
        # Calculate BB Width
        bb = ta.bbands(df['close'], length=20)
        if bb is not None:
            df['BB_width'] = bb['BBB_20_2.0'] # Bollinger Band Width
        else:
            df['BB_width'] = 0.0
            
        df['BB_width_avg'] = df['BB_width'].rolling(50).mean()
        
        df['market_regime'] = "RANGING"
        
        # Logic for Regimes
        # 1. Volatile: ATR is 50% higher than average
        cond_volatile = df['ATR_14'] > (df['ATR_50_avg'] * 1.5)
        
        # 2. Trending: ADX > 25
        cond_trend_up = (df['ADX'] > 25) & (df['DMP'] > df['DMN'])
        cond_trend_down = (df['ADX'] > 25) & (df['DMN'] > df['DMP'])
        
        # 3. Ranging: ADX < 25 or BB Width is low
        cond_ranging = (df['ADX'] <= 25) & (df['BB_width'] < df['BB_width_avg'])
        
        df.loc[cond_trend_up, 'market_regime'] = "TRENDING_UP"
        df.loc[cond_trend_down, 'market_regime'] = "TRENDING_DOWN"
        
        # Overwrite with Volatile if extreme volatility (takes precedence over trend)
        df.loc[cond_volatile, 'market_regime'] = "VOLATILE"
        
        # Overwrite with Ranging if explicitly ranging and not volatile
        df.loc[cond_ranging & ~cond_volatile, 'market_regime'] = "RANGING"
        
        # Numerical mapping for tensor
        regime_map = {
            "RANGING": 0,
            "TRENDING_UP": 1,
            "TRENDING_DOWN": -1,
            "VOLATILE": 2
        }
        df['market_regime_num'] = df['market_regime'].map(regime_map).fillna(0)
        
        return df

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    data = {
        'open': np.random.uniform(2600, 2620, 100),
        'high': np.random.uniform(2620, 2630, 100),
        'low': np.random.uniform(2590, 2600, 100),
        'close': np.random.uniform(2600, 2620, 100)
    }
    df_test = pd.DataFrame(data)
    mr = MarketRegime()
    res = mr.analyze(df_test)
    print(res[['close', 'ADX', 'ATR_14', 'market_regime']].tail())
