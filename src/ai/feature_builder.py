import logging
import pandas as pd
import numpy as np
import pandas_ta_classic as ta
import torch
from typing import Tuple, List, Optional
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore') # Ignore pandas TA warnings

from src.analysis.candlestick_patterns import CandlestickAnalyzer
from src.analysis.support_resistance import SRAnalyzer
from src.analysis.market_regime import MarketRegime

logger = logging.getLogger(__name__)

class FeatureBuilder:
    """
    Builds features from multi-timeframe data for the AI model.
    """
    
    def __init__(self, seq_len: int = 60):
        self.seq_len = seq_len
        self.scaler = MinMaxScaler()
        self.is_fitted = False

    def _compute_m5_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        
        # Technical Indicators
        df['EMA_9'] = ta.ema(df['close'], length=9)
        df['EMA_21'] = ta.ema(df['close'], length=21)
        df['EMA_50'] = ta.ema(df['close'], length=50)
        df['RSI_14'] = ta.rsi(df['close'], length=14)
        
        macd = ta.macd(df['close'])
        if macd is not None:
            df['MACD'] = macd['MACD_12_26_9']
            df['MACD_hist'] = macd['MACDh_12_26_9']
        else:
            df['MACD'] = 0.0
            df['MACD_hist'] = 0.0
            
        df['ATR_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        
        bbands = ta.bbands(df['close'], length=20)
        if bbands is not None:
            df['BB_upper'] = bbands['BBU_20_2.0']
            df['BB_lower'] = bbands['BBL_20_2.0']
        else:
            df['BB_upper'] = df['close']
            df['BB_lower'] = df['close']
            
        # Volume features
        df['vol_ratio'] = df['tick_volume'] / df['tick_volume'].rolling(20).mean()
        
        # Time features
        df['hour'] = df.index.hour
        df['day_of_week'] = df.index.dayofweek
        
        # Session flags (approximate GMT times)
        df['is_london'] = ((df['hour'] >= 8) & (df['hour'] <= 16)).astype(int)
        df['is_ny'] = ((df['hour'] >= 13) & (df['hour'] <= 21)).astype(int)
        df['is_asia'] = ((df['hour'] >= 0) & (df['hour'] <= 8)).astype(int)
        
        # Add new Candlestick and SR features
        candle_analyzer = CandlestickAnalyzer()
        df = candle_analyzer.analyze(df)
        
        sr_analyzer = SRAnalyzer()
        df = sr_analyzer.extract_sr_features(df)
        
        # Add Market Regime
        regime_analyzer = MarketRegime()
        df = regime_analyzer.analyze(df)
        
        # Encode categorical 'pattern_direction'
        dir_map = {"BUY": 1, "SELL": -1, "NEUTRAL": 0}
        df['pattern_dir_num'] = df['pattern_direction'].map(dir_map).fillna(0)
        
        # Ensure external features exist (they might be missing if external factors failed to fetch)
        ext_cols = ['dxy_change', 'us10y_change', 'vix_level', 'oil_change', 'sp500_change', 'sentiment_score', 'gold_bias']
        for col in ext_cols:
            if col not in df.columns:
                df[col] = 0.0
                if col == 'vix_level': df[col] = 15.0 # default VIX
        
        return df.fillna(method='bfill').fillna(0)

    def build_features(
        self,
        m5_data: pd.DataFrame,
        m15_data: pd.DataFrame = None,
        h1_data: pd.DataFrame = None,
        daily_data: pd.DataFrame = None,
        monthly_data: pd.DataFrame = None,
        fit_scaler: bool = False
    ) -> Optional[torch.Tensor]:
        """
        Builds the complete feature tensor.
        """
        if m5_data is None or len(m5_data) < self.seq_len + 50:
            logger.error("Not enough M5 data to build features.")
            return None
            
        df = self._compute_m5_features(m5_data)
        
        # Select columns to use as features (29 features total)
        feature_cols = [
            'close', 'EMA_9', 'EMA_21', 'EMA_50', 'RSI_14', 
            'MACD', 'MACD_hist', 'ATR_14', 'BB_upper', 'BB_lower', 
            'vol_ratio', 'hour', 'day_of_week', 'is_london', 'is_ny', 'is_asia',
            'pattern_dir_num', 'pattern_strength', 'distance_to_resistance', 
            'distance_to_support', 'zone_strength',
            'market_regime_num', 'dxy_change', 'us10y_change', 'vix_level', 
            'oil_change', 'sp500_change', 'sentiment_score', 'gold_bias'
        ]
        
        # Also store these feature cols length to pass to model
        self.feature_size = len(feature_cols)
        
        features_np = df[feature_cols].values
        
        if fit_scaler:
            features_np = self.scaler.fit_transform(features_np)
            self.is_fitted = True
        elif self.is_fitted:
            features_np = self.scaler.transform(features_np)
        else:
            # If not fitted and fit_scaler=False, we just fit it (e.g. for simple test)
            features_np = self.scaler.fit_transform(features_np)
            self.is_fitted = True
            
        # Create sequences
        # Return tensor shape: (num_sequences, seq_len, num_features)
        sequences = []
        for i in range(len(features_np) - self.seq_len + 1):
            seq = features_np[i : i + self.seq_len]
            sequences.append(seq)
            
        sequences_np = np.array(sequences)
        tensor = torch.tensor(sequences_np, dtype=torch.float32)
        
        return tensor

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.data.timeframe_manager import TimeframeManager
    from src.broker.mt5_client import MT5Client
    import yaml
    
    try:
        with open("config/settings.yaml", "r") as f:
            settings = yaml.safe_load(f)
        client = MT5Client(login=settings['broker']['login'], password=settings['broker']['password'], server=settings['broker']['server'])
        if client.connect():
            manager = TimeframeManager(client, settings['broker']['symbol'])
            if manager.load_from_csv():
                m5_data = manager.get_data("M5")
                builder = FeatureBuilder()
                tensor = builder.build_features(m5_data, fit_scaler=True)
                if tensor is not None:
                    print(f"Feature Tensor Shape: {tensor.shape}")
                    print(f"Number of features: {tensor.shape[2]}")
            client.disconnect()
    except Exception as e:
        print(f"Test failed: {e}")
