import pandas as pd
from src.ai.feature_builder import FeatureBuilder

# Create dummy df
df = pd.DataFrame({
    'open': [100.0] * 100,
    'high': [101.0] * 100,
    'low': [99.0] * 100,
    'close': [100.5] * 100,
    'tick_volume': [1000] * 100,
    'real_volume': [1000] * 100,
    'spread': [1] * 100
}, index=pd.date_range('2023-01-01', periods=100, freq='H'))

builder = FeatureBuilder()
feats = builder._compute_base_features(df)
numeric_feats = feats.select_dtypes(include=['number']).drop(columns=['pattern_dir_num'], errors='ignore')
print("Total numeric features:", len(numeric_feats.columns))
print(list(numeric_feats.columns))
