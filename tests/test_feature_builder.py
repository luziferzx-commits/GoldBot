import pytest
import pandas as pd
import numpy as np
import torch
from src.ai.feature_builder import FeatureBuilder

@pytest.fixture
def dummy_data():
    """Create dummy dataframe for feature extraction."""
    dates = pd.date_range("2026-01-01", periods=200, freq="h")
    df = pd.DataFrame({
        "time": dates,
        "open": np.random.uniform(2000, 2100, 200),
        "high": np.random.uniform(2000, 2100, 200),
        "low": np.random.uniform(2000, 2100, 200),
        "close": np.random.uniform(2000, 2100, 200),
        "tick_volume": np.random.randint(1000, 5000, 200)
    })
    
    df.set_index("time", inplace=True)
    
    # Ensure high >= open/close/low and low <= open/close/high
    df['high'] = df[['open', 'close', 'high']].max(axis=1) + 1
    df['low'] = df[['open', 'close', 'low']].min(axis=1) - 1
    
    # Add required external features to simulate the real dataframe
    df['sentiment_score'] = 0.5
    df['gold_bias'] = 0.1
    df['vix_level'] = 15.0
    df['dxy_level'] = 104.0
    df['bond_yield'] = 4.2
    
    return df

def test_feature_builder_shape(dummy_data):
    """Test that FeatureBuilder returns the correct tensor shape (Batch, Seq, Features)."""
    builder = FeatureBuilder(seq_len=60)
    
    features = builder.build_features(dummy_data)
    
    # Check if features is a torch.Tensor
    assert isinstance(features, torch.Tensor)
    
    # Check shape: Batch should be 200 - 60 + 1 = 141
    # seq_len = 60
    # Expected features = 42
    assert features.shape == (141, 60, 42)

def test_feature_builder_insufficient_data(dummy_data):
    """Test behavior when data length is less than seq_len."""
    builder = FeatureBuilder(seq_len=60)
    
    # Provide only 50 rows, while seq_len=60
    short_df = dummy_data.iloc[-50:]
    features = builder.build_features(short_df)
    
    assert features is None
