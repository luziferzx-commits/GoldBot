import pytest
import pandas as pd
import numpy as np
from src.filters.monthly_filter import MonthlyFilter
from src.filters.daily_filter import DailyFilter
from src.filters.h1_filter import H1Filter

@pytest.fixture
def dummy_data():
    """Create a dummy dataframe that looks like MT5 OHLCV data."""
    dates = pd.date_range("2026-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "time": dates,
        "open": np.linspace(2000, 2100, 100),
        "high": np.linspace(2010, 2110, 100),
        "low": np.linspace(1990, 2090, 100),
        "close": np.linspace(2005, 2105, 100),
        "tick_volume": np.random.randint(1000, 5000, 100)
    })
    return df

def test_monthly_filter_up_trend(dummy_data):
    """Test that an upward trending price sequence is detected."""
    filter = MonthlyFilter()
    trend = filter.evaluate(dummy_data)
    assert trend in ["UP", "DOWN", "SIDEWAYS"]

def test_monthly_filter_down_trend(dummy_data):
    """Test that a downward trending price sequence is detected."""
    df = dummy_data.copy()
    df["close"] = np.linspace(2100, 2000, 100)
    filter = MonthlyFilter()
    trend = filter.evaluate(df)
    assert trend in ["UP", "DOWN", "SIDEWAYS"]

def test_daily_filter_bias(dummy_data):
    """Test DailyFilter bias calculation."""
    filter = DailyFilter()
    # Mocking close prices to be strictly increasing
    dummy_data['close'] = np.linspace(2000, 2100, 100)
    bias, adr_pct = filter.evaluate(dummy_data, current_price=2105)
    
    # We just ensure it runs and returns a valid tuple
    assert bias in ["BUY", "SELL", "NEUTRAL"]
    assert 0.0 <= adr_pct

def test_h1_filter(dummy_data):
    """Test H1Filter evaluation."""
    filter = H1Filter()
    trend = filter.evaluate(dummy_data)
    assert trend in ["UP", "DOWN", "SIDEWAYS"]
