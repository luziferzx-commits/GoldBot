import pytest
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from src.ai.model import GoldLSTM
from src.ai.online_learner import OnlineLearner
import os
from pathlib import Path

@pytest.fixture
def mock_model():
    """Create a dummy small model for testing."""
    # We match the feature shape from FeatureBuilder
    model = GoldLSTM(input_size=42, hidden_size=16, num_layers=1)
    return model

@pytest.fixture
def dummy_history():
    """Create dummy dataframe for history."""
    dates = pd.date_range("2026-01-01", periods=200, freq="h")
    df = pd.DataFrame({
        "time": dates,
        "open": np.random.uniform(2000, 2100, 200),
        "high": np.random.uniform(2000, 2100, 200),
        "low": np.random.uniform(2000, 2100, 200),
        "close": np.random.uniform(2000, 2100, 200),
        "tick_volume": np.random.randint(1000, 5000, 200)
    })
    
    # Ensure high >= open/close/low and low <= open/close/high
    df['high'] = df[['open', 'close', 'high']].max(axis=1) + 1
    df['low'] = df[['open', 'close', 'low']].min(axis=1) - 1
    
    # Add external features
    df['sentiment_score'] = 0.5
    df['gold_bias'] = 0.1
    df['vix_level'] = 15.0
    df['dxy_level'] = 104.0
    df['bond_yield'] = 4.2
    
    df.set_index("time", inplace=True)
    return df

def test_online_learner_rollback_consecutive_losses(mock_model, dummy_history):
    """Test that 3 consecutive losses triggers a rollback."""
    learner = OnlineLearner(model=mock_model)
    # Use a dummy path for tests
    learner.model_path = Path("tests/test_model.pt")
    learner.checkpoint_path = Path("tests/test_model_checkpoint.pt")
    
    # Manually save a checkpoint to simulate a known-good state
    learner.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mock_model.state_dict(), learner.model_path)
    learner.save_checkpoint()
    
    # Mock some trades
    trade_time = dummy_history.index[-1]
    
    # Win 2 trades to get good stats and fill window
    trade_win = {"entry_time": trade_time, "direction": "BUY", "net_pnl": 50.0}
    learner.update(trade_win, dummy_history)
    learner.update(trade_win, dummy_history)
    assert learner.consecutive_losses == 0
    
    # Lose 3 trades
    trade_loss = {"entry_time": trade_time, "direction": "BUY", "net_pnl": -20.0}
    learner.update(trade_loss, dummy_history)
    learner.update(trade_loss, dummy_history)
    learner.update(trade_loss, dummy_history)
    
    # The 3rd loss should have triggered the rollback and cleared the window
    assert learner.consecutive_losses == 0
    assert len(learner.performance_window) == 0

    # Clean up test files
    if learner.model_path.exists():
        os.remove(learner.model_path)
    if learner.checkpoint_path.exists():
        os.remove(learner.checkpoint_path)

def test_online_learner_rollback_low_win_rate(mock_model, dummy_history):
    """Test that low win rate (<40%) triggers a rollback."""
    learner = OnlineLearner(model=mock_model)
    learner.model_path = Path("tests/test_model2.pt")
    learner.checkpoint_path = Path("tests/test_model_checkpoint2.pt")
    
    learner.model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(mock_model.state_dict(), learner.model_path)
    learner.save_checkpoint()
    
    trade_time = dummy_history.index[-1]
    trade_win = {"entry_time": trade_time, "direction": "BUY", "net_pnl": 50.0}
    trade_loss = {"entry_time": trade_time, "direction": "BUY", "net_pnl": -20.0}
    
    # Window of 5 trades to trigger evaluation: 1 Win, 4 Losses (not consecutive)
    # So we do: W, L, W, L, L, L -> Wait, if we do L, L, L it triggers consecutive.
    # We want to trigger < 40% without hitting consecutive 3.
    # Let's do: W, L, L, W, L, L -> 6 trades, 2 Wins, 4 Losses = 33% win rate
    
    learner.update(trade_win, dummy_history)
    learner.update(trade_loss, dummy_history)
    learner.update(trade_loss, dummy_history)
    learner.update(trade_win, dummy_history)
    learner.update(trade_loss, dummy_history)
    
    # At 5 trades: 2 Wins, 3 Losses (40%) -> exactly 0.40, won't trigger if strictly < 0.40.
    assert len(learner.performance_window) == 5
    
    # 6th trade: Loss -> 2 Wins, 4 Losses -> 33.3% < 40%. Rollback!
    learner.update(trade_loss, dummy_history)
    
    assert learner.consecutive_losses == 0
    assert len(learner.performance_window) == 0
    
    # Clean up test files
    if learner.model_path.exists():
        os.remove(learner.model_path)
    if learner.checkpoint_path.exists():
        os.remove(learner.checkpoint_path)
