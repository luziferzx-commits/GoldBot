from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import pandas as pd

@dataclass
class Signal:
    """
    Data class representing a trading signal.
    """
    direction: str  # "BUY", "SELL", or "HOLD"
    confidence: float  # 0.0 to 1.0
    entry_price: Optional[float] = None
    sl_price: Optional[float] = None
    tp_price: Optional[float] = None
    trailing_stop: bool = False
    reason: str = ""

class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    """
    
    @abstractmethod
    def generate_signal(
        self,
        m5_data: pd.DataFrame,
        m15_data: pd.DataFrame,
        h1_data: pd.DataFrame,
        daily_data: pd.DataFrame,
        monthly_data: pd.DataFrame
    ) -> Signal:
        """
        Generate a trading signal based on multi-timeframe data.
        
        Args:
            m5_data: 5-minute OHLCV data
            m15_data: 15-minute OHLCV data
            h1_data: 1-hour OHLCV data
            daily_data: Daily OHLCV data
            monthly_data: Monthly OHLCV data
            
        Returns:
            Signal object
        """
        pass
