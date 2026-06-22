import logging
import pandas as pd
from datetime import datetime
import numpy as np
from src.strategy.base import BaseStrategy, Signal
from src.strategy.ai_strategy import AIStrategy

logger = logging.getLogger(__name__)

class SGEStrategy(BaseStrategy):
    """
    Shanghai Gold Exchange (SGE) Spike Strategy
    Time: 08:30-09:30 GMT+7
    Detects momentum spikes after SGE opens.
    """
    def __init__(self, ai_strategy: AIStrategy):
        self.ai_strategy = ai_strategy

    def generate_signal(
        self,
        m5_data: pd.DataFrame,
        m15_data: pd.DataFrame,
        h1_data: pd.DataFrame,
        daily_data: pd.DataFrame,
        monthly_data: pd.DataFrame
    ) -> Signal:
        
        if m5_data is None or len(m5_data) < 20:
            return Signal("HOLD", 0.0, reason="Not enough M5 data")

        # Get current time and convert UTC to GMT+7
        last_time_utc = m5_data.index[-1]
        gmt7_time = last_time_utc + pd.Timedelta(hours=7)
        hour = gmt7_time.hour
        minute = gmt7_time.minute
        
        # 08:30 - 09:30 GMT+7
        if hour == 8 and minute >= 30:
            pass
        elif hour == 9 and minute <= 30:
            pass
        else:
            return Signal("HOLD", 0.0, reason="Outside SGE window")

        # Get recent M5 data
        recent_m5 = m5_data.iloc[-500:]
        current_candle = recent_m5.iloc[-1]
        current_price = current_candle['close']
        
        # Calculate 20-period average volume
        avg_volume = recent_m5['tick_volume'].iloc[-20:].mean()
        
        # Calculate M5 ATR(14) for sizing
        high_low = recent_m5['high'] - recent_m5['low']
        high_close = np.abs(recent_m5['high'] - recent_m5['close'].shift())
        low_close = np.abs(recent_m5['low'] - recent_m5['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr_14 = tr.rolling(14).mean().iloc[-1]
        
        candle_size = abs(current_candle['close'] - current_candle['open'])
        
        # Check Spike Conditions
        if current_candle['tick_volume'] <= avg_volume * 1.5:
            return Signal("HOLD", 0.0, reason="Volume too low for SGE spike")
            
        if candle_size <= atr_14 * 0.8:
            return Signal("HOLD", 0.0, reason="Candle size too small for SGE spike")
            
        h1_trend = "SIDEWAYS"
        if len(h1_data) >= 50:
            ema20 = h1_data['close'].ewm(span=20).mean().iloc[-1]
            ema50 = h1_data['close'].ewm(span=50).mean().iloc[-1]
            if ema20 > ema50:
                h1_trend = "UP"
            elif ema20 < ema50:
                h1_trend = "DOWN"

        is_bullish = current_candle['close'] > current_candle['open']
        
        if is_bullish and h1_trend == "UP":
            sl = current_candle['low'] - 0.5
            tp = current_price + (atr_14 * 1.5)
            # Use AI strategy for confidence fallback
            _, conf = self.ai_strategy.get_raw_prediction(h1_data)
            return Signal("BUY", max(0.6, conf), current_price, sl, tp, reason="SGE Bullish Spike")
            
        elif not is_bullish and h1_trend == "DOWN":
            sl = current_candle['high'] + 0.5
            tp = current_price - (atr_14 * 1.5)
            _, conf = self.ai_strategy.get_raw_prediction(h1_data)
            return Signal("SELL", max(0.6, conf), current_price, sl, tp, reason="SGE Bearish Spike")

        return Signal("HOLD", 0.0, reason="SGE spike condition unmet")
