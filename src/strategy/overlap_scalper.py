import logging
import pandas as pd
import numpy as np
from datetime import datetime
from src.strategy.base import BaseStrategy, Signal
from src.strategy.ai_strategy import AIStrategy

logger = logging.getLogger(__name__)

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

class OverlapScalper(BaseStrategy):
    """
    London/NY Overlap Scalping Strategy
    Time: 19:00-23:00 GMT+7
    Conditions: EMA9 crosses EMA21 + RSI 45-65 + Volume > 20-period avg
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
        
        if m5_data is None or len(m5_data) < 25:
            return Signal("HOLD", 0.0, reason="Not enough M5 data")

        last_time_utc = m5_data.index[-1]
        gmt7_time = last_time_utc + pd.Timedelta(hours=7)
        hour = gmt7_time.hour
        day_of_week = gmt7_time.weekday() # 0 = Monday, 6 = Sunday

        if not (19 <= hour < 23):
            return Signal("HOLD", 0.0, reason="Outside London/NY Overlap window")

        # Wednesday (2) and Thursday (3) are preferred, we can require slightly less confidence, 
        # but let's just use it as a weight
        weight = 1.0
        if day_of_week in [2, 3]:
            weight = 1.2
            
        recent_m5 = m5_data.iloc[-500:]
        current_candle = recent_m5.iloc[-1]
        current_price = current_candle['close']
        
        # Calculate Indicators
        ema9 = recent_m5['close'].ewm(span=9).mean()
        ema21 = recent_m5['close'].ewm(span=21).mean()
        rsi14 = calculate_rsi(recent_m5['close'], 14).iloc[-1]
        avg_vol20 = recent_m5['tick_volume'].rolling(20).mean().iloc[-1]
        
        # ATR 14
        high_low = recent_m5['high'] - recent_m5['low']
        high_close = np.abs(recent_m5['high'] - recent_m5['close'].shift())
        low_close = np.abs(recent_m5['low'] - recent_m5['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr_14 = tr.rolling(14).mean().iloc[-1]
        
        # Cross logic
        ema9_curr = ema9.iloc[-1]
        ema21_curr = ema21.iloc[-1]
        ema9_prev = ema9.iloc[-2]
        ema21_prev = ema21.iloc[-2]
        
        bullish_cross = ema9_curr > ema21_curr and ema9_prev <= ema21_prev
        bearish_cross = ema9_curr < ema21_curr and ema9_prev >= ema21_prev
        
        # Check volume
        if current_candle['tick_volume'] <= avg_vol20:
            return Signal("HOLD", 0.0, reason="Volume too low for overlap scalping")
            
        # Check RSI range (45-65 indicates momentum without being overextended)
        if not (45 <= rsi14 <= 65):
            return Signal("HOLD", 0.0, reason=f"RSI {rsi14:.1f} outside 45-65 bounds")

        # AI Direction
        ai_direction, ai_conf = self.ai_strategy.get_raw_prediction(h1_data)
        
        # Apply day of week weight
        adjusted_conf = min(1.0, ai_conf * weight)
        
        if bullish_cross and ai_direction == "BUY":
            sl = current_price - (atr_14 * 0.8)
            tp = current_price + (atr_14 * 1.5)
            return Signal("BUY", adjusted_conf, current_price, sl, tp, trailing_stop=True, reason="Overlap Bullish Scalp")
            
        elif bearish_cross and ai_direction == "SELL":
            sl = current_price + (atr_14 * 0.8)
            tp = current_price - (atr_14 * 1.5)
            return Signal("SELL", adjusted_conf, current_price, sl, tp, trailing_stop=True, reason="Overlap Bearish Scalp")

        return Signal("HOLD", 0.0, reason="No overlap scalp signal")
