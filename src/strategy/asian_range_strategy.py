import logging
import pandas as pd
from datetime import datetime
from src.strategy.base import BaseStrategy, Signal
from src.strategy.ai_strategy import AIStrategy

logger = logging.getLogger(__name__)

class AsianRangeStrategy(BaseStrategy):
    """
    Asian Range Breakout Strategy
    Accumulates range from 02:00 to 09:00 (GMT+7).
    Executes trades on breakout during London Open 15:00-16:00 (GMT+7).
    """

    def __init__(self, ai_strategy: AIStrategy):
        self.ai_strategy = ai_strategy
        self.asian_high = 0.0
        self.asian_low = 99999.0
        self.range_active_date = None

    def _update_range(self, m5_data: pd.DataFrame):
        """Finds Asian High/Low for the current day."""
        if m5_data is None or m5_data.empty:
            return

        last_time = m5_data.index[-1]
        current_date = last_time.date()
        
        # Reset range if new day
        if self.range_active_date != current_date:
            self.asian_high = 0.0
            self.asian_low = 99999.0
            self.range_active_date = current_date

        # Filter today's M5 data between 02:00 and 09:00
        recent = m5_data.iloc[-288:]
        today_data = recent[recent.index.date == current_date]
        asian_session = today_data[(today_data.index.hour >= 2) & (today_data.index.hour < 9)]
        
        if not asian_session.empty:
            self.asian_high = asian_session['high'].max()
            self.asian_low = asian_session['low'].min()

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

        last_time = m5_data.index[-1]
        hour = last_time.hour
        
        # 1. Update the range
        self._update_range(m5_data)

        # 2. Check if we are in the London Open breakout window (15:00 - 16:59)
        if hour not in [15, 16]:
            return Signal("HOLD", 0.0, reason="Outside London Open window")
            
        if self.asian_high == 0.0 or self.asian_low == 99999.0:
            return Signal("HOLD", 0.0, reason="Asian Range not formed")

        current_price = m5_data['close'].iloc[-1]
        
        # 3. Get AI Direction for confirmation
        ai_direction, ai_conf = self.ai_strategy.get_raw_prediction(h1_data)
        if ai_conf < 0.55:
            return Signal("HOLD", 0.0, reason="AI Confidence too low")

        range_size = self.asian_high - self.asian_low

        # 4. Check for breakout
        if current_price > self.asian_high and ai_direction == "BUY":
            sl = self.asian_low - 2.0  # SL below Asian Low
            tp = current_price + (range_size * 1.5)
            return Signal("BUY", ai_conf, current_price, sl, tp, reason="Asian Range Bullish Breakout")

        if current_price < self.asian_low and ai_direction == "SELL":
            sl = self.asian_high + 2.0 # SL above Asian High
            tp = current_price - (range_size * 1.5)
            return Signal("SELL", ai_conf, current_price, sl, tp, reason="Asian Range Bearish Breakout")

        return Signal("HOLD", 0.0, reason="No Breakout")
