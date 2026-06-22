import logging
import pandas as pd
from datetime import datetime
from src.strategy.base import BaseStrategy, Signal
from src.strategy.ai_strategy import AIStrategy

logger = logging.getLogger(__name__)

class PO3Strategy(BaseStrategy):
    """
    Power of 3 Strategy (Accumulation / Manipulation / Distribution)
    - Phase 1 Accumulation: 02:00-09:00 GMT+7 (Asian Range)
    - Phase 2 Manipulation: 13:00-15:00 GMT+7 (Fakeout)
    - Phase 3 Distribution: 15:00-23:00 GMT+7 (Entry)
    """
    def __init__(self, ai_strategy: AIStrategy):
        self.ai_strategy = ai_strategy
        self.active_date = None
        self.asian_high = 0.0
        self.asian_low = 99999.0
        self.manip_high = 0.0
        self.manip_low = 99999.0
        self.bias = "NEUTRAL"

    def _reset_daily(self, current_date):
        if self.active_date != current_date:
            self.active_date = current_date
            self.asian_high = 0.0
            self.asian_low = 99999.0
            self.manip_high = 0.0
            self.manip_low = 99999.0
            self.bias = "NEUTRAL"

    def _update_accumulation(self, df_m5: pd.DataFrame, current_date):
        """Update Asian High/Low (02:00-09:00 GMT+7)"""
        recent = df_m5.iloc[-288:]
        today_data = recent[recent.index.date == current_date]
        if today_data.empty: return
        
        # Filter for 02:00 to 08:59 GMT+7
        # First convert index to GMT+7 to check easily
        gmt7_times = today_data.index + pd.Timedelta(hours=7)
        mask = (gmt7_times.hour >= 2) & (gmt7_times.hour < 9)
        asian_session = today_data[mask]
        
        if not asian_session.empty:
            self.asian_high = asian_session['high'].max()
            self.asian_low = asian_session['low'].min()

    def _update_manipulation(self, df_m5: pd.DataFrame, current_date):
        """Update Manipulation Phase and Bias (13:00-15:00 GMT+7)"""
        recent = df_m5.iloc[-288:]
        today_data = recent[recent.index.date == current_date]
        if today_data.empty: return
        
        gmt7_times = today_data.index + pd.Timedelta(hours=7)
        mask = (gmt7_times.hour >= 13) & (gmt7_times.hour < 15)
        manip_session = today_data[mask]
        
        if not manip_session.empty:
            self.manip_high = manip_session['high'].max()
            self.manip_low = manip_session['low'].min()
            
            # Check for fakeouts (sweeps)
            if self.asian_high > 0 and self.asian_low < 99999:
                if self.manip_high > self.asian_high:
                    self.bias = "SELL" # Swept high -> expect distribution down
                elif self.manip_low < self.asian_low:
                    self.bias = "BUY"  # Swept low -> expect distribution up

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

        last_time_utc = m5_data.index[-1]
        current_date_utc = last_time_utc.date()
        
        gmt7_time = last_time_utc + pd.Timedelta(hours=7)
        hour = gmt7_time.hour
        
        self._reset_daily(current_date_utc)
        self._update_accumulation(m5_data, current_date_utc)
        self._update_manipulation(m5_data, current_date_utc)

        # We only enter in the Distribution phase (15:00-23:00 GMT+7)
        if not (15 <= hour < 23):
            return Signal("HOLD", 0.0, reason="Outside PO3 Distribution window")
            
        if self.bias == "NEUTRAL":
            return Signal("HOLD", 0.0, reason="No manipulation fakeout detected")
            
        current_price = m5_data['close'].iloc[-1]
        manip_range = self.manip_high - self.manip_low
        if manip_range <= 0:
            manip_range = 2.0 # fallback

        ai_direction, ai_conf = self.ai_strategy.get_raw_prediction(h1_data)
        
        if ai_conf < 0.55:
            return Signal("HOLD", 0.0, reason="AI Confidence too low")
            
        if self.bias == "BUY" and ai_direction == "BUY":
            sl = self.manip_low - 0.5
            tp = current_price + (manip_range * 1.618)
            return Signal("BUY", ai_conf, current_price, sl, tp, reason="PO3 Bullish Distribution")
            
        elif self.bias == "SELL" and ai_direction == "SELL":
            sl = self.manip_high + 0.5
            tp = current_price - (manip_range * 1.618)
            return Signal("SELL", ai_conf, current_price, sl, tp, reason="PO3 Bearish Distribution")
            
        return Signal("HOLD", 0.0, reason="AI direction doesn't match PO3 bias")
