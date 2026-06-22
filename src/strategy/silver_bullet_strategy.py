import logging
from datetime import datetime
import pandas as pd

from src.strategy.base import BaseStrategy, Signal
from src.analysis.ict_concepts import ICTConcepts
from src.strategy.ai_strategy import AIStrategy

logger = logging.getLogger(__name__)

class SilverBulletStrategy(BaseStrategy):
    """
    ICT Silver Bullet Strategy
    Operates specifically during:
    - London Silver Bullet: 17:00-18:00 (GMT+7)
    - NY AM Silver Bullet: 21:00-22:00 (GMT+7)
    - NY PM Silver Bullet: 01:00-02:00 (GMT+7)
    """

    def __init__(self, ai_strategy: AIStrategy):
        self.ai_strategy = ai_strategy
        
        # Define Silver Bullet windows in GMT+7 (Hour, Minute range 0-59)
        # We will check if the current hour matches these.
        self.sb_windows = [
            (17, 18), # London: 17:00-18:00
            (21, 22), # NY AM: 21:00-22:00
            (1, 2)    # NY PM: 01:00-02:00
        ]

    def _is_silver_bullet_time(self, current_time: datetime) -> bool:
        """Check if current time is inside a Silver Bullet window"""
        # In a real environment, MT5/server time might not be GMT+7, 
        # so this assumes the passed datetime object is already GMT+7.
        hour = current_time.hour
        # If hour is exactly the start hour of any window
        # Note: 17:00-18:00 means hour 17 (17:00:00 to 17:59:59)
        for (start_h, end_h) in self.sb_windows:
            if hour == start_h:
                return True
        return False

    def generate_signal(
        self,
        m5_data: pd.DataFrame,
        m15_data: pd.DataFrame,
        h1_data: pd.DataFrame,
        daily_data: pd.DataFrame,
        monthly_data: pd.DataFrame
    ) -> Signal:
        
        if m5_data is None or len(m5_data) < 50:
            return Signal("HOLD", 0.0, reason="Not enough M5 data")

        # 1. Time Window Check
        # We assume the dataframe index is localized to GMT+7 or naive but representing GMT+7
        last_time = m5_data.index[-1]
        if not self._is_silver_bullet_time(last_time):
            return Signal("HOLD", 0.0, reason="Outside Silver Bullet Window")

        # 2. Get AI Direction
        ai_direction, ai_conf = self.ai_strategy.get_raw_prediction(h1_data)
        if ai_conf < 0.55 or ai_direction not in ["BUY", "SELL"]:
            return Signal("HOLD", 0.0, reason="AI Confidence too low for Silver Bullet")

        # Detect ICT Concepts on M5
        # 1. Sweep
        df_ict = ICTConcepts.detect_liquidity_sweep(m5_data, swing_lookback=20)
        # 2. MSS
        df_ict = ICTConcepts.detect_mss(df_ict, bars_after_sweep=5)
        # 3. FVG
        df_ict = ICTConcepts.detect_fvg(df_ict, lookback=50)

        latest = df_ict.iloc[-1]
        current_price = latest['close']

        # We look back slightly to see if the setup formed recently (e.g., within the last 3-5 bars)
        # For a robust bot, we check if FVG exists and MSS triggered recently.
        # Let's simplify: check if there's an active FVG in the direction of the AI, 
        # and a recent MSS in the same direction.
        
        # Bullish Setup
        if ai_direction == "BUY":
            bullish_fvgs = df_ict[df_ict['bullish_fvg'] == True]
            if len(bullish_fvgs) == 0:
                return Signal("HOLD", 0.0, reason="No Bullish FVG")
                
            recent_fvg = bullish_fvgs.iloc[-1]
            # Need recent Bullish MSS (in last 20 bars to cover the 1-hour window if backtesting on H1)
            recent_mss = df_ict['bullish_mss'].iloc[-20:].any()
            if recent_mss:
                # Limit entry at the FVG top (Premium)
                entry_price = recent_fvg['fvg_top']
                # Find Sweep Low to put SL
                recent_sweeps = df_ict[df_ict['sweep_low'] == True]
                sl = recent_sweeps['sweep_price'].iloc[-1] if len(recent_sweeps) > 0 else recent_fvg['fvg_bottom'] - 2.0
                tp = entry_price + ((entry_price - sl) * 2) # 1:2 RR
                return Signal("BUY", ai_conf, entry_price=entry_price, sl_price=sl, tp_price=tp, reason="Silver Bullet Bullish Limit")

        # Bearish Setup
        elif ai_direction == "SELL":
            bearish_fvgs = df_ict[df_ict['bearish_fvg'] == True]
            if len(bearish_fvgs) == 0:
                return Signal("HOLD", 0.0, reason="No Bearish FVG")
                
            recent_fvg = bearish_fvgs.iloc[-1]
            # Need recent Bearish MSS
            recent_mss = df_ict['bearish_mss'].iloc[-20:].any()
            if recent_mss:
                # Limit entry at the FVG bottom (Discount)
                entry_price = recent_fvg['fvg_bottom']
                recent_sweeps = df_ict[df_ict['sweep_high'] == True]
                sl = recent_sweeps['sweep_price'].iloc[-1] if len(recent_sweeps) > 0 else recent_fvg['fvg_top'] + 2.0
                tp = entry_price - ((sl - entry_price) * 2) # 1:2 RR
                return Signal("SELL", ai_conf, entry_price=entry_price, sl_price=sl, tp_price=tp, reason="Silver Bullet Bearish Limit")

        return Signal("HOLD", 0.0, reason="Silver Bullet conditions not met")
