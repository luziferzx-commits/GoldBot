import logging
import pandas as pd
import pandas_ta_classic as ta
from typing import Optional, Tuple
from src.strategy.base import BaseStrategy, Signal
from src.filters.monthly_filter import MonthlyFilter
from src.filters.daily_filter import DailyFilter
from src.filters.h1_filter import H1Filter
from src.filters.m15_filter import M15Filter
from src.calendar.economic_calendar import EconomicCalendar

logger = logging.getLogger(__name__)

class TrendFollowStrategy(BaseStrategy):
    """
    Trend following strategy using EMA crossovers and multi-timeframe filters.
    """
    
    def __init__(self):
        self.monthly_filter = MonthlyFilter()
        self.daily_filter = DailyFilter()
        self.h1_filter = H1Filter()
        self.m15_filter = M15Filter()
        self.calendar = EconomicCalendar()
        # Pre-fetch news for the day
        self.calendar.fetch_news()

    def _get_m5_signal(self, m5_data: pd.DataFrame) -> Tuple[str, float]:
        """
        Evaluate M5 entry signal (EMA 9/21 cross + RSI).
        Returns direction and confidence.
        """
        if m5_data is None or len(m5_data) < 21:
            return "HOLD", 0.0
            
        m5_data['EMA_9'] = ta.ema(m5_data['close'], length=9)
        m5_data['EMA_21'] = ta.ema(m5_data['close'], length=21)
        m5_data['RSI_14'] = ta.rsi(m5_data['close'], length=14)
        
        # We need the last two closed candles to detect a crossover
        prev = m5_data.iloc[-2]
        curr = m5_data.iloc[-1]
        
        if pd.isna(curr['EMA_21']) or pd.isna(curr['RSI_14']):
            return "HOLD", 0.0
            
        rsi = curr['RSI_14']
        
        # Bullish Crossover: EMA9 crosses above EMA21
        if prev['EMA_9'] <= prev['EMA_21'] and curr['EMA_9'] > curr['EMA_21']:
            if 35 <= rsi <= 70:
                return "BUY", 0.70 # Baseline confidence
                
        # Bearish Crossover: EMA9 crosses below EMA21
        elif prev['EMA_9'] >= prev['EMA_21'] and curr['EMA_9'] < curr['EMA_21']:
            if 30 <= rsi <= 65:
                return "SELL", 0.70
                
        return "HOLD", 0.0

    def generate_signal(
        self,
        m5_data: pd.DataFrame,
        m15_data: pd.DataFrame,
        h1_data: pd.DataFrame,
        daily_data: pd.DataFrame,
        monthly_data: pd.DataFrame
    ) -> Signal:
        
        current_price = m5_data.iloc[-1]['close'] if m5_data is not None else 0.0
        
        # 1. Economic Calendar Filter
        if self.calendar.is_news_time():
            return Signal("HOLD", 0.0, reason="High Impact News Window")
            
        # 2. Monthly Filter
        monthly_trend = self.monthly_filter.evaluate(monthly_data)
        
        # 3. Daily Bias & ADR
        daily_bias, adr_pct = self.daily_filter.evaluate(daily_data, current_price)
        if adr_pct > 0.90:
            return Signal("HOLD", 0.0, reason=f"ADR limit reached ({adr_pct:.1%})")
            
        # 4. H1 Trend Filter
        h1_trend = self.h1_filter.evaluate(h1_data)
        
        # 5. M5 Entry Signal
        m5_direction, m5_conf = self._get_m5_signal(m5_data)
        if m5_direction == "HOLD":
            return Signal("HOLD", 0.0, reason="No M5 entry signal")
            
        # Alignment Checks
        if monthly_trend != "SIDEWAYS" and m5_direction != monthly_trend:
            return Signal("HOLD", 0.0, reason=f"M5 {m5_direction} against Monthly {monthly_trend}")
            
        if daily_bias != "NEUTRAL" and m5_direction != daily_bias:
            return Signal("HOLD", 0.0, reason=f"M5 {m5_direction} against Daily Bias {daily_bias}")
            
        # 6. M15 Confirmation
        m15_confirmed, m15_strength = self.m15_filter.evaluate(m15_data, m5_direction)
        
        # H1 Override logic
        if h1_trend == "SIDEWAYS":
            if m15_strength <= 0.6:
                return Signal("HOLD", 0.0, reason=f"H1 SIDEWAYS and M15 strength ({m15_strength:.2f}) too low")
        elif m5_direction != h1_trend:
            return Signal("HOLD", 0.0, reason=f"M5 {m5_direction} against H1 {h1_trend}")
            
        if not m15_confirmed or m15_strength <= 0.5:
            return Signal("HOLD", 0.0, reason="M15 Pattern not confirmed or weak")
            
        # Compute final confidence
        final_conf = (m5_conf + m15_strength) / 2.0
        
        return Signal(
            direction=m5_direction,
            confidence=final_conf,
            entry_price=current_price,
            reason=f"Passed all filters. Monthly:{monthly_trend}, Daily:{daily_bias}, H1:{h1_trend}"
        )

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import yaml
    from src.data.timeframe_manager import TimeframeManager
    from src.broker.mt5_client import MT5Client
    
    try:
        with open("config/settings.yaml", "r") as f:
            settings = yaml.safe_load(f)
        client = MT5Client(login=settings['broker']['login'], password=settings['broker']['password'], server=settings['broker']['server'])
        if client.connect():
            manager = TimeframeManager(client, settings['broker']['symbol'])
            if manager.load_from_csv():
                strategy = TrendFollowStrategy()
                signal = strategy.generate_signal(
                    m5_data=manager.get_data("M5"),
                    m15_data=manager.get_data("M15"),
                    h1_data=manager.get_data("H1"),
                    daily_data=manager.get_data("D1"),
                    monthly_data=manager.get_data("MN1")
                )
                print(f"Final Signal: {signal}")
            client.disconnect()
    except Exception as e:
        print(f"Test failed: {e}")
