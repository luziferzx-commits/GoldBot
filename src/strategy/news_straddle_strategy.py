import logging
from datetime import datetime, timedelta
import pandas as pd

from src.strategy.base import BaseStrategy, Signal
from src.calendar.economic_calendar import EconomicCalendar

logger = logging.getLogger(__name__)

class NewsStraddleStrategy(BaseStrategy):
    """
    Hunts high impact news events by placing BUY STOP and SELL STOP orders exactly 1 minute before the release.
    """
    
    def __init__(self, calendar: EconomicCalendar):
        self.calendar = calendar
        self.straddle_triggered_for = set()

    def generate_signal(
        self,
        m5_data: pd.DataFrame,
        m15_data: pd.DataFrame,
        h1_data: pd.DataFrame,
        daily_data: pd.DataFrame,
        monthly_data: pd.DataFrame
    ) -> Signal:
        
        if self.calendar.api_error or self.calendar.events_df.empty:
            return Signal("HOLD", 0.0, reason="No calendar data")
            
        current_time = datetime.utcnow()
        current_price = m5_data['close'].iloc[-1]
        
        # Look for High impact USD news
        high_impact = self.calendar.events_df[
            (self.calendar.events_df['impact'] == 'High') & 
            (self.calendar.events_df['country'] == 'USD')
        ]
        
        for _, row in high_impact.iterrows():
            news_time = row['date'].replace(tzinfo=None)
            
            # Check if we are exactly between 60s and 30s before the news
            time_diff = (news_time - current_time).total_seconds()
            
            if 30 <= time_diff <= 60:
                event_id = f"{row['title']}_{news_time.isoformat()}"
                
                if event_id not in self.straddle_triggered_for:
                    self.straddle_triggered_for.add(event_id)
                    logger.warning(f"🚨 NEWS STRADDLE TRIGGERED: {row['title']} in {time_diff:.0f} seconds!")
                    
                    # Return special STRADDLE signal
                    # We will use entry_price as the current price, and order_manager will handle offsets
                    return Signal("STRADDLE", 1.0, entry_price=current_price, reason=f"News Straddle: {row['title']}")
                    
        return Signal("HOLD", 0.0, reason="No imminent news")
