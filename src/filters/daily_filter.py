import logging
import pandas as pd
from typing import Tuple

logger = logging.getLogger(__name__)

class DailyFilter:
    """
    Filter based on Daily context, determining daily bias and ADR limits.
    """
    
    def __init__(self):
        pass

    def evaluate(self, daily_data: pd.DataFrame, current_price: float) -> Tuple[str, float]:
        """
        Evaluate daily bias and Average Daily Range usage.
        
        Args:
            daily_data: Daily OHLCV DataFrame
            current_price: The current price (to calculate today's range)
            
        Returns:
            Tuple[str, float]: (bias "BUY"|"SELL"|"NEUTRAL", adr_used_pct 0.0-1.0+)
        """
        if daily_data is None or len(daily_data) < 14:
            logger.warning("Not enough daily data. Defaulting to NEUTRAL bias.")
            return "NEUTRAL", 0.0
            
        # Calculate ADR (Average Daily Range) over 14 days
        daily_data['range'] = daily_data['high'] - daily_data['low']
        adr = daily_data['range'].rolling(window=14).mean().iloc[-2] # Use previous days' ADR
        
        # Determine bias based on the previous day's close vs open
        prev_day = daily_data.iloc[-2] # -1 is the current ongoing day
        bias = "NEUTRAL"
        if prev_day['close'] > prev_day['open']:
            bias = "BUY"
        elif prev_day['close'] < prev_day['open']:
            bias = "SELL"
            
        # Calculate today's range usage
        today = daily_data.iloc[-1]
        today_range_so_far = max(today['high'], current_price) - min(today['low'], current_price)
        
        adr_used_pct = 0.0
        if adr > 0 and not pd.isna(adr):
            adr_used_pct = today_range_so_far / adr
            
        return bias, adr_used_pct

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    from src.data.timeframe_manager import TimeframeManager
    from src.broker.mt5_client import MT5Client
    import yaml
    
    try:
        with open("config/settings.yaml", "r") as f:
            settings = yaml.safe_load(f)
        client = MT5Client(login=settings['broker']['login'], password=settings['broker']['password'], server=settings['broker']['server'])
        if client.connect():
            manager = TimeframeManager(client, settings['broker']['symbol'])
            if manager.load_from_csv():
                daily_data = manager.get_data("D1")
                current_price = client.get_current_price(settings['broker']['symbol'])['bid']
                f = DailyFilter()
                bias, adr_pct = f.evaluate(daily_data, current_price)
                print(f"Daily Bias: {bias}, ADR Used: {adr_pct:.2%}")
            client.disconnect()
    except Exception as e:
        print(f"Test failed: {e}")
