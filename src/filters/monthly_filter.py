import logging
import pandas as pd
import pandas_ta_classic as ta

logger = logging.getLogger(__name__)

class MonthlyFilter:
    """
    Filter based on Monthly trend.
    """
    
    def __init__(self):
        pass

    def evaluate(self, monthly_data: pd.DataFrame) -> str:
        """
        Evaluate the monthly trend using EMA 10.
        
        Args:
            monthly_data: Monthly OHLCV DataFrame
            
        Returns:
            str: "UP", "DOWN", or "SIDEWAYS"
        """
        if monthly_data is None or len(monthly_data) < 10:
            logger.warning("Not enough monthly data to compute EMA 10. Defaulting to SIDEWAYS.")
            return "SIDEWAYS"
            
        # Calculate EMA 10
        monthly_data['EMA_10'] = ta.ema(monthly_data['close'], length=10)
        
        # Get the latest closed candle (index -1)
        latest = monthly_data.iloc[-1]
        
        if pd.isna(latest['EMA_10']):
            return "SIDEWAYS"
            
        close_price = latest['close']
        ema_10 = latest['EMA_10']
        
        # Define a small threshold for sideways to prevent whipsaw
        threshold = ema_10 * 0.006 # 0.6% buffer
        
        if close_price > ema_10 + threshold:
            return "UP"
        elif close_price < ema_10 - threshold:
            return "DOWN"
        else:
            return "SIDEWAYS"

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
                monthly_data = manager.get_data("MN1")
                f = MonthlyFilter()
                trend = f.evaluate(monthly_data)
                print(f"Monthly Trend: {trend}")
            client.disconnect()
    except Exception as e:
        print(f"Test failed: {e}")
