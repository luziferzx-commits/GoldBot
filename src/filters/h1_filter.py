import logging
import pandas as pd
import pandas_ta_classic as ta

logger = logging.getLogger(__name__)

class H1Filter:
    """
    Filter based on 1-Hour trend.
    """
    
    def __init__(self):
        pass

    def evaluate(self, h1_data: pd.DataFrame) -> str:
        """
        Evaluate H1 trend using EMA 50 and EMA 200.
        
        Args:
            h1_data: H1 OHLCV DataFrame
            
        Returns:
            str: "UP", "DOWN", or "SIDEWAYS"
        """
        if h1_data is None or len(h1_data) < 200:
            logger.warning("Not enough H1 data. Defaulting to SIDEWAYS.")
            return "SIDEWAYS"
            
        h1_data['EMA_50'] = ta.ema(h1_data['close'], length=50)
        h1_data['EMA_200'] = ta.ema(h1_data['close'], length=200)
        
        latest = h1_data.iloc[-1]
        
        if pd.isna(latest['EMA_200']):
            return "SIDEWAYS"
            
        close = latest['close']
        ema_50 = latest['EMA_50']
        ema_200 = latest['EMA_200']
        
        if close > ema_50 and ema_50 > ema_200:
            return "UP"
        elif close < ema_50 and ema_50 < ema_200:
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
                h1_data = manager.get_data("H1")
                f = H1Filter()
                trend = f.evaluate(h1_data)
                print(f"H1 Trend: {trend}")
            client.disconnect()
    except Exception as e:
        print(f"Test failed: {e}")
