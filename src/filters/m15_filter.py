import logging
import pandas as pd
import pandas_ta_classic as ta
from typing import Tuple

logger = logging.getLogger(__name__)

class M15Filter:
    """
    Filter based on 15-Minute trend and momentum to confirm entry.
    """
    
    def __init__(self):
        pass

    def evaluate(self, m15_data: pd.DataFrame, direction: str) -> Tuple[bool, float]:
        """
        Evaluate M15 confirmation using RSI and EMA.
        
        Args:
            m15_data: M15 OHLCV DataFrame
            direction: "BUY" or "SELL"
            
        Returns:
            Tuple[bool, float]: (confirmed, strength 0.0-1.0)
        """
        if m15_data is None or len(m15_data) < 21:
            logger.warning("Not enough M15 data.")
            return False, 0.0
            
        m15_data['RSI_14'] = ta.rsi(m15_data['close'], length=14)
        m15_data['EMA_21'] = ta.ema(m15_data['close'], length=21)
        
        latest = m15_data.iloc[-1]
        
        if pd.isna(latest['RSI_14']) or pd.isna(latest['EMA_21']):
            return False, 0.0
            
        close = latest['close']
        rsi = latest['RSI_14']
        ema_21 = latest['EMA_21']
        
        confirmed = False
        strength = 0.0
        
        if direction == "BUY":
            # Price above EMA and RSI not overbought
            if close > ema_21 and 40 <= rsi <= 70:
                confirmed = True
                strength = (rsi - 40) / 30.0 # simple strength proxy
        elif direction == "SELL":
            # Price below EMA and RSI not oversold
            if close < ema_21 and 30 <= rsi <= 60:
                confirmed = True
                strength = (60 - rsi) / 30.0
                
        # Clamp strength between 0 and 1
        strength = max(0.0, min(1.0, strength))
        
        return confirmed, strength

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
                m15_data = manager.get_data("M15")
                f = M15Filter()
                confirmed, strength = f.evaluate(m15_data, "BUY")
                print(f"M15 BUY Confirm: {confirmed}, Strength: {strength:.2f}")
            client.disconnect()
    except Exception as e:
        print(f"Test failed: {e}")
