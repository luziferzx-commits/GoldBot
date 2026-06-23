import os
import yaml
import logging
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger("DataFetcher")

from src.broker.mt5_client import MT5Client
from src.data.timeframe_manager import TimeframeManager

def fetch_history():
    with open("config/settings.yaml", "r") as f:
        settings = yaml.safe_load(f)
        
    login_val = os.getenv('MT5_LOGIN', settings['broker'].get('login'))
    login = int(login_val) if login_val else 0
    password = os.getenv('MT5_PASSWORD', settings['broker'].get('password', ''))
    server = os.getenv('MT5_SERVER', settings['broker'].get('server', ''))
    
    client = MT5Client(login=login, password=password, server=server)
    if not client.connect():
        logger.error("Failed to connect to MT5")
        return
        
    symbol = settings['broker'].get('symbols', [settings['broker'].get('symbol', 'XAUUSDm')])[0]
    manager = TimeframeManager(client, symbol)
    
    # We fetch a huge amount of bars (e.g. 500,000) for comprehensive backtesting
    logger.info("Fetching up to 500,000 bars from MT5...")
    manager.fetch_all(count=500000)
    
    logger.info("Saving to CSV...")
    manager.save_to_csv()
    logger.info("Done.")

if __name__ == "__main__":
    fetch_history()
