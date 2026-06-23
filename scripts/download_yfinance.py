import yfinance as yf
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def download_m5_chunked():
    """
    yfinance gives max 60 days for M5.
    We chunk 55 days at a time backwards.
    """
    Path("data/historical/M5").mkdir(parents=True, exist_ok=True)

    ticker = yf.Ticker("GC=F")
    all_data = []

    end = datetime.now()
    for i in range(4):
        start = end - timedelta(days=55)
        logger.info(f"Downloading M5: {start.date()} to {end.date()}")
        df = ticker.history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="5m"
        )
        if len(df) > 0:
            all_data.append(df)
            logger.info(f"Got {len(df)} bars")
        end = start
        time.sleep(2)

    if all_data:
        combined = pd.concat(all_data)
        combined = combined[~combined.index.duplicated()]
        combined = combined.sort_index()
        combined.columns = [c.lower() for c in combined.columns]
        combined.rename(columns={'volume': 'tick_volume'}, inplace=True)
        combined.index.name = 'time'
        combined.to_csv("data/historical/M5/XAUUSDm_M5.csv")
        logger.info(f"Total M5: {len(combined)} bars saved")
        return len(combined)
    return 0

def download_h1_5y():
    """H1 for 5 years. yfinance allows 730 days max for H1, so 5y might fail or return 2y."""
    Path("data/historical/H1").mkdir(parents=True, exist_ok=True)
    ticker = yf.Ticker("GC=F")
    df = ticker.history(period="2y", interval="1h") # yfinance max for 1h is 730d (~2 years)
    df.columns = [c.lower() for c in df.columns]
    df.rename(columns={'volume': 'tick_volume'}, inplace=True)
    df.index.name = 'time'
    df.to_csv("data/historical/H1/XAUUSDm_H1.csv")
    logger.info(f"H1: {len(df)} bars saved")
    return len(df)

def download_daily_10y():
    """Daily for 10 years"""
    Path("data/historical/Daily").mkdir(parents=True, exist_ok=True)
    ticker = yf.Ticker("GC=F")
    df = ticker.history(period="10y", interval="1d")
    df.columns = [c.lower() for c in df.columns]
    df.rename(columns={'volume': 'tick_volume'}, inplace=True)
    df.index.name = 'time'
    df.to_csv("data/historical/Daily/XAUUSDm_D1.csv")
    logger.info(f"Daily: {len(df)} bars saved")
    return len(df)

if __name__ == "__main__":
    m5 = download_m5_chunked()
    h1 = download_h1_5y()
    d1 = download_daily_10y()
    print(f"\nSummary:")
    print(f"M5:    {m5} bars")
    print(f"H1:    {h1} bars")
    print(f"Daily: {d1} bars")
    print("Ready to Retrain!")
