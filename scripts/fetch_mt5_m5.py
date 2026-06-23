import MetaTrader5 as mt5
import pandas as pd
from pathlib import Path

if not mt5.initialize():
    print("Failed to initialize MT5")
    exit()

mt5.symbol_select("XAUUSDm", True)

# Try fetching as much as possible by decreasing the count
for count in [200000, 100000, 50000, 30000, 10000]:
    rates = mt5.copy_rates_from_pos("XAUUSDm", mt5.TIMEFRAME_M5, 0, count)
    if rates is not None and len(rates) > 0:
        break

if rates is not None and len(rates) > 0:
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    
    Path("data/historical/M5").mkdir(parents=True, exist_ok=True)
    df.to_csv("data/historical/M5/XAUUSDm_M5.csv")
    print(f"Saved: {len(df)} bars")
    print(f"From: {df.index[0]}")
    print(f"To: {df.index[-1]}")
else:
    print("No data retrieved from MT5")

mt5.shutdown()
