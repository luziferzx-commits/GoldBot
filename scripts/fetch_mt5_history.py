import MetaTrader5 as mt5
import pandas as pd
from pathlib import Path

if not mt5.initialize():
    print("initialize() failed")
    mt5.shutdown()
else:
    # ดึง M5 ให้ได้มากที่สุด (MT5 เก็บไว้เท่าไหร่ได้เท่านั้น)
    rates = mt5.copy_rates_from_pos("XAUUSDm", mt5.TIMEFRAME_M5, 0, 200000)
    if rates is not None and len(rates) > 0:
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        # We keep tick_volume as tick_volume because our bot's TimeframeManager expects it
        
        Path("data/historical/M5").mkdir(parents=True, exist_ok=True)
        df.to_csv("data/historical/M5/XAUUSDm_M5.csv")
        print(f"M5 bars saved: {len(df)}")
        print(f"From: {df.index[0]} To: {df.index[-1]}")
    else:
        print("Failed to get rates or 0 rates returned.")

    mt5.shutdown()
