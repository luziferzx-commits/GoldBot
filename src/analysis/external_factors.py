import yfinance as yf
import pandas as pd
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ExternalFactors:
    """
    Fetches and manages external macroeconomic factors impacting Gold:
    DXY, US10Y, VIX, WTI Oil, SP500, Bitcoin.
    """
    
    def __init__(self):
        self.symbols = {
            'dxy': 'DX-Y.NYB',
            'us10y': '^TNX',
            'vix': '^VIX',
            'oil': 'CL=F',
            'sp500': '^GSPC',
            'btc': 'BTC-USD'
        }
        self.hist_data = None
        
    def load_historical_data(self, start_date: str, end_date: str):
        """
        Loads daily historical data for backtesting.
        """
        logger.info(f"Downloading historical external data from {start_date} to {end_date}...")
        dfs = []
        for name, symbol in self.symbols.items():
            try:
                ticker = yf.Ticker(symbol)
                # Add 10 days to start to allow percent change calculations
                start_dt = pd.to_datetime(start_date) - timedelta(days=10)
                df = ticker.history(start=start_dt.strftime('%Y-%m-%d'), end=end_date)
                
                if df.empty:
                    logger.warning(f"No data returned for {symbol}")
                    continue
                    
                df.index = df.index.tz_localize(None).normalize()
                
                if name == 'vix':
                    # VIX is used as an absolute level
                    res = pd.DataFrame({f'{name}_level': df['Close']})
                elif name == 'us10y':
                    # US10Y is in % (e.g., 4.5), we track absolute change (e.g., +0.05)
                    res = pd.DataFrame({f'{name}_change': df['Close'].diff()})
                else:
                    # Others are % change
                    res = pd.DataFrame({f'{name}_change': df['Close'].pct_change() * 100})
                
                # Remove duplicates in index if any
                res = res[~res.index.duplicated(keep='last')]
                dfs.append(res)
                
            except Exception as e:
                logger.error(f"Failed to fetch {symbol}: {e}")
                
        if dfs:
            self.hist_data = pd.concat(dfs, axis=1).ffill().fillna(0.0)
            logger.info("External historical data loaded and merged.")
        else:
            self.hist_data = pd.DataFrame()
            logger.warning("No external data was successfully loaded.")
            
    def get_factors_for_date(self, target_date) -> dict:
        """
        Retrieves factors for a specific date, calculating gold bias.
        """
        result = {
            'dxy_change': 0.0,
            'us10y_change': 0.0,
            'vix_level': 15.0, # Default safe level
            'oil_change': 0.0,
            'sp500_change': 0.0,
            'btc_change': 0.0,
            'gold_bias': 0.0
        }
        
        if self.hist_data is not None and not self.hist_data.empty:
            target_dt = pd.to_datetime(target_date).normalize()
            
            # Find closest date <= target_date
            valid_idx = self.hist_data.index[self.hist_data.index <= target_dt]
            if not valid_idx.empty:
                latest_date = valid_idx[-1]
                row = self.hist_data.loc[latest_date]
                
                result['dxy_change'] = row.get('dxy_change', 0.0)
                result['us10y_change'] = row.get('us10y_change', 0.0)
                result['vix_level'] = row.get('vix_level', 15.0)
                result['oil_change'] = row.get('oil_change', 0.0)
                result['sp500_change'] = row.get('sp500_change', 0.0)
                result['btc_change'] = row.get('btc_change', 0.0)
                
        # Calculate Gold Bias
        bias = 0.0
        
        # DXY UP = Gold Down
        if result['dxy_change'] > 0.3:
            bias -= 0.2
        elif result['dxy_change'] < -0.3:
            bias += 0.2
            
        # VIX High = Fear = Gold UP
        if result['vix_level'] > 25:
            bias += 0.15
        if result['vix_level'] > 35:
            bias += 0.3
            
        # US10Y UP = Gold Down
        if result['us10y_change'] > 0.05:
            bias -= 0.15
            
        # SP500 DOWN heavily = Gold UP (Safe haven)
        if result['sp500_change'] < -1.0:
            bias += 0.2
            
        result['gold_bias'] = bias
        
        return result

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ef = ExternalFactors()
    ef.load_historical_data('2025-01-01', '2025-02-01')
    print(ef.get_factors_for_date('2025-01-15'))
