import MetaTrader5 as mt5
import pandas as pd
import logging
import time
from typing import Optional, List, Dict, Any, Tuple
from functools import wraps

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
)
logger = logging.getLogger(__name__)

def ensure_connection(func):
    """
    Decorator to ensure MT5 connection is active before executing a method.
    If the connection is dead, it attempts to reconnect seamlessly.
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # mt5.terminal_info() returns None if MT5 is disconnected/crashed
        if mt5.terminal_info() is None:
            logger.warning(f"MT5 disconnected detected before {func.__name__}. Attempting reconnection...")
            mt5.shutdown() # Free memory and clean up lingering handlers
            time.sleep(1)
            
            if not self.connect():
                logger.error(f"Failed to reconnect to MT5 for {func.__name__}.")
                return None # Return None on catastrophic failure
                
        return func(self, *args, **kwargs)
    return wrapper

class MT5Client:
    """
    Client for interacting with MetaTrader 5 Terminal.
    Handles connections, data fetching, and order execution.
    """

    def __init__(self, login: int = 0, password: str = "", server: str = ""):
        """
        Initialize MT5 client.
        
        Args:
            login: MT5 account number
            password: MT5 password
            server: MT5 broker server name
        """
        self.login = login
        self.password = password
        self.server = server

    def connect(self) -> bool:
        """
        Connect to MT5 terminal with retry mechanism.
        
        Returns:
            bool: True if connected successfully, False otherwise.
        """
        max_retries = 3
        for attempt in range(max_retries):
            logger.info(f"Connecting to MT5 (Attempt {attempt + 1}/{max_retries})...")
            
            if not mt5.initialize():
                logger.error(f"MT5 initialize failed: {mt5.last_error()}")
                time.sleep(2)
                continue
                
            if self.login and self.password and self.server:
                authorized = mt5.login(self.login, password=self.password, server=self.server)
                if not authorized:
                    logger.error(f"MT5 login failed: {mt5.last_error()}")
                    mt5.shutdown()
                    time.sleep(2)
                    continue
            
            logger.info("Successfully connected to MT5.")
            return True
            
        logger.error("Failed to connect to MT5 after maximum retries.")
        return False

    def disconnect(self) -> None:
        """Disconnect from MT5 terminal."""
        mt5.shutdown()
        logger.info("Disconnected from MT5.")

    @ensure_connection
    def get_current_price(self, symbol: str) -> Optional[Dict[str, float]]:
        """
        Get current bid, ask, and spread for a symbol.
        
        Args:
            symbol: Trading symbol (e.g., "XAUUSD")
            
        Returns:
            Dict containing 'bid', 'ask', 'spread' or None if failed.
        """
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Failed to get current price for {symbol}: {mt5.last_error()}")
            return None
            
        info = mt5.symbol_info(symbol)
        if info is None:
            logger.error(f"Failed to get symbol info for {symbol}: {mt5.last_error()}")
            return None
            
        spread = tick.ask - tick.bid
        return {
            "bid": tick.bid,
            "ask": tick.ask,
            "spread": spread
        }

    @ensure_connection
    def get_bars(self, symbol: str, timeframe_str: str, count: int) -> Optional[pd.DataFrame]:
        """
        Get historical OHLCV data.
        
        Args:
            symbol: Trading symbol
            timeframe_str: Timeframe string (M5, M15, H1, D1, MN1)
            count: Number of bars to fetch
            
        Returns:
            pandas DataFrame with OHLCV data or None if failed.
        """
        tf_map = {
            "M5": mt5.TIMEFRAME_M5,
            "M15": mt5.TIMEFRAME_M15,
            "H1": mt5.TIMEFRAME_H1,
            "D1": mt5.TIMEFRAME_D1,
            "MN1": mt5.TIMEFRAME_MN1
        }
        
        if timeframe_str not in tf_map:
            logger.error(f"Invalid timeframe: {timeframe_str}")
            return None
            
        tf = tf_map[timeframe_str]
        
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
        if rates is None or len(rates) == 0:
            logger.error(f"Failed to get bars for {symbol} {timeframe_str}: {mt5.last_error()}")
            return None
            
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        return df

    @ensure_connection
    def get_open_positions(self) -> List[Dict[str, Any]]:
        """
        Get all open positions.
        
        Returns:
            List of dictionaries containing position details.
        """
        positions = mt5.positions_get()
        if positions is None:
            logger.error(f"Failed to get open positions: {mt5.last_error()}")
            return []
            
        return [pos._asdict() for pos in positions]

    @ensure_connection
    def get_account_info(self) -> Optional[Dict[str, float]]:
        """
        Get account equity, balance, and margin.
        
        Returns:
            Dict with 'equity', 'balance', 'margin', 'free_margin' or None if failed.
        """
        info = mt5.account_info()
        if info is None:
            logger.error(f"Failed to get account info: {mt5.last_error()}")
            return None
            
        return {
            "equity": info.equity,
            "balance": info.balance,
            "margin": info.margin,
            "free_margin": info.margin_free
        }

    @ensure_connection
    def send_market_order(self, symbol: str, direction: str, lot: float, sl: float, tp: float, comment: str = "") -> Optional[int]:
        """
        Send a market order.
        
        Args:
            symbol: Trading symbol
            direction: "BUY" or "SELL"
            lot: Volume size
            sl: Stop Loss price
            tp: Take Profit price
            comment: Order comment
            
        Returns:
            Order ticket number if successful, None otherwise.
        """
        direction = direction.upper()
        if direction not in ["BUY", "SELL"]:
            logger.error(f"Invalid direction: {direction}")
            return None
            
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Symbol not found: {symbol}")
            return None
            
        if not symbol_info.visible:
            if not mt5.symbol_select(symbol, True):
                logger.error(f"Failed to select symbol {symbol}")
                return None
        
        price = mt5.symbol_info_tick(symbol).ask if direction == "BUY" else mt5.symbol_info_tick(symbol).bid
        order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "price": price,
            "sl": sl,
            "tp": tp,
            "deviation": 20,
            "magic": 123456, # Configurable magic number
            "comment": comment,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order send failed: retcode={result.retcode}, comment={result.comment}")
            return None
            
        logger.info(f"Order sent successfully. Ticket: {result.order}")
        return result.order

    @ensure_connection
    def close_position(self, ticket: int) -> bool:
        """
        Close an existing position by ticket.
        
        Args:
            ticket: Order ticket number
            
        Returns:
            bool: True if successful, False otherwise.
        """
        position = mt5.positions_get(ticket=ticket)
        if position is None or len(position) == 0:
            logger.error(f"Position {ticket} not found.")
            return False
            
        pos = position[0]
        symbol = pos.symbol
        lot = pos.volume
        order_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(symbol).bid if pos.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lot,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": 20,
            "magic": 123456,
            "comment": "Close position",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Failed to close position {ticket}: {result.comment}")
            return False
            
        logger.info(f"Position {ticket} closed successfully.")
        return True

    @ensure_connection
    def close_all_positions(self) -> int:
        """
        Close all open positions.
        
        Returns:
            int: Number of positions closed.
        """
        positions = mt5.positions_get()
        if positions is None or len(positions) == 0:
            logger.info("No open positions to close.")
            return 0
            
        closed_count = 0
        for pos in positions:
            if self.close_position(pos.ticket):
                closed_count += 1
                
        logger.info(f"Closed {closed_count}/{len(positions)} positions.")
        return closed_count

if __name__ == "__main__":
    import yaml
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Load settings
    try:
        with open("config/settings.yaml", "r") as f:
            settings = yaml.safe_load(f)
        login_val = os.getenv('MT5_LOGIN', settings['broker'].get('login'))
        login = int(login_val) if login_val else 0
        password = os.getenv('MT5_PASSWORD', settings['broker'].get('password', ''))
        server = os.getenv('MT5_SERVER', settings['broker'].get('server', ''))
        symbol = settings['broker']['symbol']
    except Exception as e:
        logger.error(f"Failed to load settings.yaml: {e}")
        login, password, server, symbol = 0, "", "", "XAUUSD"

    # Simple test
    client = MT5Client(login=login, password=password, server=server)
    if client.connect():
        try:
            info = client.get_account_info()
            if info:
                print(f"Account Info: {info}")
                
            price = client.get_current_price(symbol)
            if price:
                print(f"Current Price {symbol}: {price}")
                
            bars = client.get_bars(symbol, "M5", 10)
            if bars is not None:
                print(f"Recent M5 Bars:\n{bars.head()}")
        finally:
            client.disconnect()
