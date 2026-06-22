import logging
import MetaTrader5 as mt5
from typing import Optional, Dict, Any
import time

from src.broker.mt5_client import MT5Client
from src.storage.db import Database

logger = logging.getLogger(__name__)

class OrderManager:
    """
    Manages order execution, modification, and closing via MT5.
    """
    
    def __init__(self, client: MT5Client, db: Database, symbol: str):
        self.client = client
        self.db = db
        self.symbol = symbol

    def _has_open_positions(self) -> bool:
        """Check idempotency (only 1 open position per symbol)."""
        positions = mt5.positions_get(symbol=self.symbol)
        if positions is None:
            return False
        return len(positions) > 0

    def open_trade(self, signal, lot: float, sl: float, tp: float) -> Optional[int]:
        """
        Open a trade with 3 retries.
        signal: an object containing direction, reason, signal_source, confidence
        """
        if self._has_open_positions():
            logger.warning(f"Position already exists for {self.symbol}. Skipping.")
            return None
            
        direction = signal.direction
        order_type = mt5.ORDER_TYPE_BUY if direction == "BUY" else mt5.ORDER_TYPE_SELL
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": self.symbol,
            "volume": lot,
            "type": order_type,
            "sl": sl,
            "tp": tp,
            "magic": 100000,
            "comment": signal.reason[:20],
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # 3 Retries
        for attempt in range(3):
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                logger.error(f"Order send failed, retcode={result.retcode}. Attempt {attempt+1}/3")
                time.sleep(1)
            else:
                logger.info(f"Opened {direction} trade, ticket: {result.order}")
                
                # Log to DB
                trade_data = {
                    "symbol": self.symbol,
                    "direction": direction,
                    "lot": lot,
                    "entry_price": result.price,
                    "sl": sl,
                    "tp": tp,
                    "signal_source": getattr(signal, 'source', 'FALLBACK'),
                    "confidence": getattr(signal, 'confidence', 0.0),
                    "exit_reason": "OPEN"
                }
                self.db.log_trade(trade_data)
                return result.order
                
        return None

    def close_trade(self, ticket: int, reason: str = "SIGNAL") -> bool:
        """Close a specific position by ticket."""
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            logger.warning(f"Position {ticket} not found.")
            return False
            
        position = positions[0]
        tick = mt5.symbol_info_tick(position.symbol)
        
        close_type = mt5.ORDER_TYPE_SELL if position.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": position.symbol,
            "volume": position.volume,
            "type": close_type,
            "position": position.ticket,
            "price": price,
            "magic": 100000,
            "comment": reason,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Closed trade {ticket}. Reason: {reason}")
            # In a real system, you'd update the existing DB row here to set exit_price and pnl.
            return True
        else:
            logger.error(f"Failed to close {ticket}, retcode={result.retcode}")
            return False

    def close_all_trades(self, reason: str = "MANUAL_CLOSE"):
        """Close all open positions for this symbol."""
        positions = mt5.positions_get(symbol=self.symbol)
        if positions:
            for pos in positions:
                self.close_trade(pos.ticket, reason=reason)

    def force_close_all(self):
        """Emergency close all."""
        self.close_all_trades(reason="FORCE_CLOSE_EOD")

    def modify_sl_tp(self, ticket: int, new_sl: float, new_tp: float) -> bool:
        """Modify SL and TP of an existing position."""
        positions = mt5.positions_get(ticket=ticket)
        if not positions:
            return False
            
        pos = positions[0]
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": pos.symbol,
            "position": pos.ticket,
            "sl": new_sl,
            "tp": new_tp
        }
        result = mt5.order_send(request)
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            logger.info(f"Modified SL/TP for {ticket}")
            return True
        return False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    import yaml
    try:
        with open("config/settings.yaml", "r") as f:
            settings = yaml.safe_load(f)
        client = MT5Client(login=settings['broker']['login'], password=settings['broker']['password'], server=settings['broker']['server'])
        if client.connect():
            db = Database()
            om = OrderManager(client, db, settings['broker']['symbol'])
            om.force_close_all()
            client.disconnect()
    except Exception as e:
        print(f"Test failed: {e}")
