import logging
import yaml
from typing import Tuple

logger = logging.getLogger(__name__)

class RiskManager:
    """
    Manages risk parameters, position sizing, SL/TP calculation, and trade restrictions.
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """
        Initialize RiskManager from config.
        """
        try:
            with open(config_path, "r") as f:
                settings = yaml.safe_load(f)
            self.risk_per_trade = settings['risk']['risk_per_trade']
            self.max_daily_loss = settings['risk']['max_daily_loss']
            self.max_drawdown = settings['risk']['max_drawdown']
            self.sl_atr_multiplier = settings['risk'].get('sl_atr_multiplier', 1.5)
            self.tp1_atr_multiplier = settings['risk'].get('tp1_atr_multiplier', 1.5)
            self.tp2_atr_multiplier = settings['risk'].get('tp_atr_multiplier', 3.0)
            self.partial_tp_ratio = settings['risk'].get('partial_tp_ratio', 0.5)
            self.max_consecutive_losses = settings['risk'].get('max_consecutive_losses', 4)
            self.max_spread_pips = settings['risk'].get('max_spread', 50)
            self.max_trades_per_day = settings['trading']['max_trades_per_day']
        except Exception as e:
            logger.error(f"Failed to load risk config, using defaults: {e}")
            self.risk_per_trade = 0.005
            self.max_daily_loss = 0.03
            self.max_drawdown = 0.15
            self.sl_atr_multiplier = 1.5
            self.tp1_atr_multiplier = 1.5
            self.tp2_atr_multiplier = 3.0
            self.partial_tp_ratio = 0.5
            self.max_consecutive_losses = 4
            self.max_spread_pips = 50
            self.max_trades_per_day = 6
            
        # Example state
        self.daily_loss_pct = 0.0
        self.drawdown_pct = 0.0
        self.consecutive_losses = 0
        self.trades_today = 0

    def update_daily_stats(self):
        """Fetch today's closed trades from MT5 to calculate real daily loss."""
        try:
            import MetaTrader5 as mt5
            from datetime import datetime, time
            
            today_start = datetime.combine(datetime.today(), time.min)
            now = datetime.now()
            
            deals = mt5.history_deals_get(today_start, now)
            if deals:
                daily_pnl = sum(deal.profit for deal in deals)
                account_info = mt5.account_info()
                if account_info:
                    start_equity = account_info.balance - daily_pnl
                    if start_equity > 0:
                        self.daily_loss_pct = -daily_pnl / start_equity if daily_pnl < 0 else 0.0
            else:
                self.daily_loss_pct = 0.0
        except Exception as e:
            pass

    def record_trade_result(self, won: bool):
        self.trades_today += 1
        if won:
            self.consecutive_losses = 0
        else:
            self.consecutive_losses += 1

    def get_breakeven_price(self, entry_price: float, direction: str, spread_pips: float = 3.0) -> float:
        # Convert pips to price (XAUUSD: 1 pip = 0.1, but often 1 pip = 0.01 depending on broker. Assuming 0.01)
        spread_cost = spread_pips * 0.01
        if direction == "BUY":
            return entry_price + spread_cost
        else:
            return entry_price - spread_cost

    def get_risk_summary(self) -> dict:
        return {
            "daily_loss_pct": self.daily_loss_pct,
            "drawdown_pct": self.drawdown_pct,
            "consecutive_losses": self.consecutive_losses,
            "trades_today": self.trades_today
        }

    def evaluate(self, equity: float, entry_price: float, atr: float, direction: str, confidence: float = 0.0, spread_pips: float = 0.0) -> Tuple[bool, float, float, float, float, str]:
        """
        Evaluate if trade is allowed and compute lot, sl, tp1, tp2.
        """
        self.update_daily_stats()
        
        if spread_pips > self.max_spread_pips:
            return False, 0.0, 0.0, 0.0, 0.0, f"Spread {spread_pips} exceeds max {self.max_spread_pips}"
            
        # Hard limits check
        if self.daily_loss_pct >= self.max_daily_loss:
            return False, 0.0, 0.0, 0.0, 0.0, "Max daily loss reached"
        if self.drawdown_pct >= self.max_drawdown:
            return False, 0.0, 0.0, 0.0, 0.0, "Max drawdown reached"
        if self.trades_today >= self.max_trades_per_day:
            return False, 0.0, 0.0, 0.0, 0.0, "Max trades per day reached"
        if self.consecutive_losses >= self.max_consecutive_losses:
            return False, 0.0, 0.0, 0.0, 0.0, "Max consecutive losses reached"
            
        # Compute SL and TP distances
        sl_dist = atr * self.sl_atr_multiplier
        tp1_dist = atr * self.tp1_atr_multiplier
        tp2_dist = atr * self.tp2_atr_multiplier
        
        # SL scaling by confidence (0.85 to 1.15)
        sl_dist *= (0.85 + confidence * 0.30)
        
        if sl_dist <= 0:
            return False, 0.0, 0.0, 0.0, 0.0, "Invalid ATR"
            
        # Compute prices
        if direction == "BUY":
            sl_price = entry_price - sl_dist
            tp1_price = entry_price + tp1_dist
            tp2_price = entry_price + tp2_dist
        else: # SELL
            sl_price = entry_price + sl_dist
            tp1_price = entry_price - tp1_dist
            tp2_price = entry_price - tp2_dist
            
        # Compute Lot Size
        risk_amount = equity * self.risk_per_trade
        # Assuming XAUUSD: 1 point (0.01) move on 1 lot = $1, so 1.00 move = $100
        lot = risk_amount / (sl_dist * 100.0)
        
        # Lot scaling by confidence (0.80 to 1.20)
        lot *= (0.80 + confidence * 0.40)
        
        # Round lot to 2 decimal places
        lot = round(lot, 2)
        if lot < 0.01:
            return False, 0.0, 0.0, 0.0, 0.0, "Lot size too small"
            
        return True, lot, sl_price, tp1_price, tp2_price, "Approved"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rm = RiskManager()
    approved, lot, sl, tp1, tp2, reason = rm.evaluate(equity=10000.0, entry_price=2000.0, atr=5.0, direction="BUY", confidence=0.8, spread_pips=20.0)
    print(f"Risk Check: Approved={approved}, Lot={lot}, SL={sl}, TP1={tp1}, TP2={tp2}, Reason={reason}")
