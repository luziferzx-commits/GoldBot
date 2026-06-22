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
            self.sl_atr_multiplier = settings['risk']['sl_atr_multiplier']
            self.tp_atr_multiplier = settings['risk']['tp_atr_multiplier']
            self.max_consecutive_losses = settings['risk']['max_consecutive_losses']
            self.max_trades_per_day = settings['trading']['max_trades_per_day']
        except Exception as e:
            logger.error(f"Failed to load risk config, using defaults: {e}")
            self.risk_per_trade = 0.005
            self.max_daily_loss = 0.03
            self.max_drawdown = 0.15
            self.sl_atr_multiplier = 2.0
            self.tp_atr_multiplier = 3.0
            self.max_consecutive_losses = 3
            self.max_trades_per_day = 10
            
        # Example state (In a real system, these would be fetched from a DB or Account Info)
        self.daily_loss_pct = 0.0
        self.drawdown_pct = 0.0
        self.consecutive_losses = 0
        self.trades_today = 0

    def update_daily_stats(self):
        """Fetch today's closed trades from MT5 to calculate real daily loss."""
        import MetaTrader5 as mt5
        from datetime import datetime, time
        
        today_start = datetime.combine(datetime.today(), time.min)
        now = datetime.now()
        
        deals = mt5.history_deals_get(today_start, now)
        if deals:
            # Sum up closed profit/loss today
            daily_pnl = sum(deal.profit for deal in deals)
            
            account_info = mt5.account_info()
            if account_info:
                start_equity = account_info.balance - daily_pnl
                if start_equity > 0:
                    self.daily_loss_pct = -daily_pnl / start_equity if daily_pnl < 0 else 0.0
        else:
            self.daily_loss_pct = 0.0

    def evaluate(self, equity: float, entry_price: float, atr: float, direction: str, confidence: float = 0.0) -> Tuple[bool, float, float, float, str]:
        """
        Evaluate if trade is allowed and compute lot, sl, tp.
        
        Args:
            equity: Current account equity
            entry_price: The expected entry price
            atr: The Average True Range for SL/TP calculation
            direction: "BUY" or "SELL"
            confidence: AI Prediction confidence (0.0 to 1.0)
            
        Returns:
            Tuple[bool, float, float, float, str]: (approved, lot, sl, tp, reason)
        """
        # Update stats from MT5
        self.update_daily_stats()
        
        # Hard limits check
        if self.daily_loss_pct >= self.max_daily_loss:
            return False, 0.0, 0.0, 0.0, "Max daily loss reached"
        if self.drawdown_pct >= self.max_drawdown:
            return False, 0.0, 0.0, 0.0, "Max drawdown reached"
        if self.trades_today >= self.max_trades_per_day:
            return False, 0.0, 0.0, 0.0, "Max trades per day reached"
        if self.consecutive_losses >= self.max_consecutive_losses:
            # Here we might just reject, external logic handles 1 hr pause
            return False, 0.0, 0.0, 0.0, "Max consecutive losses reached"
            
        # Compute SL and TP distances
        sl_dist = atr * self.sl_atr_multiplier
        tp_dist = atr * self.tp_atr_multiplier
        
        if sl_dist <= 0:
            return False, 0.0, 0.0, 0.0, "Invalid ATR"
            
        # Compute prices
        if direction == "BUY":
            sl_price = entry_price - sl_dist
            tp_price = entry_price + tp_dist
        else: # SELL
            sl_price = entry_price + sl_dist
            tp_price = entry_price - tp_dist
            
        # Compute Dynamic Risk Multiplier
        multiplier = 1.0
        if confidence >= 0.85:
            multiplier = 2.0  # A+ Setup (capped at 2.0x for safety)
        elif confidence >= 0.70:
            multiplier = 1.0  # A Setup
        elif confidence > 0.0:
            multiplier = 0.5  # B/C Setup
            
        # Compute Lot Size
        # Risk amount = equity * risk_per_trade * multiplier
        # Lot size depends on contract specification. For XAUUSD, 1 lot = 100 oz.
        risk_amount = equity * self.risk_per_trade * multiplier
        tick_value = 1.0 # Approximate for XAUUSD (1 lot moves $1 = $100 profit/loss)
        # Assuming SL distance in points or absolute price. If price moves by sl_dist, how much per lot?
        # Standard XAUUSD: 1 point (0.01) move on 1 lot = $1
        # Therefore, price move of 1.00 on 1 lot = $100
        # Risk amount = lot * (sl_dist * 100)
        lot = risk_amount / (sl_dist * 100.0)
        
        # Round lot to 2 decimal places (standard MT5 step)
        lot = round(lot, 2)
        if lot < 0.01:
            return False, 0.0, 0.0, 0.0, "Lot size too small"
            
        return True, lot, sl_price, tp_price, "Approved"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    rm = RiskManager()
    approved, lot, sl, tp, reason = rm.evaluate(equity=10000.0, entry_price=2000.0, atr=5.0, direction="BUY")
    print(f"Risk Check: Approved={approved}, Lot={lot}, SL={sl}, TP={tp}, Reason={reason}")
