import logging
import time
import schedule
import signal
import sys
import os
import yaml
import pandas as pd
from datetime import datetime
from pathlib import Path
from logging.handlers import RotatingFileHandler
from dotenv import load_dotenv

load_dotenv()

from src.broker.mt5_client import MT5Client
from src.data.timeframe_manager import TimeframeManager
from src.storage.db import Database
from src.execution.order_manager import OrderManager
from src.notify.telegram_bot import TelegramNotifier
from src.summary.daily_report import DailyReporter
from src.strategy.ai_strategy import AIStrategy
from src.strategy.silver_bullet_strategy import SilverBulletStrategy
from src.strategy.asian_range_strategy import AsianRangeStrategy
from src.strategy.sge_strategy import SGEStrategy
from src.strategy.po3_strategy import PO3Strategy
from src.strategy.overlap_scalper import OverlapScalper
from src.risk.risk_manager import RiskManager
from src.calendar.economic_calendar import EconomicCalendar
from src.ai.learning_mode import LearningMode
from src.analysis.external_factors import ExternalFactors
from src.analysis.sentiment_analyzer import SentimentAnalyzer
from src.strategy.strategy_selector import StrategySelector, MarketContext
from src.ai.online_learner import OnlineLearner
from src.storage.backup_manager import DatabaseBackup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(),
        RotatingFileHandler("logs/bot.log", maxBytes=10485760, backupCount=5)
    ]
)
logger = logging.getLogger(__name__)

class GoldBot:
    def __init__(self):
        logger.info("Initializing GoldBot...")
        
        try:
            with open("config/settings.yaml", "r") as f:
                self.settings = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load settings: {e}")
            sys.exit(1)
            
        # Core Modules
        self.db = Database()
        
        login_val = os.getenv('MT5_LOGIN', self.settings['broker'].get('login'))
        self.client = MT5Client(
            login=int(login_val) if login_val else 0,
            password=os.getenv('MT5_PASSWORD', self.settings['broker'].get('password')),
            server=os.getenv('MT5_SERVER', self.settings['broker'].get('server'))
        )
        self.symbol = self.settings['broker']['symbol']
        self.manager = TimeframeManager(self.client, self.symbol)
        self.order_manager = OrderManager(self.client, self.db, self.symbol)
        
        self.manager = TimeframeManager(self.client, self.settings['broker']['symbol'])
        
        # Strategies
        is_learning = self.settings['ai'].get('learning_mode', True)
        self.strategy = AIStrategy(is_learning=is_learning)
        self.sb_strategy = SilverBulletStrategy(self.strategy)
        self.asian_strategy = AsianRangeStrategy(self.strategy)
        self.sge_strategy = SGEStrategy(self.strategy)
        self.po3_strategy = PO3Strategy(self.strategy)
        self.overlap_scalper = OverlapScalper(self.strategy)
        
        # Strategy Selector
        self.selector = StrategySelector(self.db, {
            "silver_bullet": self.sb_strategy,
            "ai_strategy": self.strategy,
            "asian_range": self.asian_strategy,
            "sge": self.sge_strategy,
            "po3": self.po3_strategy,
            "overlap": self.overlap_scalper
        })
        
        # Filters
        self.risk_manager = RiskManager()
        self.calendar = EconomicCalendar()
        self.learning_mode = LearningMode(is_learning=is_learning)
        self.external_factors = ExternalFactors()
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Online Learning
        self.online_learner = OnlineLearner(self.strategy.model)
        
        # Notifications
        self.notifier = TelegramNotifier(self.db)
        self.reporter = DailyReporter(self.db, self.notifier)
        
        # State
        self.backup_manager = DatabaseBackup()
        self.is_running = True
        self.setup_telegram_commands()

    def setup_telegram_commands(self):
        self.notifier.register_command("/status", lambda: "Bot is running. " + ("LEARNING MODE" if self.learning_mode.is_learning else "LIVE MODE"))
        self.notifier.register_command("/stop", self.handle_stop_command)
        self.notifier.register_command("/close_all", self.handle_close_all)
        self.notifier.register_command("/summary", lambda: self.reporter.generate_report())
        self.notifier.register_command("/selector", lambda: self.selector.get_status_text())
        self.notifier.register_command("/history", lambda: self.selector.get_history_text())

    def handle_stop_command(self):
        self.notifier.send_message("Stopping bot and closing all positions...")
        self.order_manager.force_close_all()
        self.is_running = False
        return "Bot stopped."

    def handle_close_all(self):
        self.order_manager.close_all_trades(reason="MANUAL_TG_CMD")
        return "All positions closed."

    def fetch_and_evaluate(self):
        """
        Main logic executed every 1 hour (or 5 min, depending on schedule).
        """
        logger.info("Executing main cycle...")
        
        # 1. Check Connection
        if not self.client.connect():
            logger.error("MT5 disconnected. Skipping cycle.")
            self.notifier.send_alert("MT5 Disconnected!")
            return
            
        # 2. Fetch Data
        if not self.manager.fetch_all():
            logger.error("Failed to fetch data. Skipping cycle.")
            return
            
        m5 = self.manager.get_data("M5")
        m15 = self.manager.get_data("M15")
        h1 = self.manager.get_data("H1")
        d1 = self.manager.get_data("D1")
        mn1 = self.manager.get_data("MN1")
        
        # Fetch external data for live
        try:
            today_str = datetime.now().strftime('%Y-%m-%d')
            self.external_factors.load_historical_data(today_str, today_str)
            if self.external_factors.hist_data is not None and not self.external_factors.hist_data.empty:
                ext_latest = self.external_factors.hist_data.iloc[-1]
                for col in ext_latest.index:
                    h1[col] = ext_latest[col]
            else:
                for col in ['dxy_change', 'us10y_change', 'vix_level', 'oil_change', 'sp500_change']:
                    h1[col] = 0.0
                    if col == 'vix_level': h1[col] = 15.0
            
            score, _, _ = self.sentiment_analyzer.fetch_and_analyze()
            h1['sentiment_score'] = score
            
            # Recompute bias
            h1['gold_bias'] = 0.0
            h1.loc[h1['dxy_change'] > 0.3, 'gold_bias'] -= 0.2
            h1.loc[h1['dxy_change'] < -0.3, 'gold_bias'] += 0.2
            h1.loc[h1['vix_level'] > 25, 'gold_bias'] += 0.15
            h1.loc[h1['vix_level'] > 35, 'gold_bias'] += 0.3
            h1.loc[h1['us10y_change'] > 0.05, 'gold_bias'] -= 0.15
            h1.loc[h1['sp500_change'] < -1.0, 'gold_bias'] += 0.2
        except Exception as e:
            logger.error(f"Error fetching external factors: {e}")
            for col in ['dxy_change', 'us10y_change', 'vix_level', 'oil_change', 'sp500_change', 'gold_bias', 'sentiment_score']:
                h1[col] = 0.0
                if col == 'vix_level': h1[col] = 15.0
        
        # 3. Check Force Close Time (e.g. 23:45)
        now_hm = datetime.now().strftime("%H:%M")
        if now_hm == self.settings['trading']['force_close_time']:
            logger.info("Force close time reached.")
            self.order_manager.force_close_all()
            return
            
        # 3.5 Check recently closed trades for Online Learning
        self._trigger_online_learning(h1)

        # 4. Check Circuit Breaker (Handled by RiskManager internally, but let's update equity first)
        account_info = self.client.get_account_info()
        equity = account_info.get('equity', 0.0)
        self.db.log_equity({
            "equity": equity,
            "balance": account_info.get('balance', 0.0),
            "daily_pnl": 0.0, # Needs calculation
            "daily_pnl_pct": 0.0,
            "drawdown": 0.0,
            "drawdown_pct": 0.0
        })
        
        # 5. Check News
        if self.calendar.is_news_time():
            logger.info("High impact news window. Skipping trading.")
            return
            
        # 6. Run Strategy (24-Hour Routing via StrategySelector)
        current_hour_gmt7 = (datetime.utcnow() + pd.Timedelta(hours=7)).hour
        
        session = "OTHER"
        if 8 <= current_hour_gmt7 < 10: session = "SGE"
        elif 10 <= current_hour_gmt7 < 15: session = "ASIAN"
        elif 15 <= current_hour_gmt7 < 19: session = "LONDON"
        elif 19 <= current_hour_gmt7 < 23: session = "OVERLAP"
        elif 23 <= current_hour_gmt7 or current_hour_gmt7 < 2: session = "NY"
        
        current_price = m5['close'].iloc[-1]
        atr = h1['D1_ATR'].iloc[-1] if 'D1_ATR' in h1.columns else 5.0
        h1_trend = h1['H1_trend'].iloc[-1] if 'H1_trend' in h1.columns else "SIDEWAYS"
        
        # Priority #2: Manage Open Positions (Breakeven & Trailing Stop)
        self.order_manager.manage_open_positions(atr)
        
        from src.analysis.market_regime import MarketRegime
        regime_analyzer = MarketRegime()
        m5 = regime_analyzer.analyze(m5)
        regime = m5['market_regime'].iloc[-1]
        
        # Ensure PO3 records manipulation phase continuously
        self.po3_strategy.generate_signal(m5, m15, h1, d1, mn1)
        
        ai_direction, ai_conf = self.strategy.get_raw_prediction(m5)
        
        context = MarketContext(
            current_time=datetime.utcnow() + pd.Timedelta(hours=7),
            market_regime=regime,
            session=session,
            ai_confidence=ai_conf,
            volatility_ratio=atr / 5.0, # Simple proxy
            volume_spike=m5['tick_volume'].iloc[-1] > m5['tick_volume'].rolling(20).mean().iloc[-1] * 1.5,
            h1_trend=h1_trend,
            asian_range_formed=self.po3_strategy.asian_high > 0,
            is_news_window=self.calendar.is_news_time()
        )
        
        strategy_name, score = self.selector.select(context, h1, d1)
        from src.strategy.base import Signal
        signal = Signal("HOLD", 0.0)
        
        if strategy_name == "SKIP":
            logger.info(f"No trade - score too low ({score:.0f})")
        else:
            logger.info(f"StrategySelector chose {strategy_name} with score {score:.0f}")
            selected_strategy = self.selector.strategies[strategy_name]
            signal = selected_strategy.generate_signal(m5, m15, h1, d1, mn1)
            signal.source = strategy_name # Used for logging to DB
            
        logger.info(f"Signal generated: {signal}")
        
        if signal.direction in ["BUY", "SELL"]:
            # 7. Risk Check
            atr = 5.0 # Should be dynamic from ATR indicator
            approved, lot, sl, tp, reason = self.risk_manager.evaluate(
                equity=equity,
                entry_price=signal.entry_price,
                atr=atr,
                direction=signal.direction
            )
            
            if approved:
                # 8. Execute
                ticket = self.order_manager.open_trade(signal, lot, sl, tp)
                if ticket:
                    self.notifier.send_trade_open(signal, lot, sl, tp)
            else:
                logger.info(f"Trade rejected by Risk Manager: {reason}")
                
        # 10. Write Heartbeat
        self._write_heartbeat()
        
    def _write_heartbeat(self):
        try:
            with open("data/heartbeat.txt", "w") as f:
                f.write(datetime.utcnow().isoformat())
        except Exception as e:
            logger.error(f"Failed to write heartbeat: {e}")
            
    def start(self):
        # Initial connections
        self.client.connect()
        self.calendar.fetch_news()
        self.notifier.start_polling()
        self.notifier.send_message("🤖 <b>GoldBot Started</b>\nMonitoring XAUUSD...")
        
        # Daily Report at 00:05
        schedule.every().day.at("00:05").do(self.reporter.send_report)
        
        # Daily DB Backup at 00:00
        schedule.every().day.at("00:00").do(self.backup_manager.backup)
        
    def run(self):
        logger.info("Starting Main Loop...")
        # Schedule the cycle every 5 minutes for Silver Bullet M5 precision
        schedule.every(5).minutes.do(self.fetch_and_evaluate)
        
        # Initial run
        self.fetch_and_evaluate()
        
        while self.is_running:
            schedule.run_pending()
            time.sleep(1)
            
    def stop(self, signum=None, frame=None):
        logger.info("Shutting down...")
        self.is_running = False
        self.notifier.send_message("🤖 <b>GoldBot Stopped</b>")
        self.notifier.stop_polling()
        self.client.disconnect()
        sys.exit(0)

    def _trigger_online_learning(self, h1_data):
        """
        Check for recently closed trades in the DB or MT5 history.
        If a trade just closed, pass it to OnlineLearner to learn from it immediately.
        """
        # In a full live implementation, we'd pull from mt5.history_deals_get() 
        # and match with our local DB to find freshly closed trades.
        # This is a hook simulating that logic.
        pass
        
if __name__ == "__main__":
    bot = GoldBot()
    signal.signal(signal.SIGINT, bot.stop)
    signal.signal(signal.SIGTERM, bot.stop)
    bot.start()
    bot.run()
