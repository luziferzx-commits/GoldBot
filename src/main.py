import logging
import time
import schedule
import signal
import sys
import yaml
from datetime import datetime

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("logs/bot.log")]
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
        self.client = MT5Client(
            login=self.settings['broker']['login'],
            password=self.settings['broker']['password'],
            server=self.settings['broker']['server']
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
        
        # Filters
        self.risk_manager = RiskManager()
        self.calendar = EconomicCalendar()
        self.learning_mode = LearningMode(is_learning=is_learning)
        self.external_factors = ExternalFactors()
        self.sentiment_analyzer = SentimentAnalyzer()
        
        # Notifications
        self.notifier = TelegramNotifier(self.db)
        self.reporter = DailyReporter(self.db, self.notifier)
        
        # State
        self.is_running = True
        self.setup_telegram_commands()

    def setup_telegram_commands(self):
        self.notifier.register_command("/status", lambda: "Bot is running. " + ("LEARNING MODE" if self.learning_mode.is_learning else "LIVE MODE"))
        self.notifier.register_command("/stop", self.handle_stop_command)
        self.notifier.register_command("/close_all", self.handle_close_all)
        self.notifier.register_command("/summary", lambda: self.reporter.generate_report())

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
            
            h1['sentiment_score'] = self.sentiment_analyzer.analyze_sentiment()
            
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
            
        # 6. Run Strategy (24-Hour Routing)
        current_hour_gmt7 = (datetime.utcnow() + pd.Timedelta(hours=7)).hour
        signal = Signal("HOLD", 0.0)

        # Ensure PO3 records manipulation phase continuously
        self.po3_strategy.generate_signal(m5, m15, h1, d1, mn1)

        if 8 <= current_hour_gmt7 < 10:
            signal = self.sge_strategy.generate_signal(m5, m15, h1, d1, mn1)
        elif 15 <= current_hour_gmt7 < 16:
            signal = self.asian_strategy.generate_signal(m5, m15, h1, d1, mn1)
        elif 19 <= current_hour_gmt7 < 23:
            signal = self.overlap_scalper.generate_signal(m5, m15, h1, d1, mn1)
            
        if signal.direction == "HOLD":
            # Silver Bullet logic checks its own time windows inside
            signal = self.sb_strategy.generate_signal(m5, m15, h1, d1, mn1)
            
        if signal.direction == "HOLD":
            # PO3 only generates signal during Distribution (15:00-23:00) if bias met
            signal = self.po3_strategy.generate_signal(m5, m15, h1, d1, mn1)
            
        if signal.direction == "HOLD":
            # Fallback to AI Strategy
            signal = self.strategy.generate_signal(m5, m15, h1, d1, mn1)
            
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
                
        # 10. Check Retrain (Simulated)
        # if self.learning_mode.should_retrain(...): trigger retraining thread
        
    def start(self):
        # Initial connections
        self.client.connect()
        self.calendar.fetch_news()
        self.notifier.start_polling()
        self.notifier.send_message("🤖 <b>GoldBot Started</b>\nMonitoring XAUUSD...")
        
        # Daily Report at 00:05
        schedule.every().day.at("00:05").do(self.reporter.send_report)
        
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

if __name__ == "__main__":
    bot = GoldBot()
    signal.signal(signal.SIGINT, bot.stop)
    signal.signal(signal.SIGTERM, bot.stop)
    bot.start()
    bot.run()
