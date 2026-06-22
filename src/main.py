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
from src.risk.risk_manager import RiskManager
from src.calendar.economic_calendar import EconomicCalendar
from src.ai.learning_mode import LearningMode

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
        
        # Strategy & Logic
        is_learning = self.settings['ai'].get('learning_mode', True)
        self.strategy = AIStrategy(is_learning=is_learning)
        self.risk_manager = RiskManager()
        self.calendar = EconomicCalendar()
        self.learning_mode = LearningMode(is_learning=is_learning)
        
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
        Main logic executed every 5 minutes.
        """
        logger.info("Executing 5-minute cycle...")
        
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
            
        # 6. Run Strategy
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
        
        # Schedule 5-minute loop (e.g., at :00, :05, :10...)
        schedule.every(5).minutes.at(":00").do(self.fetch_and_evaluate)
        
        # Daily Report at 00:05
        schedule.every().day.at("00:05").do(self.reporter.send_report)
        
        logger.info("Bot Main Loop Running. Press Ctrl+C to stop.")
        
        # First immediate run
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
