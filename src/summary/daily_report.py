import logging
from src.storage.db import Database
from src.notify.telegram_bot import TelegramNotifier

logger = logging.getLogger(__name__)

class DailyReporter:
    """
    Generates and sends the daily end-of-day summary.
    """
    
    def __init__(self, db: Database, notifier: TelegramNotifier):
        self.db = db
        self.notifier = notifier

    def generate_report(self) -> str:
        stats = self.db.get_daily_stats()
        
        pnl = stats.get('pnl', 0.0)
        trades = stats.get('trades', 0)
        wins = stats.get('wins', 0)
        losses = stats.get('losses', 0)
        win_rate = stats.get('win_rate', 0.0)
        
        emoji = "🚀" if pnl > 0 else ("📉" if pnl < 0 else "💤")
        
        report = f"📊 <b>DAILY SUMMARY</b> {emoji}\n\n"
        report += f"Daily P&L: <b>${pnl:.2f}</b>\n"
        report += f"Total Trades: {trades}\n"
        report += f"Wins: {wins} | Losses: {losses}\n"
        report += f"Win Rate: {win_rate:.1%}\n\n"
        
        report += "<i>System ready for the next day.</i>"
        return report

    def send_report(self):
        report = self.generate_report()
        logger.info("Sending daily summary to Telegram.")
        self.notifier.send_message(report)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    db = Database()
    notifier = TelegramNotifier(db)
    reporter = DailyReporter(db, notifier)
    print(reporter.generate_report())
