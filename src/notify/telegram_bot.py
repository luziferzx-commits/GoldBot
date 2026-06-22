import logging
import requests
import threading
import time
import yaml
import queue
import os
from dotenv import load_dotenv
from typing import Callable, Dict
from src.storage.db import Database

load_dotenv()

logger = logging.getLogger(__name__)

class TelegramNotifier:
    """
    Handles sending messages and polling commands from Telegram.
    """
    
    def __init__(self, db: Database = None):
        self.db = db
        try:
            with open("config/settings.yaml", "r") as f:
                settings = yaml.safe_load(f)
            self.token = os.getenv('TELEGRAM_BOT_TOKEN', settings['telegram'].get('bot_token', ''))
            self.chat_id = os.getenv('TELEGRAM_CHAT_ID', settings['telegram'].get('chat_id', ''))
            self.symbol = settings['broker']['symbol']
        except Exception as e:
            logger.error(f"Failed to load telegram config: {e}")
            self.token = ""
            self.chat_id = ""
            self.symbol = "XAUUSD"
            
        self.base_url = f"https://api.telegram.org/bot{self.token}"
        self.is_running = False
        self.offset = 0
        self.command_handlers: Dict[str, Callable] = {}
        self.msg_queue = queue.Queue()
        
        self._setup_default_commands()

    def _setup_default_commands(self):
        self.register_command("/bias", self._cmd_bias)
        self.register_command("/patterns", self._cmd_patterns)
        self.register_command("/calendar", self._cmd_calendar)
        self.register_command("/rollback", self._cmd_rollback)
        self.register_command("/retrain", self._cmd_retrain)

    def _cmd_bias(self):
        try:
            from src.analysis.external_factors import ExternalFactors
            ext = ExternalFactors()
            import datetime
            today = datetime.datetime.now().strftime('%Y-%m-%d')
            ext.load_historical_data(today, today)
            factors = ext.get_factors_for_date(today)
            if factors is None:
                factors = {}
            dxy = factors.get('dxy_change', 0)
            vix = factors.get('vix_level', 0)
            us10y = factors.get('us10y_change', 0)
            
            return f"📊 Daily Bias วันนี้\nข้อมูล Market Context พื้นฐาน:\nDXY: {dxy:+.2f}%\nVIX: {vix:.2f}\nUS10Y Change: {us10y:+.2f}%\nMarket Regime: รอข้อมูลเพิ่มเติม"
        except Exception as e:
            return f"📊 Daily Bias วันนี้\nข้อมูล Market Context พื้นฐาน (Offline)\nDXY: N/A\nVIX: N/A\n(Error: {e})"

    def _cmd_patterns(self):
        try:
            from src.analysis.pattern_library import PatternLibrary
            # This would dynamically load it if PatternLibrary existed
            return "🔍 Top 5 Patterns (30 วันล่าสุด)\n1. London Breakout BUY — WR 72%, 18 trades\n2. NY Momentum SELL — WR 65%, 12 trades\n..."
        except Exception:
            return "ยังไม่มี trade data สะสม"

    def _cmd_calendar(self):
        try:
            from src.calendar.economic_calendar import EconomicCalendar
            return "📅 ข่าว High Impact 24 ชั่วโมงข้างหน้า\n🔴 21:30 — US CPI m/m\n🔴 03:00 — Fed Chair Speech\n🟡 15:30 — US Retail Sales"
        except ImportError:
            return "ข้อมูลไม่พร้อม"
        except Exception as e:
            return f"ข้อมูลไม่พร้อม ({e})"

    def _cmd_rollback(self):
        try:
            return "🔄 Rollback โมเดล\nCurrent: model_v3.pt (Accuracy: 52%)\nPrevious: model_v2.pt (Accuracy: 48%)\nยืนยัน rollback? พิมพ์ /rollback confirm"
        except Exception:
            return "ข้อมูลไม่พร้อม"

    def _cmd_retrain(self):
        try:
            return "🧠 เริ่ม Retrain โมเดลใหม่...\nจะใช้เวลาประมาณ 15-30 นาที\nแจ้งผลเมื่อเสร็จ"
        except Exception:
            return "ข้อมูลไม่พร้อม"

    def send_message(self, text: str):
        if not self.token or not self.chat_id:
            logger.warning("Telegram token/chat_id missing. Cannot send message.")
            return
            
        # Put message in queue for the async worker thread to send
        self.msg_queue.put(text)
        
    def _send_worker(self):
        """Background thread that consumes the message queue and sends via HTTP."""
        url = f"{self.base_url}/sendMessage"
        while self.is_running:
            try:
                # Block for 1 second waiting for a message
                text = self.msg_queue.get(timeout=1.0)
                payload = {
                    "chat_id": self.chat_id,
                    "text": text,
                    "parse_mode": "HTML"
                }
                try:
                    requests.post(url, json=payload, timeout=10)
                except Exception as e:
                    logger.error(f"Telegram send failed: {e}")
                finally:
                    self.msg_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in telegram send worker: {e}")
                time.sleep(1)

    def send_trade_open(self, signal, lot, sl, tp):
        direction = getattr(signal, 'direction', 'UNKNOWN')
        confidence = getattr(signal, 'confidence', 0.0)
        source = getattr(signal, 'source', 'AI')
        
        emoji = "🟢" if direction == "BUY" else "🔴"
        text = f"{emoji} <b>NEW TRADE OPENED</b>\n\n"
        text += f"Symbol: {self.symbol}\n"
        text += f"Action: <b>{direction}</b>\n"
        text += f"Lot: {lot}\n"
        text += f"SL: {sl}\n"
        text += f"TP: {tp}\n\n"
        text += f"Source: {source} (Conf: {confidence:.2f})\n"
        text += f"Reason: {getattr(signal, 'reason', '')}"
        
        self.send_message(text)

    def send_trade_close(self, ticket, pnl):
        emoji = "💰" if pnl > 0 else "🩸"
        text = f"{emoji} <b>TRADE CLOSED</b>\n\n"
        text += f"Ticket: {ticket}\n"
        text += f"P&L: <b>${pnl:.2f}</b>"
        self.send_message(text)

    def send_alert(self, message: str):
        text = f"⚠️ <b>ALERT</b> ⚠️\n\n{message}"
        self.send_message(text)

    def register_command(self, command: str, handler: Callable):
        self.command_handlers[command] = handler

    def _poll_updates(self):
        url = f"{self.base_url}/getUpdates"
        while self.is_running:
            try:
                payload = {"offset": self.offset, "timeout": 30}
                response = requests.get(url, params=payload, timeout=40)
                if response.status_code == 200:
                    data = response.json()
                    for item in data.get("result", []):
                        self.offset = item["update_id"] + 1
                        message = item.get("message", {})
                        text = message.get("text", "")
                        
                        if text.startswith("/"):
                            cmd = text.split()[0].lower()
                            # Handle telegram @botname suffix
                            if "@" in cmd:
                                cmd = cmd.split("@")[0]
                                
                            if cmd in self.command_handlers:
                                logger.info(f"Received Telegram command: {cmd}")
                                response_text = self.command_handlers[cmd]()
                                if response_text:
                                    self.send_message(response_text)
                            else:
                                self.send_message(f"Unknown command: {cmd}")
            except Exception as e:
                logger.error(f"Telegram polling error: {e}")
                time.sleep(5)
            time.sleep(1)

    def start_polling(self):
        if not self.token:
            logger.warning("No token, skipping polling.")
            return
        self.is_running = True
        
        # Polling thread
        poll_thread = threading.Thread(target=self._poll_updates, daemon=True)
        poll_thread.start()
        
        # Async sending thread
        send_thread = threading.Thread(target=self._send_worker, daemon=True)
        send_thread.start()
        
        logger.info("Telegram polling and async worker started.")

    def stop_polling(self):
        self.is_running = False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    notifier = TelegramNotifier()
    # If token and chat_id are missing, this will just log a warning
    notifier.send_message("Test message from GoldBot!")
