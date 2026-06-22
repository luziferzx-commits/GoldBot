import logging
import requests
import threading
import time
import yaml
from typing import Callable, Dict
from src.storage.db import Database

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
            self.token = settings['telegram']['bot_token']
            self.chat_id = settings['telegram']['chat_id']
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

    def send_message(self, text: str):
        if not self.token or not self.chat_id:
            logger.warning("Telegram token/chat_id missing. Cannot send message.")
            return
            
        url = f"{self.base_url}/sendMessage"
        payload = {
            "chat_id": self.chat_id,
            "text": text,
            "parse_mode": "HTML"
        }
        try:
            requests.post(url, json=payload, timeout=10)
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")

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
                            if cmd in self.command_handlers:
                                logger.info(f"Received Telegram command: {cmd}")
                                response_text = self.command_handlers[cmd]()
                                if response_text:
                                    self.send_message(response_text)
                            else:
                                self.send_message("Unknown command.")
            except Exception as e:
                logger.error(f"Telegram polling error: {e}")
                time.sleep(5)
            time.sleep(1)

    def start_polling(self):
        if not self.token:
            logger.warning("No token, skipping polling.")
            return
        self.is_running = True
        thread = threading.Thread(target=self._poll_updates, daemon=True)
        thread.start()
        logger.info("Telegram polling started.")

    def stop_polling(self):
        self.is_running = False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    notifier = TelegramNotifier()
    # If token and chat_id are missing, this will just log a warning
    notifier.send_message("Test message from GoldBot!")
