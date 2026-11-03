"""ارسال هشدار به تلگرام (اختیاری)"""
import os
import requests
from loguru import logger


class TelegramAlerter:
    def __init__(self, bot_token: str = None, chat_id: str = None):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.enabled = (
            self.bot_token
            and self.chat_id
            and self.bot_token != "your_bot_token_here"
        )
        if self.enabled:
            logger.info("Telegram alerter enabled")
        else:
            logger.info("Telegram alerter disabled (no credentials)")

    def send(self, message: str):
        if not self.enabled:
            logger.debug(f"[TELEGRAM-OFF] {message}")
            return False
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            resp = requests.post(
                url,
                json={"chat_id": self.chat_id, "text": message, "parse_mode": "Markdown"},
                timeout=10,
            )
            return resp.ok
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    def send_signal(self, signal):
        emoji = "🟢" if signal.signal.value == "BUY" else "🔴" if signal.signal.value == "SELL" else "⚪"
        text = (
            f"{emoji} *{signal.signal.value}* BTCUSDT\n"
            f"قیمت: `{signal.current_price:.2f}`\n"
            f"اطمینان: `{signal.confidence*100:.1f}%` ({signal.confidence_level.value})\n"
            f"رژیم: `{signal.regime}`\n"
            f"SL: `{signal.stop_loss:.2f}` | TP: `{signal.take_profit:.2f}`\n"
            f"تأییدها: `{signal.confirmations}/4`"
        )
        return self.send(text)
