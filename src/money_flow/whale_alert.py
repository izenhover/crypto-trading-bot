"""اتصال به Whale Alert API (اختیاری) + پروکسی حجمی"""
import os
from datetime import datetime, timezone, timedelta
from typing import Optional
import requests
from loguru import logger

WHALE_ALERT_URL = "https://api.whale-alert.io/v1/transactions"


class WhaleAlertClient:
    """
    کلاینت Whale Alert.
    اگر کلید API نباشد، به حالت «آفلاین» می‌رود و از داده‌های حجمی محلی استفاده می‌کند.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("WHALE_ALERT_API_KEY")
        self.online = bool(self.api_key and self.api_key != "your_whale_alert_key_here")
        if self.online:
            logger.info("Whale Alert: online mode")
        else:
            logger.warning("Whale Alert: offline mode (no API key)")

    def fetch_transactions(
        self,
        min_value_usd: int = 1_000_000,
        hours_back: int = 1,
        currency: str = "btc",
    ) -> list:
        """دریافت تراکنش‌های بزرگ از API (اگر آنلاین باشد)"""
        if not self.online:
            return []

        start = int((datetime.now(timezone.utc) - timedelta(hours=hours_back)).timestamp())
        params = {
            "api_key": self.api_key,
            "min_value": min_value_usd,
            "start": start,
            "currency": currency,
        }
        try:
            resp = requests.get(WHALE_ALERT_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            transactions = data.get("transactions", [])
            logger.info(f"Whale Alert: fetched {len(transactions)} transactions")
            return transactions
        except Exception as e:
            logger.error(f"Whale Alert error: {e}")
            return []

    def summarize_flow(self, transactions: list) -> dict:
        """خلاصه جریان پول از تراکنش‌ها"""
        inflow = 0.0
        outflow = 0.0
        n = 0
        max_tx = 0.0
        for tx in transactions:
            amount = float(tx.get("amount_usd", 0))
            tx_type = tx.get("transaction_type", "")
            n += 1
            max_tx = max(max_tx, amount)
            if "inflow" in tx_type:
                inflow += amount
            elif "outflow" in tx_type:
                outflow += amount
        return {
            "inflow_usd": inflow,
            "outflow_usd": outflow,
            "net_flow_usd": inflow - outflow,
            "tx_count": n,
            "max_tx_usd": max_tx,
        }


def volume_proxy_flow(df) -> dict:
    """
    پروکسی جریان پول از داده‌های حجمی.
    حجم غیرعادی بالا = احتمال حرکت نهنگ
    """
    if len(df) < 60:
        return {"volume_ratio": 1.0, "is_spike": False, "spike_size": 0.0}

    latest_vol = float(df["volume"].iloc[-1])
    rolling_mean = float(df["volume"].iloc[-60:].mean())
    rolling_std = float(df["volume"].iloc[-60:].std())

    ratio = latest_vol / rolling_mean if rolling_mean > 0 else 1.0
    z_score = (latest_vol - rolling_mean) / rolling_std if rolling_std > 0 else 0.0
    is_spike = ratio > 2.5 or z_score > 3.0

    return {
        "volume_ratio": ratio,
        "volume_z_score": z_score,
        "is_spike": is_spike,
        "spike_size": max(0.0, ratio - 1.0),
    }
