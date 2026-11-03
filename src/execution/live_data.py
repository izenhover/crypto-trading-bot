"""دریافت داده لحظه‌ای از صرافی‌های غیر بلاک‌شده"""
from datetime import datetime, timezone
import time
import pandas as pd
import ccxt
from loguru import logger

FALLBACK_EXCHANGES = ["kucoin", "okx", "bybit", "coinex", "gateio", "mexc"]


class LiveDataFetcher:
    """دریافت داده لحظه‌ای با تلاش چند صرافی"""

    def __init__(self, symbol: str = "BTC/USDT", timeframe: str = "1h"):
        self.symbol = symbol
        self.timeframe = timeframe
        self.exchange = None
        self.exchange_name = None
        self._init_exchange()

    def _init_exchange(self):
        errors = []
        for name in FALLBACK_EXCHANGES:
            try:
                if not hasattr(ccxt, name):
                    continue
                cls = getattr(ccxt, name)
                ex = cls({"enableRateLimit": True, "timeout": 20000})
                # تست سریع
                ohlcv = ex.fetch_ohlcv(self.symbol, self.timeframe, limit=5)
                if ohlcv and len(ohlcv) > 0:
                    self.exchange = ex
                    self.exchange_name = name
                    logger.info(f"Connected to {name}")
                    return
            except Exception as e:
                errors.append(f"{name}: {str(e)[:60]}")
                logger.debug(f"{name} failed: {e}")
        raise RuntimeError(
            f"Could not connect to any exchange. Tried: {FALLBACK_EXCHANGES}\n"
            + "\n".join(errors)
        )

    def fetch_recent(self, limit: int = 500) -> pd.DataFrame:
        ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=limit)
        if not ohlcv:
            raise RuntimeError("No data fetched")
        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        df = df[~df.index.duplicated(keep="last")].sort_index()
        return df

    def fetch_since(self, since_ts) -> pd.DataFrame:
        since_ms = int(pd.Timestamp(since_ts).timestamp() * 1000) + 1
        all_data = []
        while True:
            ohlcv = self.exchange.fetch_ohlcv(
                self.symbol, self.timeframe, since=since_ms, limit=1000
            )
            if not ohlcv:
                break
            all_data.extend(ohlcv)
            last_ts = ohlcv[-1][0]
            if last_ts == since_ms or len(ohlcv) < 1000:
                break
            since_ms = last_ts + 1
            time.sleep(0.2)
        if not all_data:
            return pd.DataFrame()
        df = pd.DataFrame(
            all_data, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        return df[~df.index.duplicated(keep="last")].sort_index()
