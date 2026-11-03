"""کلاینت عمومی صرافی با ccxt"""
from datetime import datetime, timezone, timedelta
import pandas as pd
import ccxt
from loguru import logger


class ExchangeClient:
    def __init__(self, exchange_name: str = "binance"):
        if not hasattr(ccxt, exchange_name):
            raise ValueError(f"Unknown exchange: {exchange_name}")
        self.exchange_name = exchange_name
        cls = getattr(ccxt, exchange_name)
        self.exchange = cls({"enableRateLimit": True, "timeout": 30000})
        logger.info(f"Initialized {exchange_name} client")

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=1000):
        try:
            return self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
        except Exception as e:
            logger.error(f"fetch_ohlcv error: {e}")
            raise

    def fetch_historical(self, symbol: str, timeframe: str, days: int = 180) -> pd.DataFrame:
        """دریافت داده تاریخی در بلوک‌های ۱۰۰۰ کندلی"""
        since_dt = datetime.now(timezone.utc) - timedelta(days=days)
        since = int(since_dt.timestamp() * 1000)
        all_candles = []
        logger.info(f"Fetching {symbol} {timeframe} for {days} days...")

        while True:
            try:
                candles = self.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            except Exception as e:
                logger.error(f"Error at since={since}: {e}")
                break

            if not candles:
                break

            all_candles.extend(candles)
            last_ts = candles[-1][0]

            if last_ts == since or len(candles) < 1000:
                break

            since = last_ts + 1
            logger.info(
                f"  Total {len(all_candles)} candles | last: "
                f"{datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)}"
            )

        if not all_candles:
            return pd.DataFrame()

        df = pd.DataFrame(
            all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp")
        df = df[~df.index.duplicated(keep="last")].sort_index()
        logger.info(f"Fetched {len(df)} candles for {symbol} {timeframe}")
        return df
