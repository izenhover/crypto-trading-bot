# setup_phase1_files.py
"""ساخت فایل‌های فاز ۱: خط لوله داده"""
from pathlib import Path

FILES = {}

# ---------- src/data_collection/storage.py ----------
FILES["src/data_collection/storage.py"] = '''"""ذخیره و بارگذاری کندل‌ها در فایل‌های Parquet"""
from pathlib import Path
import pandas as pd
from loguru import logger

DATA_DIR = Path("data/raw")
DATA_DIR.mkdir(parents=True, exist_ok=True)


def get_parquet_path(symbol: str, interval: str) -> Path:
    return DATA_DIR / f"{symbol}_{interval}.parquet"


def save_candles(df: pd.DataFrame, symbol: str, interval: str) -> int:
    """ذخیره کندل‌ها با ادغام در فایل موجود"""
    path = get_parquet_path(symbol, interval)
    if path.exists():
        existing = pd.read_parquet(path)
        combined = pd.concat([existing, df])
        combined = combined[~combined.index.duplicated(keep="last")].sort_index()
    else:
        combined = df
    combined.to_parquet(path)
    logger.info(f"Saved {len(combined)} candles -> {path}")
    return len(combined)


def load_candles(symbol: str, interval: str) -> pd.DataFrame:
    path = get_parquet_path(symbol, interval)
    if not path.exists():
        logger.warning(f"No data file: {path}")
        return None
    return pd.read_parquet(path)
'''

# ---------- src/data_collection/exchange_client.py ----------
FILES["src/data_collection/exchange_client.py"] = '''"""کلاینت عمومی صرافی با ccxt"""
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
'''

# ---------- src/data_collection/historical.py ----------
FILES["src/data_collection/historical.py"] = '''"""دریافت و ذخیره داده تاریخی"""
from src.data_collection.exchange_client import ExchangeClient
from src.data_collection.storage import save_candles, load_candles
from loguru import logger


def download_historical_data(
    symbol: str,
    intervals: list,
    days: int = 180,
    exchange_name: str = "binance",
):
    """دریافت داده تاریخی برای یک نماد در چند تایم‌فریم"""
    client = ExchangeClient(exchange_name)
    results = {}

    for interval in intervals:
        logger.info(f"=== {symbol} {interval} ===")
        try:
            df = client.fetch_historical(symbol, interval, days=days)
            if df.empty:
                logger.warning(f"No data for {symbol} {interval}")
                continue
            save_candles(df, symbol, interval)
            results[interval] = len(df)
        except Exception as e:
            logger.error(f"Failed {symbol} {interval}: {e}")

    return results
'''

# ---------- src/data_collection/binance_stream.py ----------
FILES["src/data_collection/binance_stream.py"] = '''"""دریافت داده لحظه‌ای از Binance WebSocket"""
import json
import asyncio
from datetime import datetime, timezone
from typing import Callable, Optional
import websockets
from loguru import logger


BINANCE_WS_BASE = "wss://stream.binance.com:9443/ws"
BINANCE_WS_TESTNET = "wss://stream.testnet.binance.vision/ws"


class BinanceStreamer:
    """جریان داده لحظه‌ای Binance"""

    def __init__(self, symbols: list, intervals: list, use_testnet: bool = False):
        self.symbols = [s.lower() for s in symbols]
        self.intervals = intervals
        self.base_url = BINANCE_WS_TESTNET if use_testnet else BINANCE_WS_BASE
        self.callback: Optional[Callable] = None
        self._running = False

    def _build_stream_names(self):
        names = []
        for sym in self.symbols:
            for interval in self.intervals:
                names.append(f"{sym}@kline_{interval}")
        return names

    async def _listen_single(self, stream_name: str):
        url = f"{self.base_url}/{stream_name}"
        while self._running:
            try:
                async with websockets.connect(url, ping_interval=20) as ws:
                    logger.info(f"Connected: {stream_name}")
                    async for msg in ws:
                        if not self._running:
                            break
                        data = json.loads(msg)
                        if self.callback:
                            self.callback(data)
            except Exception as e:
                logger.error(f"Stream error {stream_name}: {e}")
                await asyncio.sleep(5)

    async def _run_all(self):
        tasks = [self._listen_single(name) for name in self._build_stream_names()]
        await asyncio.gather(*tasks)

    def start(self, callback: Callable, duration_seconds: Optional[int] = None):
        """شروع جریان داده. اگر duration بدهید، بعد از آن متوقف می‌شود."""
        self.callback = callback
        self._running = True

        async def _main():
            if duration_seconds:
                try:
                    await asyncio.wait_for(self._run_all(), timeout=duration_seconds)
                except asyncio.TimeoutError:
                    logger.info(f"Stream stopped after {duration_seconds}s")
            else:
                await self._run_all()

        try:
            asyncio.run(_main())
        except KeyboardInterrupt:
            logger.info("Stream interrupted by user")
        finally:
            self._running = False
'''

# ---------- src/data_collection/scheduler.py ----------
FILES["src/data_collection/scheduler.py"] = '''"""زمان‌بند به‌روزرسانی داده"""
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger


def create_scheduler():
    return BlockingScheduler(timezone="UTC")


def add_historical_refresh_job(scheduler, job_func, interval_minutes=30):
    """هر N دقیقه داده تاریخی را به‌روزرسانی کن"""
    scheduler.add_job(
        job_func,
        trigger=IntervalTrigger(minutes=interval_minutes),
        id="historical_refresh",
        replace_existing=True,
        max_instances=1,
    )
    logger.info(f"Scheduled historical refresh every {interval_minutes} min")
'''

# ---------- src/data_collection/__init__.py ----------
FILES["src/data_collection/__init__.py"] = '''from src.data_collection.exchange_client import ExchangeClient
from src.data_collection.historical import download_historical_data
from src.data_collection.storage import save_candles, load_candles
from src.data_collection.binance_stream import BinanceStreamer

__all__ = [
    "ExchangeClient",
    "download_historical_data",
    "save_candles",
    "load_candles",
    "BinanceStreamer",
]
'''

# ---------- scripts/download_historical.py ----------
FILES["scripts/download_historical.py"] = '''"""اسکریپت دانلود داده تاریخی"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.data_collection.historical import download_historical_data
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger


def main():
    setup_logger("download")
    config = load_config()
    symbols = config["exchange"]["symbols"]
    intervals = config["exchange"]["timeframes"]
    days = config["data"]["historical_days"]
    exchange_name = config["exchange"]["name"]

    logger.info(f"Downloading {days} days for {symbols} @ {intervals}")

    for symbol in symbols:
        try:
            result = download_historical_data(symbol, intervals, days=days, exchange_name=exchange_name)
            logger.info(f"{symbol}: {result}")
        except Exception as e:
            logger.error(f"Failed {symbol}: {e}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
'''

# ---------- scripts/test_stream.py ----------
FILES["scripts/test_stream.py"] = '''"""تست دریافت داده لحظه‌ای به مدت ۶۰ ثانیه"""
import sys
from pathlib import Path
from collections import Counter
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.data_collection.binance_stream import BinanceStreamer
from src.utils.logger import setup_logger


def main():
    setup_logger("stream")

    symbols = ["BTCUSDT"]
    intervals = ["1m", "5m"]
    duration = 60

    logger.info(f"Testing stream for {duration}s on {symbols} {intervals}")

    msg_count = Counter()
    last_prices = {}

    def on_message(data):
        try:
            kline = data.get("k", {})
            symbol = data.get("s", "?")
            interval = kline.get("i", "?")
            close = kline.get("c", "?")
            is_closed = kline.get("x", False)
            key = f"{symbol}_{interval}"
            msg_count[key] += 1
            last_prices[key] = close
            if is_closed:
                logger.info(f"[CLOSED] {key} close={close}")
        except Exception as e:
            logger.error(f"Parse error: {e}")

    streamer = BinanceStreamer(symbols, intervals, use_testnet=False)
    streamer.start(callback=on_message, duration_seconds=duration)

    logger.info("=" * 50)
    logger.info("Stream test summary:")
    for key, count in msg_count.items():
        logger.info(f"  {key}: {count} messages, last close = {last_prices.get(key)}")
    if not msg_count:
        logger.warning("No messages received! Check network/geo-restriction.")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
'''

# ---------- tests/test_data_collection.py ----------
FILES["tests/test_data_collection.py"] = '''"""تست خط لوله داده"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_collection.storage import load_candles, get_parquet_path
from src.utils.logger import setup_logger

setup_logger("test")


def test_files_exist():
    for symbol in ["BTCUSDT", "ETHUSDT"]:
        for interval in ["1m", "5m", "30m", "1h", "4h"]:
            path = get_parquet_path(symbol, interval)
            if path.exists():
                df = load_candles(symbol, interval)
                print(f"[OK] {symbol} {interval}: {len(df)} rows, "
                      f"{df.index.min()} -> {df.index.max()}")
            else:
                print(f"[MISSING] {symbol} {interval}")


if __name__ == "__main__":
    test_files_exist()
'''


def main():
    root = Path(".")
    created = 0
    for file_path, content in FILES.items():
        path = root / file_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        created += 1
        print(f"  + {file_path}")
    print()
    print("=" * 50)
    print(f"Created {created} files successfully!")
    print("=" * 50)
    print()
    print("Next steps:")
    print("  1. python scripts/download_historical.py")
    print("  2. python tests/test_data_collection.py")
    print("  3. python scripts/test_stream.py")
    print()


if __name__ == "__main__":
    main()