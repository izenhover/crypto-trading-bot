"""تست دریافت داده لحظه‌ای به مدت ۶۰ ثانیه"""
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
