# scripts/download_historical.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.data_collection.binance_public import BinancePublicData
from src.data_collection.storage import save_candles
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger


def main():
    setup_logger("download")
    config = load_config()
    symbols = config["exchange"]["symbols"]
    intervals = config["exchange"]["timeframes"]
    days = config["data"]["historical_days"]

    logger.info(f"Using Binance Public Data (no geo-restriction)")
    logger.info(f"Downloading {days} days for {symbols} @ {intervals}")

    for symbol in symbols:
        for interval in intervals:
            try:
                client = BinancePublicData(symbol, interval)
                df = client.fetch_historical(days=days)
                if df.empty:
                    logger.warning(f"No data for {symbol} {interval}")
                    continue
                save_candles(df, symbol, interval)
            except Exception as e:
                logger.error(f"Failed {symbol} {interval}: {e}")

    logger.info("Done!")


if __name__ == "__main__":
    main()