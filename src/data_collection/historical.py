"""دریافت و ذخیره داده تاریخی"""
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
