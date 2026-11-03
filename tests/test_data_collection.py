"""تست خط لوله داده"""
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
