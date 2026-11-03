"""تست ویژگی‌های ساخته‌شده"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.feature_engineering.pipeline import get_processed_path
from src.utils.logger import setup_logger

setup_logger("test")


def test_features():
    print()
    print("=" * 60)
    print("Testing Feature Engineering Output")
    print("=" * 60)

    for symbol in ["BTCUSDT", "ETHUSDT"]:
        for interval in ["1m", "5m", "30m", "1h", "4h"]:
            path = get_processed_path(symbol, interval)
            if not path.exists():
                print(f"[MISSING] {symbol} {interval}")
                continue

            df = pd.read_parquet(path)
            nan_cols = df.isna().any().sum()
            print(
                f"[OK] {symbol} {interval}: {len(df):>7} rows, "
                f"{df.shape[1]:>3} cols, "
                f"NaN cols: {nan_cols}, "
                f"range: {df.index.min()} -> {df.index.max()}"
            )

    # نمایش لیست ویژگی‌ها برای یک نمونه
    print()
    print("-" * 60)
    print("Sample feature columns (BTCUSDT 1h):")
    print("-" * 60)
    df = pd.read_parquet(get_processed_path("BTCUSDT", "1h"))
    for i, col in enumerate(df.columns, 1):
        print(f"  {i:>3}. {col}")


if __name__ == "__main__":
    test_features()
