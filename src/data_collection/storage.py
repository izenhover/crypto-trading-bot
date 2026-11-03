"""ذخیره و بارگذاری کندل‌ها در فایل‌های Parquet"""
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
