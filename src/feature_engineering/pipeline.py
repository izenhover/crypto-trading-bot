"""خط لوله ساخت ویژگی‌ها"""
from pathlib import Path
import pandas as pd
from loguru import logger

from src.data_collection.storage import load_candles
from src.feature_engineering.technical import add_all_technical
from src.feature_engineering.regime_features import build_regime_features
from src.utils.config_loader import load_config

PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def get_processed_path(symbol: str, interval: str) -> Path:
    return PROCESSED_DIR / f"{symbol}_{interval}_features.parquet"


def build_features_for(symbol: str, interval: str, config: dict) -> pd.DataFrame:
    """ساخت ویژگی‌ها برای یک جفت نماد/تایم‌فریم"""
    raw = load_candles(symbol, interval)
    if raw is None or raw.empty:
        logger.warning(f"No raw data for {symbol} {interval}")
        return None

    df = raw.copy()
    df = add_all_technical(df, config)

    # حذف ردیف‌هایی که هنوز NaN دارند (بازه گرم‌شدن اندیکاتورها)
    df = df.replace([float("inf"), float("-inf")], pd.NA)
    df = df.dropna()

    if df.empty:
        logger.warning(f"All rows dropped after feature engineering for {symbol} {interval}")
        return None

    # ذخیره
    out_path = get_processed_path(symbol, interval)
    df.to_parquet(out_path)
    logger.info(
        f"Saved {len(df)} rows, {df.shape[1]} features -> {out_path}"
    )
    return df


def build_all(config: dict = None):
    """ساخت ویژگی‌ها برای همه نمادها و تایم‌فریم‌ها"""
    config = config or load_config()
    symbols = config["exchange"]["symbols"]
    intervals = config["exchange"]["timeframes"]

    summary = {}
    for symbol in symbols:
        for interval in intervals:
            try:
                df = build_features_for(symbol, interval, config)
                if df is not None:
                    summary[f"{symbol}_{interval}"] = len(df)
            except Exception as e:
                logger.error(f"Failed {symbol} {interval}: {e}")

    logger.info("=" * 50)
    logger.info("Feature engineering summary:")
    for k, v in summary.items():
        logger.info(f"  {k}: {v} rows")
    logger.info("=" * 50)
    return summary
