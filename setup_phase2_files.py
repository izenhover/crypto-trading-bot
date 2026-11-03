# setup_phase2_files.py
"""ساخت فایل‌های فاز ۲: مهندسی ویژگی"""
from pathlib import Path

FILES = {}

# ---------- src/feature_engineering/technical.py ----------
FILES["src/feature_engineering/technical.py"] = '''"""محاسبه اندیکاتورهای تکنیکال"""
import numpy as np
import pandas as pd


def add_rsi(df: pd.DataFrame, period: int = 14, col: str = "close") -> pd.DataFrame:
    """شاخص قدرت نسبی (RSI)"""
    delta = df[col].diff()
    gain = delta.where(delta > 0, 0.0).rolling(window=period, min_periods=period).mean()
    loss = -delta.where(delta < 0, 0.0).rolling(window=period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    df[f"rsi_{period}"] = 100 - (100 / (1 + rs))
    return df


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9, col: str = "close") -> pd.DataFrame:
    """MACD + Signal + Histogram"""
    ema_fast = df[col].ewm(span=fast, adjust=False).mean()
    ema_slow = df[col].ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    df["macd"] = macd
    df["macd_signal"] = macd_signal
    df["macd_hist"] = macd - macd_signal
    return df


def add_bollinger(df: pd.DataFrame, period: int = 20, std_mult: float = 2.0, col: str = "close") -> pd.DataFrame:
    """باندهای بولینگر و موقعیت قیمت نسبت به باندها"""
    mid = df[col].rolling(period).mean()
    std = df[col].rolling(period).std()
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    df["bb_mid"] = mid
    df["bb_upper"] = upper
    df["bb_lower"] = lower
    df["bb_width"] = (upper - lower) / mid.replace(0, np.nan)
    df["bb_position"] = (df[col] - lower) / (upper - lower).replace(0, np.nan)
    return df


def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """میانگین دامنه واقعی (ATR)"""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    df[f"atr_{period}"] = tr.rolling(window=period, min_periods=period).mean()
    df[f"atr_{period}_pct"] = df[f"atr_{period}"] / close * 100
    return df


def add_volume_features(df: pd.DataFrame, windows: list = [5, 20, 60]) -> pd.DataFrame:
    """ویژگی‌های حجم"""
    for w in windows:
        ma = df["volume"].rolling(w).mean()
        df[f"volume_sma_{w}"] = ma
        df[f"volume_ratio_{w}"] = df["volume"] / ma.replace(0, np.nan)
    return df


def add_returns(df: pd.DataFrame, windows: list = [1, 3, 5, 10, 20, 60], col: str = "close") -> pd.DataFrame:
    """بازدهی در بازه‌های مختلف"""
    for w in windows:
        df[f"return_{w}"] = df[col].pct_change(w)
    return df


def add_volatility(df: pd.DataFrame, windows: list = [10, 20, 60], col: str = "close") -> pd.DataFrame:
    """نوسان (انحراف معیار بازدهی) در بازه‌های مختلف"""
    ret = df[col].pct_change()
    for w in windows:
        df[f"volatility_{w}"] = ret.rolling(w).std()
    return df


def add_candle_features(df: pd.DataFrame) -> pd.DataFrame:
    """ویژگی‌های ساختار کندل"""
    body = df["close"] - df["open"]
    range_ = (df["high"] - df["low"]).replace(0, np.nan)
    df["candle_body"] = body
    df["candle_range"] = range_
    df["candle_body_pct"] = body / range_
    df["candle_upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / range_
    df["candle_lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / range_
    return df


def add_all_technical(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """همه ویژگی‌های تکنیکال را با تنظیمات config اضافه می‌کند"""
    t = config["features"]["technical"]
    df = add_rsi(df, period=t["rsi_period"])
    df = add_macd(df, fast=t["macd_fast"], slow=t["macd_slow"], signal=t["macd_signal"])
    df = add_bollinger(df, period=t["bb_period"], std_mult=t["bb_std"])
    df = add_atr(df, period=t["atr_period"])
    df = add_volume_features(df)
    df = add_returns(df)
    df = add_volatility(df)
    df = add_candle_features(df)
    return df
'''

# ---------- src/feature_engineering/regime_features.py ----------
FILES["src/feature_engineering/regime_features.py"] = '''"""ویژگی‌های ورودی برای HMM تشخیص رژیم"""
import numpy as np
import pandas as pd


def build_regime_features(df: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """سه ویژگی استاندارد برای HMM: بازدهی، نوسان، تغییر حجم"""
    returns = df["close"].pct_change()
    volatility = returns.rolling(lookback).std()
    volume_change = df["volume"].pct_change().replace([np.inf, -np.inf], np.nan)

    features = pd.DataFrame({
        "returns": returns,
        "volatility": volatility,
        "volume_change": volume_change,
    }, index=df.index)

    return features.dropna()
'''

# ---------- src/feature_engineering/pipeline.py ----------
FILES["src/feature_engineering/pipeline.py"] = '''"""خط لوله ساخت ویژگی‌ها"""
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
'''

# ---------- src/feature_engineering/__init__.py ----------
FILES["src/feature_engineering/__init__.py"] = '''from src.feature_engineering.technical import add_all_technical
from src.feature_engineering.regime_features import build_regime_features
from src.feature_engineering.pipeline import build_all, build_features_for

__all__ = [
    "add_all_technical",
    "build_regime_features",
    "build_all",
    "build_features_for",
]
'''

# ---------- scripts/build_features.py ----------
FILES["scripts/build_features.py"] = '''"""اسکریپت ساخت ویژگی‌ها"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_engineering.pipeline import build_all
from src.utils.logger import setup_logger


def main():
    setup_logger("features")
    build_all()


if __name__ == "__main__":
    main()
'''

# ---------- tests/test_features.py ----------
FILES["tests/test_features.py"] = '''"""تست ویژگی‌های ساخته‌شده"""
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
    print("  1. python scripts/build_features.py")
    print("  2. python tests/test_features.py")
    print()


if __name__ == "__main__":
    main()