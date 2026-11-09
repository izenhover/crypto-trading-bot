# src/models/dataset.py
"""آماده‌سازی داده برای آموزش مدل‌ها"""
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

# ایمپورت ماژول ماکرو (به صورت امن)
try:
    from src.data_collection.macro_data import fetch_macro_data
    HAS_MACRO = True
except ImportError:
    HAS_MACRO = False
    logger.warning("ماژول macro_data در دسترس نیست.")

PROCESSED_DIR = Path("data/processed")
RAW_COLS = ["open", "high", "low", "close"]

def load_features(symbol: str, interval: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{symbol}_{interval}_features.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Features not found: {path}")
    df = pd.read_parquet(path)
    logger.info(f"Loaded {len(df)} rows, {df.shape[1]} cols from {path}")
    return df

def predict_regimes(symbol: str, interval: str, df: pd.DataFrame) -> pd.Series:
    try:
        from src.regime_detection import RegimeDetector
        detector = RegimeDetector.load()
        regimes = detector.predict(df)
        return regimes
    except Exception as e:
        logger.warning(f"Regime prediction failed: {e}")
        return None

def get_feature_columns(df: pd.DataFrame) -> list:
    exclude = set(RAW_COLS + ["regime", "target"])
    return [c for c in df.columns if c not in exclude]

def create_target(df: pd.DataFrame, horizon: int = 3, threshold: float = 0.001) -> pd.Series:
    future_close = df["close"].shift(-horizon)
    change = (future_close - df["close"]) / df["close"]

    if threshold > 0:
        target = pd.Series(np.nan, index=df.index)
        target[change > threshold] = 1
        target[change < -threshold] = 0
    else:
        target = (change > 0).astype(float)
        target[change.isna()] = np.nan

    target.name = "target"
    return target

def prepare_dataset(symbol: str, interval: str, horizon: int = 3, threshold: float = 0.001, return_regime: bool = True):
    df = load_features(symbol, interval)
    y = create_target(df, horizon=horizon, threshold=threshold)

    regimes_full = None
    if return_regime:
        regimes_full = predict_regimes(symbol, interval, df)

    valid = y.notna()
    df = df[valid]
    y = y[valid].astype(int)

    # ★★★ ادغام داده‌های کلان اقتصادی ★★★
    if HAS_MACRO:
        macro_df = fetch_macro_data(years=3)
        if not macro_df.empty:
            # هم‌ترازی تاریخ‌ها با ایندکس df
            macro_aligned = macro_df.reindex(df.index, method='ffill').fillna(method='bfill')
            # اضافه کردن به عنوان ستون جدید
            for col in macro_aligned.columns:
                df[f'macro_{col}'] = macro_aligned[col]
            logger.info(f"داده‌های کلان اضافه شدند: {list(macro_aligned.columns)}")

    feature_cols = get_feature_columns(df)
    X = df[feature_cols]

    regimes = None
    if regimes_full is not None:
        regimes = regimes_full.reindex(X.index)

    logger.info(
        f"Dataset: X={X.shape}, y={y.shape}, "
        f"class balance: up={y.mean()*100:.1f}%, down={(1-y.mean())*100:.1f}%"
    )
    if regimes is not None and not regimes.isna().all():
        logger.info(f"Regimes: {regimes.value_counts().to_dict()}")

    return X, y, feature_cols, regimes

def train_val_test_split(X, y, regimes=None, train_ratio=0.70, val_ratio=0.15):
    n = len(X)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    splits = {
        "train": (X.iloc[:train_end], y.iloc[:train_end]),
        "val": (X.iloc[train_end:val_end], y.iloc[train_end:val_end]),
        "test": (X.iloc[val_end:], y.iloc[val_end:]),
    }

    if regimes is not None:
        splits["train"] = (*splits["train"], regimes.iloc[:train_end])
        splits["val"] = (*splits["val"], regimes.iloc[train_end:val_end])
        splits["test"] = (*splits["test"], regimes.iloc[val_end:])

    logger.info(f"Split: train={len(splits['train'][0])}, val={len(splits['val'][0])}, test={len(splits['test'][0])}")
    return splits