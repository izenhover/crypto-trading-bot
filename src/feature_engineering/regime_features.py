"""ویژگی‌های ورودی برای HMM تشخیص رژیم"""
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
