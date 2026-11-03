"""محاسبه اندیکاتورهای تکنیکال"""
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
