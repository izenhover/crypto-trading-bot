# src/signal_engine/event_detector.py
"""تشخیص رویدادهای هم‌راستایی با فیلتر روند و جهت"""
import pandas as pd
import numpy as np


def compute_trend(df: pd.DataFrame) -> dict:
    """محاسبه وضعیت روند بلندمدت با MA200"""
    if len(df) < 200:
        return {"ma200": None, "price_above_ma200": None, "trend": "UNKNOWN"}

    close = df["close"]
    ma200 = float(close.rolling(200).mean().iloc[-1])
    current = float(close.iloc[-1])

    return {
        "ma200": ma200,
        "price_above_ma200": current > ma200,
        "trend": "UP" if current > ma200 else "DOWN",
        "distance_pct": (current - ma200) / ma200 * 100,
    }


def check_trend_filter(trend_info: dict, direction: str) -> dict:
    """
    فیلتر روند با ناحیه بی‌طرف (buffer zone)
    - بیش از 2% بالای MA200: BUY مجاز، SELL ممنوع
    - بیش از 2% زیر MA200: SELL مجاز، BUY ممنوع
    - در بازه ±2%: هر دو جهت مجاز
    """
    if trend_info["trend"] == "UNKNOWN":
        return {"passed": False, "reason": "روند بلندمدت ناشناخته (داده کافی نیست)"}

    distance = trend_info["distance_pct"]
    BUFFER = 2.0

    # ناحیه بی‌طرف
    if abs(distance) < BUFFER:
        return {"passed": True,
                "reason": f"ناحیه بی‌طرف (فاصله {distance:+.2f}% از MA200)"}

    # روند صعودی قوی
    if distance >= BUFFER:
        if direction == "sell":
            return {"passed": False,
                    "reason": f"فروش ممنوع: قیمت {distance:+.2f}% بالای MA200"}
        return {"passed": True,
                "reason": f"خرید مجاز: روند صعودی (+{distance:.2f}%)"}

    # روند نزولی قوی
    if direction == "buy":
        return {"passed": False,
                "reason": f"خرید ممنوع: قیمت {distance:+.2f}% زیر MA200"}
    return {"passed": True,
            "reason": f"فروش مجاز: روند نزولی ({distance:.2f}%)"}


def check_direction_filter(df: pd.DataFrame, direction: str,
                            fast_period: int = 20) -> dict:
    """
    فیلتر جهت با MA سریع:
    - BUY فقط اگر قیمت بالای MA20
    - SELL فقط اگر قیمت زیر MA20
    """
    if len(df) < fast_period:
        return {"passed": False, "reason": "داده کافی نیست"}

    close = df["close"]
    ma20 = float(close.rolling(fast_period).mean().iloc[-1])
    current = float(close.iloc[-1])

    if direction == "buy":
        if current < ma20:
            return {"passed": False,
                    "reason": f"BUY رد شد: قیمت زیر MA20 (${current:.0f} < ${ma20:.0f})"}
        return {"passed": True, "reason": "قیمت بالای MA20"}
    else:
        if current > ma20:
            return {"passed": False,
                    "reason": f"SELL رد شد: قیمت بالای MA20 (${current:.0f} > ${ma20:.0f})"}
        return {"passed": True, "reason": "قیمت زیر MA20"}


def check_technical_confirmation(row: pd.Series, direction: str = "buy") -> dict:
    """بررسی تأیید تکنیکال"""
    checks = {}

    if direction == "buy":
        checks["rsi_ok"] = 30 < row.get("rsi_14", 50) < 70
        checks["macd_bullish"] = row.get("macd_hist", 0) > 0
        checks["not_overbought"] = row.get("bb_position", 0.5) < 0.85
        checks["above_bb_mid"] = row.get("close", 0) > row.get("bb_mid", 0)
    else:
        checks["rsi_ok"] = 30 < row.get("rsi_14", 50) < 70
        checks["macd_bearish"] = row.get("macd_hist", 0) < 0
        checks["not_oversold"] = row.get("bb_position", 0.5) > 0.15
        checks["below_bb_mid"] = row.get("close", 0) < row.get("bb_mid", 0)

    score = sum(checks.values()) / len(checks)
    return {"checks": checks, "score": score, "passed": score >= 0.75}


def check_volume_confirmation(row: pd.Series, min_ratio: float = 1.3) -> dict:
    """بررسی تأیید حجم"""
    ratio = row.get("volume_ratio_20", 1.0)
    return {"volume_ratio": float(ratio), "passed": ratio >= min_ratio}


def check_regime_confirmation(regime: str, direction: str) -> dict:
    """
    قوانین سخت‌گیرانه برای SELL، آسان‌گیرانه برای BUY
    (بازار کریپتو ذاتاً long bias دارد)
    """
    if direction == "buy":
        allowed = regime in ("BULL", "SIDEWAYS_LOW", "SIDEWAYS_HIGH", "BEAR")
        preferred = regime in ("BULL", "SIDEWAYS_LOW")
    else:  # sell
        allowed = regime in ("BEAR", "SIDEWAYS_HIGH")
        preferred = regime == "BEAR"

    return {
        "regime": regime,
        "allowed": allowed,
        "preferred": preferred,
        "passed": allowed,
    }


def check_flow_confirmation(anomaly_score: float, is_anomaly: bool,
                             volume_ratio: float, direction: str = "buy") -> dict:
    """بررسی تأیید جریان پول"""
    if is_anomaly and anomaly_score > 0.7:
        return {"flow_ok": False,
                "reason": f"ناهنجاری شدید (score={anomaly_score:.2f})",
                "passed": False}
    return {"flow_ok": True,
            "reason": "بدون هشدار جریان",
            "passed": True}