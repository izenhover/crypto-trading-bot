"""ابزارهای زمان"""
from datetime import datetime, timezone, timedelta


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def to_timestamp(dt: datetime) -> int:
    """تبدیل datetime به timestamp میلی‌ثانیه"""
    return int(dt.timestamp() * 1000)


def from_timestamp(ts: int) -> datetime:
    """تبدیل timestamp میلی‌ثانیه به datetime"""
    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc)


def align_to_candle(dt: datetime, interval_minutes: int) -> datetime:
    """هم‌ترازی زمان با شروع کندل"""
    minutes = (dt.minute // interval_minutes) * interval_minutes
    return dt.replace(minute=minutes, second=0, microsecond=0)


def is_candle_closed(dt: datetime, interval_minutes: int) -> bool:
    """بررسی بسته شدن کندل"""
    next_candle = align_to_candle(dt, interval_minutes) + timedelta(minutes=interval_minutes)
    return dt >= next_candle
