"""محاسبه حد ضرر و حد سود مبتنی بر ATR"""
from typing import Optional


def atr_stop_loss(
    entry_price: float,
    direction: str,
    atr: float,
    multiplier: float = 1.5,
) -> float:
    """حد ضرر مبتنی بر ATR"""
    if direction.upper() == "BUY":
        return entry_price - multiplier * atr
    return entry_price + multiplier * atr


def atr_take_profit(
    entry_price: float,
    direction: str,
    atr: float,
    multiplier: float = 3.0,
) -> float:
    """حد سود مبتنی بر ATR"""
    if direction.upper() == "BUY":
        return entry_price + multiplier * atr
    return entry_price - multiplier * atr


def trailing_stop(
    current_price: float,
    highest_price: float,
    lowest_price: float,
    direction: str,
    atr: float,
    multiplier: float = 1.5,
) -> float:
    """
    حد ضرر متحرک: در پوزیشن خرید، حد ضرر را بالا می‌برد
    """
    if direction.upper() == "BUY":
        return max(highest_price - multiplier * atr, current_price - 3 * atr)
    return min(lowest_price + multiplier * atr, current_price + 3 * atr)


def risk_reward_ratio(
    entry: float, stop: float, target: float
) -> Optional[float]:
    """نسبت ریسک به ریوارد"""
    risk = abs(entry - stop)
    reward = abs(target - entry)
    if risk <= 0:
        return None
    return round(reward / risk, 2)
