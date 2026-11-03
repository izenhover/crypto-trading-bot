# src/risk_management/position_sizing.py
"""محاسبه اندازه موقعیت با Kelly کسری + محدودیت‌های ایمنی"""
from loguru import logger


def fractional_kelly(
    win_rate: float,
    win_loss_ratio: float,
    fraction: float = 0.25,
    max_risk: float = 0.02,
    min_risk: float = 0.001,
) -> float:
    """
    محاسبه اندازه موقعیت با Kelly کسری
    win_rate: نرخ برد (0..1)
    win_loss_ratio: نسبت میانگین سود به میانگین زیان
    fraction: ضریب Kelly (0.25 = ربع کلی، محافظه‌کارانه)
    max_risk: حداکثر ریسک هر معامله (مثلاً 0.02 = 2%)
    """
    if win_loss_ratio <= 0:
        return min_risk
    kelly_full = win_rate - (1 - win_rate) / win_loss_ratio
    if kelly_full <= 0:
        return 0.0
    kelly_frac = kelly_full * fraction
    return max(min_risk, min(kelly_frac, max_risk))


def confidence_based_size(
    confidence: float,
    base_size: float = 0.01,
    max_size: float = 0.02,
) -> float:
    """تنظیم اندازه بر اساس اطمینان سیگنال"""
    if confidence < 0.7:
        return 0.0
    scaled = base_size + (confidence - 0.7) / 0.3 * (max_size - base_size)
    return min(scaled, max_size)


def compute_position_value(
    capital: float,
    risk_pct: float,
    entry_price: float,
    stop_loss: float,
    max_position_pct: float = 0.25,        # حداکثر ۲۵٪ سرمایه در یک پوزیشن
    min_stop_distance_pct: float = 0.005,  # حداقل ۰.۵٪ فاصله تا SL
) -> dict:
    """
    محاسبه ارزش پوزیشن با محدودیت‌های ایمنی
    
    - max_position_pct: سقف ارزش پوزیشن نسبت به سرمایه (پیش‌فرض ۲۵٪)
    - min_stop_distance_pct: حداقل فاصله SL (برای جلوگیری از ATR کوچک)
    
    خروجی: dict با position_value, quantity, risk_amount, stop_loss, price_risk
    """
    # اعتبارسنجی
    if entry_price <= 0 or capital <= 0 or risk_pct <= 0:
        return {
            "position_value": 0.0,
            "quantity": 0.0,
            "risk_amount": 0.0,
            "stop_loss": entry_price,
            "price_risk": 0.0,
        }

    # ۱. فاصله فعلی تا stop
    price_risk = abs(entry_price - stop_loss)

    # ۲. اعمال حداقل فاصله (جلوگیری از ATR خیلی کوچک)
    min_distance = entry_price * min_stop_distance_pct
    if price_risk < min_distance:
        price_risk = min_distance
        # SL را تنظیم مجدد کن (SL به سمت پایین برای BUY و به سمت بالا برای SELL)
        if entry_price > stop_loss:
            stop_loss = entry_price - price_risk
        else:
            stop_loss = entry_price + price_risk

    # ۳. محاسبه quantity بر اساس ریسک
    risk_amount = capital * risk_pct
    quantity = risk_amount / price_risk
    position_value = quantity * entry_price

    # ۴. اعمال سقف position size (25% سرمایه)
    max_position_value = capital * max_position_pct
    if position_value > max_position_value:
        position_value = max_position_value
        quantity = position_value / entry_price

    # ۵. اعمال سقف کل سرمایه (احتیاط نهایی)
    if position_value > capital:
        position_value = capital
        quantity = capital / entry_price

    return {
        "position_value": round(position_value, 2),
        "quantity": round(quantity, 8),
        "risk_amount": round(risk_amount, 4),
        "stop_loss": round(stop_loss, 2),
        "price_risk": round(price_risk, 4),
    }