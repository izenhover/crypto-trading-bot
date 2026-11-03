# setup_phase7_files.py
"""ساخت فایل‌های فاز ۷: مدیریت ریسک"""
from pathlib import Path

FILES = {}

# ---------- src/risk_management/position_sizing.py ----------
FILES["src/risk_management/position_sizing.py"] = '''"""محاسبه اندازه موقعیت با Kelly کسری"""
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
) -> dict:
    """
    محاسبه ارزش پوزیشن بر اساس ریسک و فاصله تا حد ضرر
    """
    risk_amount = capital * risk_pct
    price_risk = abs(entry_price - stop_loss)
    if price_risk <= 0:
        return {"position_value": 0.0, "quantity": 0.0, "risk_amount": 0.0}
    quantity = risk_amount / price_risk
    position_value = quantity * entry_price
    # محدود کردن به کل سرمایه
    if position_value > capital:
        position_value = capital
        quantity = capital / entry_price
    return {
        "position_value": round(position_value, 2),
        "quantity": round(quantity, 8),
        "risk_amount": round(risk_amount, 2),
    }
'''

# ---------- src/risk_management/stop_loss.py ----------
FILES["src/risk_management/stop_loss.py"] = '''"""محاسبه حد ضرر و حد سود مبتنی بر ATR"""
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
'''

# ---------- src/risk_management/portfolio.py ----------
FILES["src/risk_management/portfolio.py"] = '''"""مدیریت پوزیشن‌ها و سرمایه"""
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional, List
from loguru import logger


@dataclass
class Position:
    id: int
    symbol: str
    direction: str          # BUY / SELL
    entry_price: float
    quantity: float
    position_value: float
    stop_loss: float
    take_profit: float
    entry_time: str
    confidence: float
    regime: str
    status: str = "OPEN"    # OPEN / CLOSED
    exit_price: Optional[float] = None
    exit_time: Optional[str] = None
    pnl_pct: Optional[float] = None
    pnl_amount: Optional[float] = None

    def to_dict(self):
        return asdict(self)


class Portfolio:
    """مدیریت پوزیشن‌های باز"""

    def __init__(self, initial_capital: float = 10000.0, max_open: int = 3):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.max_open = max_open
        self.positions: List[Position] = []
        self.closed_positions: List[Position] = []
        self._next_id = 1

    @property
    def open_positions(self) -> List[Position]:
        return [p for p in self.positions if p.status == "OPEN"]

    @property
    def open_value(self) -> float:
        return sum(p.position_value for p in self.open_positions)

    @property
    def available_capital(self) -> float:
        return self.capital - self.open_value

    def can_open(self, direction: str = None) -> tuple:
        """آیا می‌توان پوزیشن جدید باز کرد؟"""
        if len(self.open_positions) >= self.max_open:
            return False, f"حداکثر پوزیشن باز ({self.max_open})"

        # چک: نباید در همان جهت پوزیشن باز داشته باشیم (جلوگیری از over-trading)
        if direction:
            same_dir = [p for p in self.open_positions if p.direction == direction.upper()]
            if same_dir:
                return False, f"پوزیشن {direction} از قبل باز است"

        return True, "OK"

    def open_position(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        quantity: float,
        position_value: float,
        stop_loss: float,
        take_profit: float,
        confidence: float,
        regime: str,
    ) -> Position:
        pos = Position(
            id=self._next_id,
            symbol=symbol,
            direction=direction.upper(),
            entry_price=entry_price,
            quantity=quantity,
            position_value=position_value,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_time=datetime.now(timezone.utc).isoformat(),
            confidence=confidence,
            regime=regime,
        )
        self._next_id += 1
        self.positions.append(pos)
        logger.info(
            f"OPEN #{pos.id} {direction} {symbol} @ {entry_price:.2f} "
            f"qty={quantity:.6f} SL={stop_loss:.2f} TP={take_profit:.2f}"
        )
        return pos

    def check_exits(self, current_high: float, current_low: float, current_close: float):
        """بررسی بسته شدن پوزیشن‌های باز بر اساس SL/TP"""
        for pos in self.open_positions:
            exit_price = None
            if pos.direction == "BUY":
                if current_low <= pos.stop_loss:
                    exit_price = pos.stop_loss
                elif current_high >= pos.take_profit:
                    exit_price = pos.take_profit
            else:  # SELL
                if current_high >= pos.stop_loss:
                    exit_price = pos.stop_loss
                elif current_low <= pos.take_profit:
                    exit_price = pos.take_profit

            if exit_price is not None:
                self.close_position(pos, exit_price)

    def close_position(self, pos: Position, exit_price: float):
        if pos.direction == "BUY":
            pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
        else:
            pnl_pct = (pos.entry_price - exit_price) / pos.entry_price

        pnl_amount = pnl_pct * pos.position_value
        pos.status = "CLOSED"
        pos.exit_price = exit_price
        pos.exit_time = datetime.now(timezone.utc).isoformat()
        pos.pnl_pct = round(pnl_pct, 6)
        pos.pnl_amount = round(pnl_amount, 4)
        self.capital += pnl_amount
        self.closed_positions.append(pos)
        self.positions.remove(pos)

        logger.info(
            f"CLOSE #{pos.id} {pos.direction} @ {exit_price:.2f} "
            f"PnL={pnl_pct*100:+.2f}% ({pnl_amount:+.2f} USDT)"
        )

    def stats(self) -> dict:
        """آمار کلی"""
        closed = self.closed_positions
        if not closed:
            return {
                "capital": round(self.capital, 2),
                "total_trades": 0,
                "open": len(self.open_positions),
                "return_pct": round((self.capital / self.initial_capital - 1) * 100, 2),
            }

        wins = [p for p in closed if p.pnl_pct > 0]
        losses = [p for p in closed if p.pnl_pct <= 0]
        total_pnl = sum(p.pnl_amount for p in closed)
        avg_win = sum(p.pnl_pct for p in wins) / len(wins) if wins else 0
        avg_loss = sum(p.pnl_pct for p in losses) / len(losses) if losses else 0

        win_rate = len(wins) / len(closed)
        wl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

        return {
            "capital": round(self.capital, 2),
            "return_pct": round((self.capital / self.initial_capital - 1) * 100, 2),
            "total_trades": len(closed),
            "open": len(self.open_positions),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(win_rate, 4),
            "avg_win_pct": round(avg_win * 100, 4),
            "avg_loss_pct": round(avg_loss * 100, 4),
            "win_loss_ratio": round(wl_ratio, 2),
            "total_pnl": round(total_pnl, 2),
        }
'''

# ---------- src/risk_management/__init__.py ----------
FILES["src/risk_management/__init__.py"] = '''from src.risk_management.position_sizing import (
    fractional_kelly,
    confidence_based_size,
    compute_position_value,
)
from src.risk_management.stop_loss import (
    atr_stop_loss,
    atr_take_profit,
    trailing_stop,
    risk_reward_ratio,
)
from src.risk_management.portfolio import Portfolio, Position

__all__ = [
    "fractional_kelly",
    "confidence_based_size",
    "compute_position_value",
    "atr_stop_loss",
    "atr_take_profit",
    "trailing_stop",
    "risk_reward_ratio",
    "Portfolio",
    "Position",
]
'''

# ---------- scripts/simulate_trading.py ----------
FILES["scripts/simulate_trading.py"] = '''"""شبیه‌سازی معامله با مدیریت ریسک"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger

from src.feature_engineering.pipeline import get_processed_path
from src.models import Ensemble
from src.regime_detection import RegimeDetector
from src.money_flow import FlowAnomalyDetector, CircuitBreaker
from src.signal_engine import SignalGenerator, SignalType
from src.risk_management import (
    fractional_kelly, confidence_based_size,
    compute_position_value, atr_stop_loss, atr_take_profit,
    Portfolio,
)
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger


def main():
    setup_logger("sim")
    config = load_config()

    symbol, interval = "BTCUSDT", "1h"
    df = pd.read_parquet(get_processed_path(symbol, interval))
    logger.info(f"Loaded {len(df)} rows")

    # بارگذاری ماژول‌ها
    ensemble = Ensemble.load_all(include_lstm=False)
    regime_detector = RegimeDetector.load()
    anomaly_detector = FlowAnomalyDetector.load()

    # تنظیمات
    sig_cfg = config["signal"]
    risk_cfg = config["risk"]
    bt_cfg = config["backtest"]

    generator = SignalGenerator(
        threshold_buy=sig_cfg["threshold_buy"],
        threshold_sell=sig_cfg["threshold_sell"],
        min_confirmations=sig_cfg["min_confirmations"],
        atr_stop_multiplier=risk_cfg["atr_stop_multiplier"],
        atr_take_profit_multiplier=risk_cfg["atr_take_profit_multiplier"],
    )
    portfolio = Portfolio(
        initial_capital=bt_cfg["initial_capital"],
        max_open=risk_cfg["max_open_positions"],
    )
    cb_cfg = risk_cfg["circuit_breaker"]
    cb = CircuitBreaker(
        max_consecutive_losses=cb_cfg["max_consecutive_losses"],
        max_daily_loss_pct=cb_cfg["max_daily_loss_pct"],
        halt_duration_hours=cb_cfg["halt_duration_hours"],
    )

    commission = bt_cfg["commission_pct"]
    slippage = bt_cfg["slippage_pct"]

    # شبیه‌سازی روی کل داده
    from src.models.dataset import get_feature_columns
    feature_cols = get_feature_columns(df)
    start_idx = 500  # warm-up

    logger.info(f"Simulating from candle {start_idx} to {len(df)}")

    for i in range(start_idx, len(df)):
        current_df = df.iloc[:i+1]
        current_candle = df.iloc[i]

        # ابتدا چک بسته شدن پوزیشن‌های باز
        portfolio.check_exits(
            current_high=current_candle["high"],
            current_low=current_candle["low"],
            current_close=current_candle["close"],
        )

        if not cb.can_trade():
            continue

        # پیش‌بینی
        X_current = current_df[feature_cols].iloc[-1:]
        ensemble_score = float(ensemble.predict_proba(X_current)[0])

        try:
            regime = regime_detector.predict_current(current_df, window=150)
        except Exception:
            regime = "UNKNOWN"

        try:
            anomaly = anomaly_detector.current_score(current_df)
        except Exception:
            anomaly = {"anomaly_score": 0.0, "is_anomaly": False}

        cb.record_anomaly(anomaly["anomaly_score"], anomaly["is_anomaly"])

        # سیگنال
        signal = generator.generate(
            df=current_df,
            ensemble_score=ensemble_score,
            regime=regime,
            anomaly_score=anomaly["anomaly_score"],
            is_anomaly=anomaly["is_anomaly"],
            circuit_breaker_active=not cb.can_trade(),
        )

        if signal.signal == SignalType.HOLD:
            continue

        # چک امکان باز کردن
        can, reason = portfolio.can_open(signal.signal.value)
        if not can:
            continue

        # اندازه‌گذاری
        stats = portfolio.stats()
        if stats["total_trades"] >= 10:
            win_rate = stats["win_rate"]
            wl = stats["win_loss_ratio"] if stats["win_loss_ratio"] > 0 else 1.0
            risk_pct = fractional_kelly(win_rate, wl, fraction=risk_cfg["kelly_fraction"],
                                        max_risk=risk_cfg["max_risk_per_trade"])
        else:
            risk_pct = confidence_based_size(
                signal.confidence,
                base_size=0.005,
                max_size=risk_cfg["max_risk_per_trade"],
            )

        if risk_pct <= 0:
            continue

        entry_price = current_candle["close"] * (1 + slippage if signal.signal == SignalType.BUY else 1 - slippage)
        entry_price *= (1 + commission) if signal.signal == SignalType.BUY else (1 - commission)

        sizing = compute_position_value(
            capital=portfolio.available_capital,
            risk_pct=risk_pct,
            entry_price=entry_price,
            stop_loss=signal.stop_loss,
        )

        if sizing["quantity"] <= 0:
            continue

        portfolio.open_position(
            symbol=symbol,
            direction=signal.signal.value,
            entry_price=entry_price,
            quantity=sizing["quantity"],
            position_value=sizing["position_value"],
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            confidence=signal.confidence,
            regime=regime,
        )

    # بستن پوزیشن‌های باز در پایان
    last_close = df.iloc[-1]["close"]
    for pos in list(portfolio.open_positions):
        portfolio.close_position(pos, last_close)

    # آمار نهایی
    final = portfolio.stats()
    logger.info("")
    logger.info("=" * 60)
    logger.info("=== TRADING SIMULATION RESULTS ===")
    logger.info("=" * 60)
    for k, v in final.items():
        logger.info(f"  {k:<20}: {v}")
    logger.info("=" * 60)

    # ذخیره معاملات
    if portfolio.closed_positions:
        trades_df = pd.DataFrame([p.to_dict() for p in portfolio.closed_positions])
        out = Path("data/processed") / f"{symbol}_{interval}_trades.parquet"
        trades_df.to_parquet(out)
        logger.info(f"Saved {len(trades_df)} trades -> {out}")


if __name__ == "__main__":
    main()
'''

# ---------- tests/test_risk.py ----------
FILES["tests/test_risk.py"] = '''"""تست ماژول‌های مدیریت ریسک"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.risk_management import (
    fractional_kelly, confidence_based_size, compute_position_value,
    atr_stop_loss, atr_take_profit, trailing_stop, risk_reward_ratio,
    Portfolio,
)
from src.utils.logger import setup_logger

setup_logger("test")


def main():
    print()
    print("=" * 60)
    print("Testing Risk Management")
    print("=" * 60)

    # ۱. Kelly
    k1 = fractional_kelly(win_rate=0.55, win_loss_ratio=1.5, fraction=0.25)
    print(f"[OK] Kelly (WR=55%, W/L=1.5): {k1*100:.2f}% per trade")

    k2 = fractional_kelly(win_rate=0.45, win_loss_ratio=1.0, fraction=0.25)
    print(f"[OK] Kelly (WR=45%, W/L=1.0): {k2*100:.2f}% (should be 0)")

    # ۲. Confidence-based
    s1 = confidence_based_size(0.90)
    s2 = confidence_based_size(0.65)
    print(f"[OK] Confidence size (90%): {s1*100:.2f}%")
    print(f"[OK] Confidence size (65%): {s2*100:.2f}% (below threshold)")

    # ۳. Position value
    pv = compute_position_value(
        capital=10000, risk_pct=0.02,
        entry_price=60000, stop_loss=59000,
    )
    print(f"[OK] Position value: {pv}")

    # ۴. ATR Stop
    sl = atr_stop_loss(60000, "BUY", atr=500, multiplier=1.5)
    tp = atr_take_profit(60000, "BUY", atr=500, multiplier=3.0)
    print(f"[OK] BUY: entry=60000 SL={sl:.2f} TP={tp:.2f} RR={risk_reward_ratio(60000, sl, tp)}")

    # ۵. Trailing stop
    ts = trailing_stop(current_price=61000, highest_price=62000, lowest_price=59000,
                       direction="BUY", atr=500)
    print(f"[OK] Trailing stop: {ts:.2f}")

    # ۶. Portfolio
    pf = Portfolio(initial_capital=10000, max_open=3)
    print(f"[OK] Portfolio initial capital: {pf.capital}")

    pf.open_position("BTCUSDT", "BUY", entry_price=60000,
                     quantity=0.1, position_value=6000,
                     stop_loss=59000, take_profit=63000,
                     confidence=0.85, regime="BULL")
    print(f"[OK] After opening: open_positions={len(pf.open_positions)}, "
          f"available={pf.available_capital:.2f}")

    # شبیه‌سازی خروج
    pf.check_exits(current_high=63500, current_low=59500, current_close=63400)
    print(f"[OK] After TP hit: closed={len(pf.closed_positions)}, "
          f"capital={pf.capital:.2f}")

    stats = pf.stats()
    print(f"[OK] Stats: return={stats['return_pct']}%, win_rate={stats['win_rate']}")

    print()
    print("=" * 60)
    print("All risk management tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
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
    print("  1. python tests/test_risk.py")
    print("  2. python scripts/simulate_trading.py")
    print()


if __name__ == "__main__":
    main()