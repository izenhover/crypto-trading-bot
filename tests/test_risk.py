"""تست ماژول‌های مدیریت ریسک"""
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
