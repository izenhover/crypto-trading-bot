"""مدیریت پوزیشن‌ها و سرمایه"""
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
