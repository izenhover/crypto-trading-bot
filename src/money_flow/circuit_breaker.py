"""قطع‌کننده مدار برای توقف معاملات در شرایط بحرانی"""
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional
from loguru import logger


@dataclass
class CircuitBreakerState:
    halted: bool = False
    halt_until: Optional[datetime] = None
    reason: str = ""
    consecutive_losses: int = 0
    daily_pnl: float = 0.0
    last_reset: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class CircuitBreaker:
    """مدیریت توقف اضطراری معاملات"""

    def __init__(
        self,
        max_consecutive_losses: int = 3,
        max_daily_loss_pct: float = 0.05,
        halt_duration_hours: int = 24,
        anomaly_threshold: float = 0.95,
    ):
        self.max_consecutive_losses = max_consecutive_losses
        self.max_daily_loss_pct = max_daily_loss_pct
        self.halt_duration_hours = halt_duration_hours
        self.anomaly_threshold = anomaly_threshold
        self.state = CircuitBreakerState()

    def _reset_if_new_day(self):
        now = datetime.now(timezone.utc)
        if now.date() != self.state.last_reset.date():
            logger.info("New day: resetting circuit breaker daily stats")
            self.state.daily_pnl = 0.0
            self.state.consecutive_losses = 0
            self.state.last_reset = now

    def record_trade(self, pnl_pct: float):
        """ثبت نتیجه معامله (pnl_pct: بازده به‌صورت اعشار، مثلاً 0.02 = +2%)"""
        self._reset_if_new_day()
        self.state.daily_pnl += pnl_pct

        if pnl_pct < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0

        # چک شرایط توقف
        if self.state.consecutive_losses >= self.max_consecutive_losses:
            self.halt(
                f"{self.state.consecutive_losses} consecutive losses",
                hours=self.halt_duration_hours,
            )
        elif self.state.daily_pnl <= -self.max_daily_loss_pct:
            self.halt(
                f"daily loss limit hit: {self.state.daily_pnl*100:.2f}%",
                hours=self.halt_duration_hours * 2,
            )

    def record_anomaly(self, anomaly_score: float, is_anomaly: bool):
        """ثبت ناهنجاری جریان پول"""
        if is_anomaly and anomaly_score > self.anomaly_threshold:
            self.halt(
                f"severe flow anomaly (score={anomaly_score:.2f})",
                hours=2,
            )

    def halt(self, reason: str, hours: int = 24):
        self.state.halted = True
        self.state.halt_until = datetime.now(timezone.utc) + timedelta(hours=hours)
        self.state.reason = reason
        logger.warning(f"CIRCUIT BREAKER: {reason} — halted for {hours}h")

    def can_trade(self) -> bool:
        self._reset_if_new_day()
        if not self.state.halted:
            return True
        if self.state.halt_until and datetime.now(timezone.utc) >= self.state.halt_until:
            logger.info("Circuit breaker released")
            self.state.halted = False
            self.state.halt_until = None
            self.state.reason = ""
            return True
        return False

    def status(self) -> dict:
        return {
            "halted": self.state.halted,
            "halt_until": self.state.halt_until.isoformat() if self.state.halt_until else None,
            "reason": self.state.reason,
            "consecutive_losses": self.state.consecutive_losses,
            "daily_pnl": self.state.daily_pnl,
        }
