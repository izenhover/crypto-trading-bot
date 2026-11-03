"""انواع داده‌ای سیگنال"""
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
import json


class SignalType(Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


class ConfidenceLevel(Enum):
    VERY_LOW = "VERY_LOW"      # 0.50 - 0.60
    LOW = "LOW"                # 0.60 - 0.70
    MEDIUM = "MEDIUM"          # 0.70 - 0.80
    HIGH = "HIGH"              # 0.80 - 0.90
    VERY_HIGH = "VERY_HIGH"    # 0.90 - 1.00


@dataclass
class Signal:
    """سیگنال نهایی"""
    signal: SignalType
    confidence: float
    confidence_level: ConfidenceLevel
    ensemble_score: float
    regime: str
    confirmations: int
    reasons: list
    current_price: float
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    atr: Optional[float] = None
    anomaly_score: float = 0.0
    volume_ratio: float = 1.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        d = asdict(self)
        d["signal"] = self.signal.value
        d["confidence_level"] = self.confidence_level.value
        return d

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)

    def __str__(self):
        return (
            f"[{self.signal.value}] conf={self.confidence:.2f} "
            f"({self.confidence_level.value}) regime={self.regime} "
            f"price={self.current_price:.2f} "
            f"SL={self.stop_loss:.2f} TP={self.take_profit:.2f}"
            if self.stop_loss and self.take_profit
            else f"[{self.signal.value}] conf={self.confidence:.2f} "
                 f"({self.confidence_level.value}) regime={self.regime}"
        )


def confidence_to_level(conf: float) -> ConfidenceLevel:
    if conf >= 0.90:
        return ConfidenceLevel.VERY_HIGH
    if conf >= 0.80:
        return ConfidenceLevel.HIGH
    if conf >= 0.70:
        return ConfidenceLevel.MEDIUM
    if conf >= 0.60:
        return ConfidenceLevel.LOW
    return ConfidenceLevel.VERY_LOW
