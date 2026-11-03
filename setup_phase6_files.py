# setup_phase6_files.py
"""ساخت فایل‌های فاز ۶: موتور سیگنال رویدادمحور"""
from pathlib import Path

FILES = {}

# ---------- src/signal_engine/signal_types.py ----------
FILES["src/signal_engine/signal_types.py"] = '''"""انواع داده‌ای سیگنال"""
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
'''

# ---------- src/signal_engine/event_detector.py ----------
FILES["src/signal_engine/event_detector.py"] = '''"""تشخیص رویدادهای هم‌راستایی"""
import pandas as pd
import numpy as np


def check_technical_confirmation(row: pd.Series, direction: str = "buy") -> dict:
    """
    بررسی تأیید تکنیکال روی آخرین کندل
    direction: 'buy' یا 'sell'
    """
    checks = {}

    if direction == "buy":
        checks["rsi_ok"] = row.get("rsi_14", 50) < 70
        checks["rsi_not_oversold"] = row.get("rsi_14", 50) > 25
        checks["macd_bullish"] = row.get("macd_hist", 0) > -0.001
        checks["bb_not_overbought"] = row.get("bb_position", 0.5) < 0.95
        checks["above_bb_mid"] = row.get("close", 0) > row.get("bb_mid", 0)
    else:  # sell
        checks["rsi_ok"] = row.get("rsi_14", 50) > 30
        checks["rsi_not_overbought"] = row.get("rsi_14", 50) < 75
        checks["macd_bearish"] = row.get("macd_hist", 0) < 0.001
        checks["bb_not_oversold"] = row.get("bb_position", 0.5) > 0.05
        checks["below_bb_mid"] = row.get("close", 0) < row.get("bb_mid", 0)

    score = sum(checks.values()) / len(checks)
    return {"checks": checks, "score": score, "passed": score >= 0.6}


def check_volume_confirmation(row: pd.Series, min_ratio: float = 1.2) -> dict:
    """بررسی تأیید حجم"""
    ratio = row.get("volume_ratio_20", 1.0)
    return {
        "volume_ratio": float(ratio),
        "passed": ratio >= min_ratio,
    }


def check_regime_confirmation(regime: str, direction: str) -> dict:
    """بررسی تأیید رژیم"""
    if direction == "buy":
        allowed = regime in ("BULL", "SIDEWAYS_LOW", "SIDEWAYS_HIGH")
        preferred = regime in ("BULL", "SIDEWAYS_HIGH")
    else:  # sell
        allowed = regime in ("BEAR", "SIDEWAYS_HIGH", "SIDEWAYS_LOW")
        preferred = regime in ("BEAR", "SIDEWAYS_HIGH")

    return {
        "regime": regime,
        "allowed": allowed,
        "preferred": preferred,
        "passed": allowed,
    }


def check_flow_confirmation(anomaly_score: float, is_anomaly: bool,
                             volume_ratio: float, direction: str = "buy") -> dict:
    """
    بررسی تأیید جریان پول
    برای خرید: ناهنجاری شدید = بد؛ حجم بالا = خوب
    برای فروش: ناهنجاری شدید = هشدار
    """
    if is_anomaly and anomaly_score > 0.7:
        return {
            "flow_ok": False,
            "reason": f"ناهنجاری شدید جریان (score={anomaly_score:.2f})",
            "passed": False,
        }

    if direction == "buy":
        volume_ok = volume_ratio >= 0.9
        return {
            "flow_ok": volume_ok,
            "reason": f"حجم عادی (ratio={volume_ratio:.2f})",
            "passed": volume_ok,
        }
    else:
        return {
            "flow_ok": True,
            "reason": f"بدون هشدار جریان",
            "passed": True,
        }
'''

# ---------- src/signal_engine/signal_generator.py ----------
FILES["src/signal_engine/signal_generator.py"] = '''"""موتور اصلی تولید سیگنال رویدادمحور"""
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger

from src.signal_engine.signal_types import (
    Signal, SignalType, confidence_to_level,
)
from src.signal_engine.event_detector import (
    check_technical_confirmation,
    check_volume_confirmation,
    check_regime_confirmation,
    check_flow_confirmation,
)

PROCESSED_DIR = Path("data/processed")


class SignalGenerator:
    """تولید سیگنال با هم‌راستایی چندلایه"""

    def __init__(
        self,
        threshold_buy: float = 0.80,
        threshold_sell: float = 0.20,
        min_confirmations: int = 2,
        atr_stop_multiplier: float = 1.5,
        atr_take_profit_multiplier: float = 3.0,
    ):
        self.threshold_buy = threshold_buy
        self.threshold_sell = threshold_sell
        self.min_confirmations = min_confirmations
        self.atr_stop_multiplier = atr_stop_multiplier
        self.atr_take_profit_multiplier = atr_take_profit_multiplier

    def generate(
        self,
        df: pd.DataFrame,
        ensemble_score: float,
        regime: str,
        anomaly_score: float,
        is_anomaly: bool,
        circuit_breaker_active: bool = False,
    ) -> Signal:
        """
        تولید سیگنال بر اساس هم‌راستایی همه لایه‌ها

        df: دیتافریم ویژگی‌ها (آخرین ردیف، کندل بسته‌شده)
        ensemble_score: احتمال صعود از Ensemble (0..1)
        regime: رژیم فعلی
        anomaly_score: امتیاز ناهنجاری جریان پول
        is_anomaly: آیا ناهنجاری تشخیص داده شده
        circuit_breaker_active: آیا Circuit Breaker فعال است
        """
        reasons = []
        last_row = df.iloc[-1]
        current_price = float(last_row["close"])
        atr = float(last_row.get("atr_14", current_price * 0.01))

        # 1️⃣ چک Circuit Breaker
        if circuit_breaker_active:
            reasons.append("Circuit Breaker فعال — توقف کامل")
            return Signal(
                signal=SignalType.HOLD,
                confidence=0.0,
                confidence_level=confidence_to_level(0.0),
                ensemble_score=ensemble_score,
                regime=regime,
                confirmations=0,
                reasons=reasons,
                current_price=current_price,
                atr=atr,
                anomaly_score=anomaly_score,
            )

        # 2️⃣ بررسی آستانه Ensemble
        direction = None
        if ensemble_score >= self.threshold_buy:
            direction = "buy"
            confidence = ensemble_score
        elif ensemble_score <= self.threshold_sell:
            direction = "sell"
            confidence = 1.0 - ensemble_score
        else:
            reasons.append(
                f"امتیاز Ensemble ({ensemble_score:.2f}) در محدوده بی‌طرف "
                f"[{self.threshold_sell:.2f}, {self.threshold_buy:.2f}]"
            )
            return Signal(
                signal=SignalType.HOLD,
                confidence=0.0,
                confidence_level=confidence_to_level(0.0),
                ensemble_score=ensemble_score,
                regime=regime,
                confirmations=0,
                reasons=reasons,
                current_price=current_price,
                atr=atr,
                anomaly_score=anomaly_score,
            )

        # 3️⃣ تأییدهای چندلایه
        tech = check_technical_confirmation(last_row, direction=direction)
        vol = check_volume_confirmation(last_row)
        reg = check_regime_confirmation(regime, direction=direction)
        flow = check_flow_confirmation(
            anomaly_score, is_anomaly,
            volume_ratio=float(last_row.get("volume_ratio_20", 1.0)),
            direction=direction,
        )

        confirmations = sum([
            tech["passed"],
            vol["passed"],
            reg["passed"],
            flow["passed"],
        ])

        reasons.append(f"امتیاز Ensemble: {ensemble_score:.3f}")
        reasons.append(f"رژیم: {regime} ({'✔' if reg['passed'] else '✘'})")
        reasons.append(f"تکنیکال: {tech['score']:.2f} ({'✔' if tech['passed'] else '✘'})")
        reasons.append(f"حجم: {vol['volume_ratio']:.2f} ({'✔' if vol['passed'] else '✘'})")
        reasons.append(f"جریان: {flow['reason']} ({'✔' if flow['passed'] else '✘'})")

        # 4️⃣ تصمیم نهایی
        if confirmations < self.min_confirmations:
            reasons.append(
                f"❌ تأییدها کافی نیست ({confirmations}/{self.min_confirmations})"
            )
            return Signal(
                signal=SignalType.HOLD,
                confidence=0.0,
                confidence_level=confidence_to_level(0.0),
                ensemble_score=ensemble_score,
                regime=regime,
                confirmations=confirmations,
                reasons=reasons,
                current_price=current_price,
                atr=atr,
                anomaly_score=anomaly_score,
            )

        # 5️⃣ ساخت سیگنال نهایی
        signal_type = SignalType.BUY if direction == "buy" else SignalType.SELL

        if direction == "buy":
            stop_loss = current_price - self.atr_stop_multiplier * atr
            take_profit = current_price + self.atr_take_profit_multiplier * atr
        else:
            stop_loss = current_price + self.atr_stop_multiplier * atr
            take_profit = current_price - self.atr_take_profit_multiplier * atr

        # تنظیم اطمینان بر اساس تعداد تأییدها
        adjusted_confidence = confidence * (0.7 + 0.1 * confirmations)
        adjusted_confidence = min(1.0, adjusted_confidence)

        reasons.append(f"✅ سیگنال {signal_type.value} صادر شد")

        return Signal(
            signal=signal_type,
            confidence=adjusted_confidence,
            confidence_level=confidence_to_level(adjusted_confidence),
            ensemble_score=ensemble_score,
            regime=regime,
            confirmations=confirmations,
            reasons=reasons,
            current_price=current_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            atr=atr,
            anomaly_score=anomaly_score,
            volume_ratio=float(last_row.get("volume_ratio_20", 1.0)),
        )

    def should_evaluate(self, df: pd.DataFrame, interval_minutes: int = 60) -> bool:
        """آیا الان زمان ارزیابی است؟ (فقط روی کندل بسته‌شده)"""
        from src.utils.time_utils import now_utc
        if df.empty:
            return False
        last_ts = df.index[-1]
        now = now_utc()
        elapsed = (now - last_ts).total_seconds() / 60
        return elapsed >= interval_minutes
'''

# ---------- src/signal_engine/__init__.py ----------
FILES["src/signal_engine/__init__.py"] = '''from src.signal_engine.signal_types import (
    Signal,
    SignalType,
    ConfidenceLevel,
    confidence_to_level,
)
from src.signal_engine.event_detector import (
    check_technical_confirmation,
    check_volume_confirmation,
    check_regime_confirmation,
    check_flow_confirmation,
)
from src.signal_engine.signal_generator import SignalGenerator

__all__ = [
    "Signal",
    "SignalType",
    "ConfidenceLevel",
    "confidence_to_level",
    "check_technical_confirmation",
    "check_volume_confirmation",
    "check_regime_confirmation",
    "check_flow_confirmation",
    "SignalGenerator",
]
'''

# ---------- scripts/generate_signal.py ----------
FILES["scripts/generate_signal.py"] = '''"""اسکریپت تست تولید سیگنال روی داده تاریخی"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger

from src.feature_engineering.pipeline import get_processed_path
from src.models import Ensemble, LSTMModel
from src.regime_detection import RegimeDetector
from src.money_flow import FlowAnomalyDetector, CircuitBreaker, volume_proxy_flow
from src.signal_engine import SignalGenerator, SignalType
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger


def main():
    setup_logger("signal")
    config = load_config()

    symbol, interval = "BTCUSDT", "1h"
    logger.info(f"=== Signal Generation Test for {symbol} {interval} ===")

    # ۱. بارگذاری داده
    df = pd.read_parquet(get_processed_path(symbol, interval))
    logger.info(f"Loaded {len(df)} rows")

    # ۲. بارگذاری مدل‌ها
    ensemble = Ensemble.load_all(include_lstm=False)  # LSTM فعلاً ضعیف است، کنار می‌گذاریم
    regime_detector = RegimeDetector.load()
    anomaly_detector = FlowAnomalyDetector.load()

    # ۳. بارگذاری تنظیمات سیگنال
    sig_cfg = config["signal"]
    risk_cfg = config["risk"]
    generator = SignalGenerator(
        threshold_buy=sig_cfg["threshold_buy"],
        threshold_sell=sig_cfg["threshold_sell"],
        min_confirmations=sig_cfg["min_confirmations"],
        atr_stop_multiplier=risk_cfg["atr_stop_multiplier"],
        atr_take_profit_multiplier=risk_cfg["atr_take_profit_multiplier"],
    )

    # ۴. Circuit Breaker
    cb_cfg = risk_cfg["circuit_breaker"]
    cb = CircuitBreaker(
        max_consecutive_losses=cb_cfg["max_consecutive_losses"],
        max_daily_loss_pct=cb_cfg["max_daily_loss_pct"],
        halt_duration_hours=cb_cfg["halt_duration_hours"],
    )

    # ۵. محاسبه روی ۲۰۰ کندل آخر (شبیه‌سازی حالت زنده)
    window = 200
    logger.info(f"Evaluating last {window} candles (one by one)")

    # برای Ensemble نیاز به ویژگی‌ها داریم
    from src.models.dataset import get_feature_columns
    feature_cols = get_feature_columns(df)

    signals = []
    for i in range(len(df) - window, len(df)):
        current_df = df.iloc[:i+1]
        if len(current_df) < 60:
            continue

        X_current = current_df[feature_cols].iloc[-1:]
        ensemble_score = float(ensemble.predict_proba(X_current)[0])

        # رژیم
        try:
            regime = regime_detector.predict_current(current_df, window=150)
        except Exception:
            regime = "UNKNOWN"

        # ناهنجاری
        try:
            anomaly = anomaly_detector.current_score(current_df)
        except Exception:
            anomaly = {"anomaly_score": 0.0, "is_anomaly": False}

        # سیگنال
        signal = generator.generate(
            df=current_df,
            ensemble_score=ensemble_score,
            regime=regime,
            anomaly_score=anomaly["anomaly_score"],
            is_anomaly=anomaly["is_anomaly"],
            circuit_breaker_active=not cb.can_trade(),
        )

        signals.append({
            "timestamp": current_df.index[-1],
            "close": float(current_df["close"].iloc[-1]),
            "signal": signal.signal.value,
            "confidence": signal.confidence,
            "regime": regime,
            "ensemble_score": ensemble_score,
            "confirmations": signal.confirmations,
        })

    result_df = pd.DataFrame(signals).set_index("timestamp")
    out = Path("data/processed") / f"{symbol}_{interval}_signals.parquet"
    result_df.to_parquet(out)

    # ۶. آمار
    logger.info("")
    logger.info("=" * 60)
    logger.info("Signal Distribution (last 200 candles)")
    logger.info("=" * 60)
    dist = result_df["signal"].value_counts()
    for s, c in dist.items():
        logger.info(f"  {s:<6}: {c:>4}  ({c/len(result_df)*100:.1f}%)")

    # سیگنال‌های قوی
    strong = result_df[result_df["signal"] != "HOLD"]
    logger.info("")
    logger.info(f"Non-HOLD signals: {len(strong)}")
    if not strong.empty:
        logger.info("")
        logger.info("Top signals:")
        for ts, row in strong.iterrows():
            logger.info(
                f"  {ts} | {row['signal']} | conf={row['confidence']:.3f} | "
                f"regime={row['regime']} | confirm={row['confirmations']}"
            )

    logger.info("")
    logger.info(f"Saved signals -> {out}")


if __name__ == "__main__":
    main()
'''

# ---------- tests/test_signal.py ----------
FILES["tests/test_signal.py"] = '''"""تست موتور سیگنال"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.feature_engineering.pipeline import get_processed_path
from src.signal_engine import SignalGenerator, SignalType
from src.utils.logger import setup_logger

setup_logger("test")


def main():
    print()
    print("=" * 60)
    print("Testing Signal Generator")
    print("=" * 60)

    df = pd.read_parquet(get_processed_path("BTCUSDT", "1h"))
    generator = SignalGenerator(
        threshold_buy=0.80,
        threshold_sell=0.20,
        min_confirmations=2,
    )

    # تست سناریوهای مختلف
    scenarios = [
        {
            "name": "STRONG BUY",
            "ensemble_score": 0.90,
            "regime": "BULL",
            "anomaly_score": 0.3,
            "is_anomaly": False,
        },
        {
            "name": "MARGINAL (in neutral zone)",
            "ensemble_score": 0.55,
            "regime": "SIDEWAYS_LOW",
            "anomaly_score": 0.3,
            "is_anomaly": False,
        },
        {
            "name": "STRONG SELL",
            "ensemble_score": 0.10,
            "regime": "BEAR",
            "anomaly_score": 0.3,
            "is_anomaly": False,
        },
        {
            "name": "CIRCUIT BREAKER ACTIVE",
            "ensemble_score": 0.95,
            "regime": "BULL",
            "anomaly_score": 0.3,
            "is_anomaly": False,
            "circuit_breaker": True,
        },
        {
            "name": "SEVERE ANOMALY",
            "ensemble_score": 0.90,
            "regime": "BULL",
            "anomaly_score": 0.95,
            "is_anomaly": True,
        },
    ]

    for scenario in scenarios:
        print()
        print("-" * 60)
        print(f"Scenario: {scenario['name']}")
        print("-" * 60)

        signal = generator.generate(
            df=df,
            ensemble_score=scenario["ensemble_score"],
            regime=scenario["regime"],
            anomaly_score=scenario["anomaly_score"],
            is_anomaly=scenario["is_anomaly"],
            circuit_breaker_active=scenario.get("circuit_breaker", False),
        )

        print(f"  Signal: {signal.signal.value}")
        print(f"  Confidence: {signal.confidence:.3f} ({signal.confidence_level.value})")
        print(f"  Confirmations: {signal.confirmations}")
        if signal.stop_loss:
            print(f"  Stop Loss: {signal.stop_loss:.2f}")
            print(f"  Take Profit: {signal.take_profit:.2f}")
        print("  Reasons:")
        for r in signal.reasons:
            print(f"    - {r}")

    print()
    print("=" * 60)
    print("All signal tests passed!")
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
    print("  1. python tests/test_signal.py")
    print("  2. python scripts/generate_signal.py")
    print()


if __name__ == "__main__":
    main()