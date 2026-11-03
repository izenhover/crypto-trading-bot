# src/signal_engine/signal_generator.py
"""موتور اصلی تولید سیگنال با فیلتر روند + جهت"""
from pathlib import Path
import pandas as pd
import numpy as np
from loguru import logger

from src.signal_engine.signal_types import (
    Signal, SignalType, confidence_to_level,
)
from src.signal_engine.event_detector import (
    compute_trend,
    check_trend_filter,
    check_direction_filter,
    check_technical_confirmation,
    check_volume_confirmation,
    check_regime_confirmation,
    check_flow_confirmation,
)

PROCESSED_DIR = Path("data/processed")


class SignalGenerator:
    """تولید سیگنال با هم‌راستایی چندلایه + فیلتر روند MA200 + MA20"""

    def __init__(
        self,
        threshold_buy: float = 0.65,
        threshold_sell: float = 0.35,
        min_confirmations: int = 2,
        atr_stop_multiplier: float = 2.0,
        atr_take_profit_multiplier: float = 4.0,
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
        reasons = []
        last_row = df.iloc[-1]
        current_price = float(last_row["close"])
        atr = float(last_row.get("atr_14", current_price * 0.01))

        # ۱. Circuit Breaker
        if circuit_breaker_active:
            reasons.append("Circuit Breaker فعال — توقف کامل")
            return self._hold_signal(
                ensemble_score, regime, reasons, current_price, atr, anomaly_score
            )

        # ۲. آستانه Ensemble
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
            return self._hold_signal(
                ensemble_score, regime, reasons, current_price, atr, anomaly_score
            )

        # ۳. فیلتر روند بلندمدت MA200
        trend_info = compute_trend(df)
        trend_check = check_trend_filter(trend_info, direction)
        if not trend_check["passed"]:
            reasons.append(f"❌ فیلتر روند: {trend_check['reason']}")
            return self._hold_signal(
                ensemble_score, regime, reasons, current_price, atr, anomaly_score
            )
        reasons.append(f"✅ روند: {trend_check['reason']}")

        # ۴. فیلتر جهت MA20
        dir_check = check_direction_filter(df, direction)
        if not dir_check["passed"]:
            reasons.append(f"❌ فیلتر جهت: {dir_check['reason']}")
            return self._hold_signal(
                ensemble_score, regime, reasons, current_price, atr, anomaly_score
            )
        reasons.append(f"✅ جهت: {dir_check['reason']}")

        # ۵. تأییدهای چندلایه
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

        # ۶. تصمیم نهایی
        if confirmations < self.min_confirmations:
            reasons.append(
                f"❌ تأییدها کافی نیست ({confirmations}/{self.min_confirmations})"
            )
            return self._hold_signal(
                ensemble_score, regime, reasons, current_price, atr, anomaly_score
            )

        # ۷. ساخت سیگنال نهایی
        signal_type = SignalType.BUY if direction == "buy" else SignalType.SELL

        if direction == "buy":
            stop_loss = current_price - self.atr_stop_multiplier * atr
            take_profit = current_price + self.atr_take_profit_multiplier * atr
        else:
            stop_loss = current_price + self.atr_stop_multiplier * atr
            take_profit = current_price - self.atr_take_profit_multiplier * atr

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

    def _hold_signal(self, ensemble_score, regime, reasons,
                     current_price, atr, anomaly_score) -> Signal:
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

    def should_evaluate(self, df: pd.DataFrame, interval_minutes: int = 60) -> bool:
        from src.utils.time_utils import now_utc
        if df.empty:
            return False
        last_ts = df.index[-1]
        now = now_utc()
        elapsed = (now - last_ts).total_seconds() / 60
        return elapsed >= interval_minutes