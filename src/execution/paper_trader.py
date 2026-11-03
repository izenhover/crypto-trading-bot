# src/execution/paper_trader.py
"""Paper Trader: معامله شبیه‌سازی‌شده با محدودیت‌های ایمنی"""
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from loguru import logger

from src.signal_engine import Signal, SignalType
from src.risk_management import (
    Portfolio, Position,
    fractional_kelly, confidence_based_size,
    compute_position_value,
)
from src.money_flow import CircuitBreaker


STATE_DIR = Path("data/state")
STATE_DIR.mkdir(parents=True, exist_ok=True)

PORTFOLIO_FILE = STATE_DIR / "paper_portfolio.json"
TRADES_FILE = STATE_DIR / "paper_trades.jsonl"
EQUITY_FILE = STATE_DIR / "paper_equity.jsonl"


class PaperTrader:
    def __init__(self, initial_capital: float = 10000.0,
                 max_open: int = 2,
                 kelly_fraction: float = 0.25,
                 max_risk_per_trade: float = 0.015,
                 min_confidence: float = 0.70,
                 cb_config: dict = None,
                 max_position_pct: float = 0.25,
                 min_stop_distance_pct: float = 0.005):
        self.portfolio = Portfolio(initial_capital=initial_capital, max_open=max_open)
        self.kelly_fraction = kelly_fraction
        self.max_risk_per_trade = max_risk_per_trade
        self.min_confidence = min_confidence
        self.max_position_pct = max_position_pct
        self.min_stop_distance_pct = min_stop_distance_pct

        cb_cfg = cb_config or {}
        self.cb = CircuitBreaker(
            max_consecutive_losses=cb_cfg.get("max_consecutive_losses", 3),
            max_daily_loss_pct=cb_cfg.get("max_daily_loss_pct", 0.05),
            halt_duration_hours=cb_cfg.get("halt_duration_hours", 24),
        )
        self.last_processed_ts: Optional[str] = None
        self._bars_since_last_trade = 999
        self._load_state()

    def _load_state(self):
        if PORTFOLIO_FILE.exists():
            try:
                data = json.loads(PORTFOLIO_FILE.read_text(encoding="utf-8"))
                self.portfolio.capital = data.get("capital", self.portfolio.capital)
                self.last_processed_ts = data.get("last_processed_ts")
                for p_dict in data.get("open_positions", []):
                    try:
                        pos = Position(**p_dict)
                        self.portfolio.positions.append(pos)
                    except Exception:
                        pass
                logger.info(f"Restored: capital={self.portfolio.capital:.2f}, "
                            f"open={len(self.portfolio.open_positions)}")
            except Exception as e:
                logger.warning(f"State restore failed: {e}")

    def _save_state(self):
        data = {
            "capital": self.portfolio.capital,
            "last_processed_ts": self.last_processed_ts,
            "open_positions": [p.to_dict() for p in self.portfolio.open_positions],
            "circuit_breaker": self.cb.status(),
            "saved_at": datetime.now(timezone.utc).isoformat(),
        }
        PORTFOLIO_FILE.write_text(
            json.dumps(data, default=str, ensure_ascii=False),
            encoding="utf-8"
        )

    def _append_trade_log(self, pos: Position):
        with open(TRADES_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(pos.to_dict(), default=str, ensure_ascii=False) + "\n")

    def _append_equity(self, timestamp, equity: float):
        with open(EQUITY_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps({"ts": str(timestamp), "equity": float(equity)}) + "\n")

    def _compute_risk_pct(self, confidence: float) -> float:
        stats = self.portfolio.stats()
        if stats["total_trades"] >= 10 and stats["win_loss_ratio"] > 0:
            return fractional_kelly(
                win_rate=stats["win_rate"],
                win_loss_ratio=stats["win_loss_ratio"],
                fraction=self.kelly_fraction,
                max_risk=self.max_risk_per_trade,
            )
        return confidence_based_size(
            confidence, base_size=0.005, max_size=self.max_risk_per_trade
        )

    def process_candle(self, timestamp, candle, signal: Optional[Signal]):
        self._bars_since_last_trade += 1

        # ۱. چک بسته شدن پوزیشن‌های باز
        n_before = len(self.portfolio.open_positions)
        self.portfolio.check_exits(
            current_high=float(candle["high"]),
            current_low=float(candle["low"]),
            current_close=float(candle["close"]),
        )
        n_after = len(self.portfolio.open_positions)

        if n_after < n_before:
            last = self.portfolio.closed_positions[-1]
            self.cb.record_trade(last.pnl_pct or 0)
            self._append_trade_log(last)
            logger.info(f"PAPER CLOSE: {last.direction} "
                        f"PnL={last.pnl_pct*100:+.2f}% ({last.pnl_amount:+.2f})")

        # ۲. Circuit Breaker
        if not self.cb.can_trade():
            self.last_processed_ts = str(timestamp)
            self._append_equity(timestamp, self.portfolio.capital)
            self._save_state()
            return

        # ۳. سیگنال HOLD
        if signal is None or signal.signal == SignalType.HOLD:
            self.last_processed_ts = str(timestamp)
            self._append_equity(timestamp, self.portfolio.capital)
            self._save_state()
            return

        # ۴. حداقل فاصله بین معاملات
        if self._bars_since_last_trade < 3:
            self.last_processed_ts = str(timestamp)
            self._append_equity(timestamp, self.portfolio.capital)
            self._save_state()
            return

        # ۵. اطمینان پایین
        if signal.confidence < self.min_confidence:
            self.last_processed_ts = str(timestamp)
            self._append_equity(timestamp, self.portfolio.capital)
            self._save_state()
            return

        # ۶. امکان باز کردن
        can, reason = self.portfolio.can_open(signal.signal.value)
        if not can:
            self.last_processed_ts = str(timestamp)
            self._append_equity(timestamp, self.portfolio.capital)
            self._save_state()
            return

        # ۷. محاسبه ریسک
        risk_pct = self._compute_risk_pct(signal.confidence)
        if risk_pct <= 0:
            self.last_processed_ts = str(timestamp)
            self._append_equity(timestamp, self.portfolio.capital)
            self._save_state()
            return

        # ۸. محاسبه اندازه با محدودیت‌ها
        sizing = compute_position_value(
            capital=self.portfolio.available_capital,
            risk_pct=risk_pct,
            entry_price=signal.current_price,
            stop_loss=signal.stop_loss,
            max_position_pct=self.max_position_pct,
            min_stop_distance_pct=self.min_stop_distance_pct,
        )

        if sizing["quantity"] > 0:
            self.portfolio.open_position(
                symbol="BTCUSDT",
                direction=signal.signal.value,
                entry_price=signal.current_price,
                quantity=sizing["quantity"],
                position_value=sizing["position_value"],
                stop_loss=sizing["stop_loss"],       # ← SL جدید از sizing
                take_profit=signal.take_profit,
                confidence=signal.confidence,
                regime=signal.regime,
            )
            self._bars_since_last_trade = 0
            logger.info(f"PAPER OPEN: {signal.signal.value} @ {signal.current_price:.2f} "
                        f"qty={sizing['quantity']:.6f} "
                        f"value=${sizing['position_value']:.2f} "
                        f"SL={sizing['stop_loss']:.2f} TP={signal.take_profit:.2f} "
                        f"risk=${sizing['risk_amount']:.2f}")

        self.last_processed_ts = str(timestamp)
        self._append_equity(timestamp, self.portfolio.capital)
        self._save_state()

    def record_anomaly(self, score: float, is_anomaly: bool):
        self.cb.record_anomaly(score, is_anomaly)

    def status(self) -> dict:
        return {
            "capital": round(self.portfolio.capital, 2),
            "available_capital": round(self.portfolio.available_capital, 2),
            "open_positions": len(self.portfolio.open_positions),
            "total_trades": len(self.portfolio.closed_positions),
            "last_processed_ts": self.last_processed_ts,
            "circuit_breaker": self.cb.status(),
            "stats": self.portfolio.stats(),
        }