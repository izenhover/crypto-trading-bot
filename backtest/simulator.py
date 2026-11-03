# backtest/simulator.py
"""شبیه‌ساز معاملات با محدودیت‌های ایمنی"""
from dataclasses import dataclass
from typing import Optional
import pandas as pd
from loguru import logger

from src.signal_engine import Signal, SignalType
from src.risk_management import (
    fractional_kelly, confidence_based_size,
    compute_position_value, Portfolio,
)
from src.money_flow import CircuitBreaker


@dataclass
class SimConfig:
    initial_capital: float = 10000.0
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005
    max_open_positions: int = 2
    kelly_fraction: float = 0.25
    max_risk_per_trade: float = 0.015
    min_confidence: float = 0.70
    max_consecutive_losses: int = 3
    max_daily_loss_pct: float = 0.05
    halt_hours: int = 24
    min_bars_between_trades: int = 3
    max_position_pct: float = 0.25
    min_stop_distance_pct: float = 0.005


class Simulator:
    def __init__(self, config: SimConfig):
        self.config = config
        self.portfolio = Portfolio(
            initial_capital=config.initial_capital,
            max_open=config.max_open_positions,
        )
        self.cb = CircuitBreaker(
            max_consecutive_losses=config.max_consecutive_losses,
            max_daily_loss_pct=config.max_daily_loss_pct,
            halt_duration_hours=config.halt_hours,
        )
        self.equity_curve = []
        self._bars_since_last_trade = 999

    def _apply_costs(self, price: float, direction: str, is_entry: bool) -> float:
        total = self.config.commission_pct + self.config.slippage_pct
        if direction.upper() == "BUY":
            return price * (1 + total) if is_entry else price * (1 - total)
        return price * (1 - total) if is_entry else price * (1 + total)

    def process_candle(self, timestamp, candle: pd.Series, signal: Optional[Signal]):
        self._bars_since_last_trade += 1

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

        if not self.cb.can_trade():
            self.equity_curve.append((timestamp, self.portfolio.capital))
            return

        if signal is None or signal.signal == SignalType.HOLD:
            self.equity_curve.append((timestamp, self.portfolio.capital))
            return

        if self._bars_since_last_trade < self.config.min_bars_between_trades:
            self.equity_curve.append((timestamp, self.portfolio.capital))
            return

        if signal.confidence < self.config.min_confidence:
            self.equity_curve.append((timestamp, self.portfolio.capital))
            return

        can, reason = self.portfolio.can_open(signal.signal.value)
        if not can:
            self.equity_curve.append((timestamp, self.portfolio.capital))
            return

        risk_pct = self._compute_risk_pct(signal)
        if risk_pct <= 0:
            self.equity_curve.append((timestamp, self.portfolio.capital))
            return

        entry_price = self._apply_costs(candle["close"], signal.signal.value, is_entry=True)

        # ★ محاسبه با محدودیت‌ها ★
        sizing = compute_position_value(
            capital=self.portfolio.available_capital,
            risk_pct=risk_pct,
            entry_price=entry_price,
            stop_loss=signal.stop_loss,
            max_position_pct=self.config.max_position_pct,
            min_stop_distance_pct=self.config.min_stop_distance_pct,
        )

        if sizing["quantity"] > 0:
            self.portfolio.open_position(
                symbol="BTCUSDT",
                direction=signal.signal.value,
                entry_price=entry_price,
                quantity=sizing["quantity"],
                position_value=sizing["position_value"],
                stop_loss=sizing["stop_loss"],       # ← SL جدید
                take_profit=signal.take_profit,
                confidence=signal.confidence,
                regime=signal.regime,
            )
            self._bars_since_last_trade = 0

        self.equity_curve.append((timestamp, self.portfolio.capital))

    def record_anomaly(self, anomaly_score: float, is_anomaly: bool):
        self.cb.record_anomaly(anomaly_score, is_anomaly)

    def _compute_risk_pct(self, signal: Signal) -> float:
        stats = self.portfolio.stats()
        if stats["total_trades"] >= 10 and stats["win_loss_ratio"] > 0:
            return fractional_kelly(
                win_rate=stats["win_rate"],
                win_loss_ratio=stats["win_loss_ratio"],
                fraction=self.config.kelly_fraction,
                max_risk=self.config.max_risk_per_trade,
            )
        return confidence_based_size(
            signal.confidence, base_size=0.005,
            max_size=self.config.max_risk_per_trade,
        )

    def finalize(self, last_price: float):
        for pos in list(self.portfolio.open_positions):
            self.portfolio.close_position(pos, last_price)

    def get_equity_series(self) -> pd.Series:
        if not self.equity_curve:
            return pd.Series(dtype=float)
        idx, vals = zip(*self.equity_curve)
        return pd.Series(vals, index=pd.Index(idx, name="timestamp"))

    def get_trades_df(self) -> pd.DataFrame:
        if not self.portfolio.closed_positions:
            return pd.DataFrame()
        return pd.DataFrame([p.to_dict() for p in self.portfolio.closed_positions])