"""شبیه‌سازی معامله با مدیریت ریسک"""
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
