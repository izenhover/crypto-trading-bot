# scripts/run_backtest.py
"""اجرای بک‌تست فقط روی داده تست با Circuit Breaker داخلی Simulator"""
import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

import pandas as pd
import numpy as np
from loguru import logger

from src.feature_engineering.pipeline import get_processed_path
from src.models import Ensemble
from src.models.dataset import (
    get_feature_columns,
    prepare_dataset,
    train_val_test_split,
)
from src.regime_detection import RegimeDetector
from src.money_flow import FlowAnomalyDetector
from src.signal_engine import SignalGenerator
from backtest import (
    Simulator, SimConfig, compute_all_metrics,
    print_metrics, save_report, summarize_trades,
    generate_walk_forward_report,
)
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger


TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
HORIZON = 3
THRESHOLD = 0.001


def main():
    setup_logger("backtest")
    config = load_config()

    symbol, interval = "BTCUSDT", "1h"
    df = pd.read_parquet(get_processed_path(symbol, interval))
    logger.info(f"Loaded {len(df)} rows")

    # ---------- محاسبه مرز داده تست (همان splits آموزش) ----------
    X_full, y_full, feature_cols, regimes = prepare_dataset(
        symbol, interval, horizon=HORIZON, threshold=THRESHOLD
    )
    splits = train_val_test_split(X_full, y_full, regimes=regimes)

    X_train = splits["train"][0]
    X_val = splits["val"][0]
    X_test = splits["test"][0]

    test_start_ts = X_test.index[0]
    test_end_ts = X_test.index[-1]

    test_start_idx = df.index.get_loc(test_start_ts)
    test_end_idx = df.index.get_loc(test_end_ts)

    logger.info(f"Training range: up to {X_train.index[-1]}")
    logger.info(f"Validation range: {X_val.index[0]} to {X_val.index[-1]}")
    logger.info(f"TEST range (backtest): {test_start_ts} to {test_end_ts}")
    logger.info(f"Test candles: {test_end_idx - test_start_idx + 1}")

    if test_end_idx - test_start_idx < 50:
        logger.error("Not enough test data for meaningful backtest")
        return

    # ---------- بارگذاری ماژول‌ها ----------
    ensemble = Ensemble.load_all(include_lstm=False)
    regime_detector = RegimeDetector.load()
    anomaly_detector = FlowAnomalyDetector.load()

    sig_cfg = config["signal"]
    risk_cfg = config["risk"]
    bt_cfg = config["backtest"]
    cb_cfg = risk_cfg["circuit_breaker"]

    generator = SignalGenerator(
        threshold_buy=sig_cfg["threshold_buy"],
        threshold_sell=sig_cfg["threshold_sell"],
        min_confirmations=sig_cfg["min_confirmations"],
        atr_stop_multiplier=risk_cfg["atr_stop_multiplier"],
        atr_take_profit_multiplier=risk_cfg["atr_take_profit_multiplier"],
    )

    # ---------- ساخت Simulator با Circuit Breaker داخلی ----------
    sim_config = SimConfig(
        initial_capital=bt_cfg["initial_capital"],
        commission_pct=bt_cfg["commission_pct"],
        slippage_pct=bt_cfg["slippage_pct"],
        max_open_positions=risk_cfg["max_open_positions"],
        kelly_fraction=risk_cfg["kelly_fraction"],
        max_risk_per_trade=risk_cfg["max_risk_per_trade"],
        min_confidence=sig_cfg["threshold_buy"],  # حداقل اطمینان = آستانه خرید
        max_consecutive_losses=cb_cfg["max_consecutive_losses"],
        max_daily_loss_pct=cb_cfg["max_daily_loss_pct"],
        halt_hours=cb_cfg["halt_duration_hours"],
        min_bars_between_trades=3,
    )
    simulator = Simulator(sim_config)

    # ---------- Walk-Forward گزارش ----------
    wf = generate_walk_forward_report(
        n_samples=test_end_idx - test_start_idx + 1,
        train_size=int((test_end_idx - test_start_idx + 1) * 0.6),
        test_size=int((test_end_idx - test_start_idx + 1) * 0.1),
    )
    logger.info(f"Walk-Forward windows (reference): {wf['n_windows']}")

    # ---------- شبیه‌سازی روی داده تست ----------
    logger.info(f"Simulating TEST period only ({test_end_idx - test_start_idx + 1} candles)")

    signals_generated = {"BUY": 0, "SELL": 0, "HOLD": 0}

    # warm-up حداقلی برای HMM و anomaly
    warmup = 200
    real_start = max(test_start_idx, warmup)

    for i in range(real_start, test_end_idx + 1):
        current_df = df.iloc[:i + 1]
        candle = df.iloc[i]

        # اگر Circuit Breaker فعال است، فقط چک بسته شدن پوزیشن‌ها
        if not simulator.cb.can_trade():
            simulator.process_candle(candle.name, candle, None)
            continue

        # پیش‌بینی Ensemble
        X_arr = np.ascontiguousarray(
            current_df[feature_cols].iloc[-1:].to_numpy(dtype=np.float32)
        )
        X_df = pd.DataFrame(X_arr, columns=feature_cols)
        ensemble_score = float(ensemble.predict_proba(X_df)[0])

        # رژیم فعلی
        try:
            regime = regime_detector.predict_current(current_df, window=150)
        except Exception:
            regime = "UNKNOWN"

        # ناهنجاری جریان
        try:
            anomaly = anomaly_detector.current_score(current_df)
        except Exception:
            anomaly = {"anomaly_score": 0.0, "is_anomaly": False}

        # ثبت ناهنجاری در Circuit Breaker داخلی Simulator
        simulator.record_anomaly(anomaly["anomaly_score"], anomaly["is_anomaly"])

        # تولید سیگنال
        signal = generator.generate(
            df=current_df,
            ensemble_score=ensemble_score,
            regime=regime,
            anomaly_score=anomaly["anomaly_score"],
            is_anomaly=anomaly["is_anomaly"],
            circuit_breaker_active=not simulator.cb.can_trade(),
        )

        signals_generated[signal.signal.value] += 1
        simulator.process_candle(candle.name, candle, signal)

    # بستن پوزیشن‌های باز در پایان
    simulator.finalize(df.iloc[test_end_idx]["close"])

    # ---------- معیارها ----------
    equity = simulator.get_equity_series()
    trades = simulator.get_trades_df()

    metrics = compute_all_metrics(equity, trades, periods_per_year=8760)
    metrics["signals_buy"] = signals_generated["BUY"]
    metrics["signals_sell"] = signals_generated["SELL"]
    metrics["signals_hold"] = signals_generated["HOLD"]
    metrics["test_period_days"] = round(
        (test_end_ts - test_start_ts).total_seconds() / 86400, 1
    )
    metrics["cb_halted_count"] = int(
        sum(1 for _ in [] )  # placeholder — قابل حذف
    ) if False else 0  # نگه داشتن ساختار

    print_metrics(metrics, title="Backtest Results — TEST SET ONLY")
    save_report(metrics, trades, equity, name=f"{symbol}_{interval}_backtest_clean")

    # ---------- آمار Circuit Breaker ----------
    cb_status = simulator.cb.status()
    logger.info("")
    logger.info("=== Circuit Breaker Status ===")
    logger.info(f"  halted                 : {cb_status['halted']}")
    logger.info(f"  consecutive_losses     : {cb_status['consecutive_losses']}")
    logger.info(f"  daily_pnl              : {cb_status['daily_pnl']:.4f}")
    logger.info(f"  reason                 : {cb_status['reason']}")

    # ---------- خلاصه بر اساس رژیم ----------
    if not trades.empty:
        logger.info("")
        logger.info("=== Performance by Regime ===")
        logger.info("\n" + summarize_trades(trades).to_string())

        # خلاصه بر اساس جهت
        logger.info("")
        logger.info("=== Performance by Direction ===")
        by_dir = trades.groupby("direction").agg(
            trades=("pnl_amount", "size"),
            wins=("pnl_amount", lambda x: (x > 0).sum()),
            total_pnl=("pnl_amount", "sum"),
            avg_pnl=("pnl_amount", "mean"),
        ).round(4)
        by_dir["win_rate"] = (by_dir["wins"] / by_dir["trades"]).round(4)
        logger.info("\n" + by_dir.to_string())


if __name__ == "__main__":
    main()