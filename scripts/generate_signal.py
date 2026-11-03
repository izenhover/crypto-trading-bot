"""اسکریپت تست تولید سیگنال روی داده تاریخی"""
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
