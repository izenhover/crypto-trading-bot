# scripts/run_live.py
"""اجرای ربات در حالت Paper Trading (زنده) — نسخه نهایی"""
import sys
import json
import time
import warnings
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

import pandas as pd
import numpy as np
from loguru import logger

from src.execution import LiveDataFetcher, PaperTrader
from src.feature_engineering.technical import add_all_technical
from src.models import Ensemble
from src.models.dataset import get_feature_columns
from src.regime_detection import RegimeDetector
from src.money_flow import FlowAnomalyDetector
from src.signal_engine import SignalGenerator
from monitoring import TelegramAlerter
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger


SYMBOL_CCXT = "BTC/USDT"
TIMEFRAME = "1h"
INTERVAL_SECONDS = 300
WARMUP_CANDLES = 200
STATE_DIR = Path("data/state")
STATE_DIR.mkdir(parents=True, exist_ok=True)


def build_features_from_raw(df, config):
    df = df.copy()
    df = add_all_technical(df, config)
    df = df.replace([float("inf"), float("-inf")], pd.NA).dropna()
    return df


def save_latest_state(regime, signal, ensemble_score, anomaly):
    """ذخیره وضعیت لحظه‌ای برای داشبورد"""
    try:
        state = {
            "regime": regime,
            "signal": signal.signal.value,
            "confidence": float(signal.confidence),
            "confidence_level": signal.confidence_level.value,
            "confirmations": int(signal.confirmations),
            "ensemble_score": float(ensemble_score),
            "anomaly_score": float(anomaly["anomaly_score"]),
            "is_anomaly": bool(anomaly["is_anomaly"]),
            "stop_loss": float(signal.stop_loss) if signal.stop_loss else None,
            "take_profit": float(signal.take_profit) if signal.take_profit else None,
            "current_price": float(signal.current_price),
            "reasons": signal.reasons,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        (STATE_DIR / "last_signal.json").write_text(
            json.dumps(state, ensure_ascii=False, default=str),
            encoding="utf-8"
        )
    except Exception as e:
        logger.warning(f"Failed to save latest state: {e}")


def main():
    setup_logger("live")
    config = load_config()

    logger.info("=" * 60)
    logger.info("🚀 Crypto Bot — Paper Trading Live")
    logger.info(f"Symbol: {SYMBOL_CCXT} | Timeframe: {TIMEFRAME}")
    logger.info("=" * 60)

    try:
        fetcher = LiveDataFetcher(SYMBOL_CCXT, TIMEFRAME)
        logger.info(f"Connected: {fetcher.exchange_name}")
    except Exception as e:
        logger.error(f"Cannot connect: {e}")
        return

    try:
        ensemble = Ensemble.load_all(include_lstm=False)
        logger.info("Ensemble loaded")
    except Exception as e:
        logger.error(f"Ensemble failed: {e}")
        return

    try:
        regime_detector = RegimeDetector.load()
        logger.info("Regime detector loaded")
    except Exception:
        regime_detector = None

    try:
        anomaly_detector = FlowAnomalyDetector.load()
        logger.info("Anomaly detector loaded")
    except Exception:
        anomaly_detector = None

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

    trader = PaperTrader(
        initial_capital=bt_cfg["initial_capital"],
        max_open=risk_cfg["max_open_positions"],
        kelly_fraction=risk_cfg["kelly_fraction"],
        max_risk_per_trade=risk_cfg["max_risk_per_trade"],
        min_confidence=sig_cfg["threshold_buy"],
        cb_config=risk_cfg["circuit_breaker"],
        max_position_pct=0.25,
        min_stop_distance_pct=0.005,
    )

    alerter = TelegramAlerter()
    logger.info("Starting main loop")
    cycle_count = 0

    while True:
        cycle_count += 1
        logger.info(f"\n{'─'*60}")
        logger.info(f"⏱  Cycle #{cycle_count} — {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC")
        logger.info(f"{'─'*60}")

        try:
            raw_df = fetcher.fetch_recent(limit=500)
            if raw_df.empty:
                time.sleep(60)
                continue

            latest_ts = raw_df.index[-1]
            close_price = float(raw_df["close"].iloc[-1])
            logger.info(f"Latest: {latest_ts} | Close: {close_price:.2f}")

            df = build_features_from_raw(raw_df, config)
            if len(df) < WARMUP_CANDLES:
                logger.warning(f"Not enough: {len(df)}")
                time.sleep(60)
                continue

            feature_cols = get_feature_columns(df)
            logger.info(f"Features: {len(feature_cols)} cols, {len(df)} rows")

            X_last = df[feature_cols].iloc[-1:]
            X_arr = np.ascontiguousarray(X_last.to_numpy(dtype=np.float32))
            X_df = pd.DataFrame(X_arr, columns=feature_cols)
            ensemble_score = float(ensemble.predict_proba(X_df)[0])
            logger.info(f"📊 Ensemble: {ensemble_score:.4f}")

            regime = "UNKNOWN"
            if regime_detector is not None:
                try:
                    regime = regime_detector.predict_current(df, window=200)
                    logger.info(f"🎯 Regime: {regime}")
                except Exception:
                    pass

            anomaly = {"anomaly_score": 0.0, "is_anomaly": False}
            if anomaly_detector is not None:
                try:
                    anomaly = anomaly_detector.current_score(df)
                    logger.info(f"🌊 Anomaly: {anomaly['anomaly_score']:.4f}")
                except Exception:
                    pass

            trader.record_anomaly(anomaly["anomaly_score"], anomaly["is_anomaly"])

            cb_active = not trader.cb.can_trade()
            signal = generator.generate(
                df=df,
                ensemble_score=ensemble_score,
                regime=regime,
                anomaly_score=anomaly["anomaly_score"],
                is_anomaly=anomaly["is_anomaly"],
                circuit_breaker_active=cb_active,
            )

            logger.info(f"🎯 Signal: {signal.signal.value} (conf={signal.confidence:.3f})")
            for r in signal.reasons:
                logger.info(f"   • {r}")

            # ذخیره state برای داشبورد
            save_latest_state(regime, signal, ensemble_score, anomaly)

            candle = raw_df.iloc[-1]
            trader.process_candle(latest_ts, candle, signal)

            if signal.signal.value != "HOLD":
                alerter.send_signal(signal)

            status = trader.status()
            logger.info(f"💰 Capital: {status['capital']:.2f} | "
                        f"Open: {status['open_positions']} | "
                        f"Trades: {status['total_trades']}")
            logger.info(f"💤 Sleeping...")

            time.sleep(INTERVAL_SECONDS)

        except KeyboardInterrupt:
            logger.info("Stopped by user")
            trader._save_state()
            break
        except Exception as e:
            logger.exception(f"Loop error: {e}")
            time.sleep(60)


if __name__ == "__main__":
    main()
