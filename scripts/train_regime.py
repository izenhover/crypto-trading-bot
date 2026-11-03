"""آموزش HMM برای تشخیص رژیم بازار"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger

from src.feature_engineering.pipeline import get_processed_path
from src.regime_detection import RegimeDetector, regime_distribution, regime_transitions, regime_durations
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger


def main():
    setup_logger("regime")
    config = load_config()
    n_regimes = config["regime"]["n_regimes"]

    # استفاده از BTCUSDT 1h به عنوان نماینده بازار
    symbol = "BTCUSDT"
    interval = "1h"

    logger.info(f"Training regime detector on {symbol} {interval}")
    df = pd.read_parquet(get_processed_path(symbol, interval))

    detector = RegimeDetector(n_regimes=n_regimes)
    detector.fit(df)
    detector.save()

    # ارزیابی
    regimes = detector.predict(df)
    logger.info("")
    logger.info("=== Regime Distribution ===")
    logger.info("\n" + regime_distribution(regimes).to_string())

    logger.info("")
    logger.info("=== Regime Transition Matrix ===")
    logger.info("\n" + regime_transitions(regimes).to_string())

    logger.info("")
    logger.info("=== Regime Durations (in candles) ===")
    logger.info("\n" + regime_durations(regimes).to_string())

    # رژیم فعلی
    current = regimes.iloc[-1]
    logger.info("")
    logger.info(f"Current regime: {current}")

    # ذخیره رژیم‌ها در فایل
    regimes.to_frame().to_parquet(Path("data/processed") / f"{symbol}_{interval}_regimes.parquet")
    logger.info(f"Saved regimes -> data/processed/{symbol}_{interval}_regimes.parquet")


if __name__ == "__main__":
    main()
