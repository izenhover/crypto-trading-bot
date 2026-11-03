"""آموزش مدل تشخیص ناهنجاری جریان پول"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger

from src.feature_engineering.pipeline import get_processed_path
from src.money_flow import FlowAnomalyDetector
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger


def main():
    setup_logger("anomaly")
    config = load_config()
    contamination = config["money_flow"]["anomaly"]["contamination"]

    symbol, interval = "BTCUSDT", "1h"
    df = pd.read_parquet(get_processed_path(symbol, interval))
    logger.info(f"Loaded {len(df)} rows from {symbol} {interval}")

    detector = FlowAnomalyDetector(contamination=contamination)
    detector.fit(df)
    detector.save()

    # نمونه امتیازها
    scores = detector.score(df)
    n_anom = int(scores["is_anomaly"].sum())
    logger.info(f"Detected {n_anom} anomalies ({n_anom/len(scores)*100:.2f}%)")

    current = detector.current_score(df)
    logger.info(f"Current anomaly score: {current['anomaly_score']:.4f} "
                f"(is_anomaly={current['is_anomaly']})")

    # ذخیره امتیازها
    out = Path("data/processed") / f"{symbol}_{interval}_anomaly.parquet"
    scores.to_parquet(out)
    logger.info(f"Saved anomaly scores -> {out}")


if __name__ == "__main__":
    main()
