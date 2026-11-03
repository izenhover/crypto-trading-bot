"""بازآموزی خودکار مدل‌ها"""
import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

import pandas as pd
from loguru import logger

from src.utils.versioning import archive_current_models, save_metrics_snapshot
from monitoring import log_performance
from src.utils.logger import setup_logger


def main():
    setup_logger("retrain")
    logger.info("=" * 60)
    logger.info("🤖 Starting automated retraining")
    logger.info("=" * 60)

    # ۱. بک‌آپ از مدل‌های فعلی
    tag = archive_current_models()
    logger.info(f"Archived current models with tag: {tag}")

    # ۲. دریافت داده جدید (اختیاری — اگر آنلاین هست)
    try:
        from src.execution.live_data import LiveDataFetcher
        from src.data_collection.storage import save_candles
        fetcher = LiveDataFetcher("BTC/USDT", "1h")
        new_data = fetcher.fetch_recent(limit=500)
        if not new_data.empty:
            save_candles(new_data, "BTCUSDT", "1h")
            logger.info(f"Added {len(new_data)} new candles")
    except Exception as e:
        logger.warning(f"Live data update skipped: {e}")

    # ۳. بازسازی ویژگی‌ها
    logger.info("Rebuilding features...")
    from src.feature_engineering.pipeline import build_all
    build_all()

    # ۴. بازآموزی رژیم
    logger.info("Retraining regime detector...")
    from src.regime_detection import RegimeDetector
    from src.feature_engineering.pipeline import get_processed_path
    df = pd.read_parquet(get_processed_path("BTCUSDT", "1h"))
    detector = RegimeDetector(n_regimes=4)
    detector.fit(df)
    detector.save()

    # ۵. بازآموزی anomaly
    logger.info("Retraining anomaly detector...")
    from src.money_flow import FlowAnomalyDetector
    anomaly = FlowAnomalyDetector(contamination=0.05)
    anomaly.fit(df)
    anomaly.save()

    # ۶. بازآموزی مدل‌های Ensemble
    logger.info("Retraining ensemble models...")
    import subprocess
    result = subprocess.run(
        [sys.executable, "scripts/train_models.py"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        logger.error(f"train_models failed:\n{result.stderr}")
        return
    logger.info("Ensemble retrained successfully")

    # ۷. ذخیره snapshot
    save_metrics_snapshot({"tag": tag, "retrained_at": str(pd.Timestamp.now())}, tag)
    log_performance({"event": "retrain", "tag": tag})

    logger.info("=" * 60)
    logger.info(f"✅ Retraining complete. Archive tag: {tag}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
