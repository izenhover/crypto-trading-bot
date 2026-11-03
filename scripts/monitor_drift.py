"""پایش Drift"""
import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

import pandas as pd
from loguru import logger

from monitoring import compute_reference, detect_drift, log_performance
from src.feature_engineering.pipeline import get_processed_path
from src.models.dataset import get_feature_columns
from src.utils.logger import setup_logger


def main():
    setup_logger("drift")
    df = pd.read_parquet(get_processed_path("BTCUSDT", "1h"))
    feature_cols = get_feature_columns(df)

    # اگر reference موجود نیست، از ۷۰٪ اول به‌عنوان مرجع استفاده کن
    ref_path = Path("data/drift/reference_stats.json")
    if not ref_path.exists():
        logger.info("Creating reference from first 70% of data")
        split = int(len(df) * 0.7)
        compute_reference(df.iloc[:split], feature_cols, name="reference")

    # چک drift روی ۳۰٪ آخر
    recent = df.tail(500)
    drift = detect_drift(recent, feature_cols, reference_name="reference")

    logger.info("=" * 60)
    logger.info("=== DRIFT REPORT ===")
    logger.info("=" * 60)
    logger.info(f"Has drift: {drift['has_drift']}")
    logger.info(f"Drifted features: {drift.get('drifted_features', [])}")
    logger.info(f"Drift ratio: {drift.get('drift_ratio', 0):.2%}")

    log_performance({
        "event": "drift_check",
        "has_drift": drift["has_drift"],
        "drift_ratio": drift.get("drift_ratio", 0),
    })

    if drift["has_drift"]:
        logger.warning("⚠️ Drift detected! Consider retraining.")
    else:
        logger.info("✅ No significant drift")


if __name__ == "__main__":
    main()
