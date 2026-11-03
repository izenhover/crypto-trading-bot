# scripts/analyze_scores.py
"""تحلیل توزیع امتیازهای Ensemble روی داده تست"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

import numpy as np
import pandas as pd
from loguru import logger

from src.feature_engineering.pipeline import get_processed_path
from src.models import Ensemble
from src.models.dataset import get_feature_columns
from src.utils.logger import setup_logger


def main():
    setup_logger("analyze")
    df = pd.read_parquet(get_processed_path("BTCUSDT", "1h"))
    feature_cols = get_feature_columns(df)
    ensemble = Ensemble.load_all(include_lstm=False)

    # محاسبه امتیازها روی ۵۰۰ کندل آخر
    recent = df.tail(500)
    scores = []
    for i in range(60, len(recent)):
        X = recent[feature_cols].iloc[i:i+1]
        X_arr = np.ascontiguousarray(X.to_numpy(dtype=np.float32))
        X_df = pd.DataFrame(X_arr, columns=feature_cols)
        s = float(ensemble.predict_proba(X_df)[0])
        scores.append(s)

    scores = np.array(scores)

    logger.info("=" * 60)
    logger.info("📊 DISTRIBUTION OF ENSEMBLE SCORES (last 500 candles)")
    logger.info("=" * 60)
    logger.info(f"Min:      {scores.min():.4f}")
    logger.info(f"Max:      {scores.max():.4f}")
    logger.info(f"Mean:     {scores.mean():.4f}")
    logger.info(f"Median:   {np.median(scores):.4f}")
    logger.info(f"Std:      {scores.std():.4f}")
    logger.info("")
    logger.info("Percentiles:")
    for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
        logger.info(f"  {p:>3}%: {np.percentile(scores, p):.4f}")
    logger.info("")
    logger.info(f"Above 0.85 (BUY threshold):    {(scores >= 0.85).sum()} times ({(scores >= 0.85).mean()*100:.1f}%)")
    logger.info(f"Below 0.15 (SELL threshold):   {(scores <= 0.15).sum()} times ({(scores <= 0.15).mean()*100:.1f}%)")
    logger.info(f"Above 0.65:                     {(scores >= 0.65).sum()} times ({(scores >= 0.65).mean()*100:.1f}%)")
    logger.info(f"Below 0.35:                     {(scores <= 0.35).sum()} times ({(scores <= 0.35).mean()*100:.1f}%)")
    logger.info("=" * 60)

    # ذخیره
    out = Path("data/reports/scores_distribution.parquet")
    pd.DataFrame({"score": scores}).to_parquet(out)
    logger.info(f"Saved -> {out}")


if __name__ == "__main__":
    main()