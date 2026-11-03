# scripts/train_models.py
"""آموزش همه مدل‌ها با تنظیمات بهبودیافته"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, roc_auc_score
from loguru import logger

from src.models import (
    prepare_dataset,
    train_val_test_split,
    XGBoostModel,
    LightGBMModel,
    RandomForestModel,
    LSTMModel,
    Ensemble,
)
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger


HORIZON = 3
THRESHOLD = 0.001


def _safe_auc(y, proba):
    try:
        return roc_auc_score(y, proba)
    except ValueError:
        return float("nan")


def main():
    setup_logger("train")
    config = load_config()
    mc = config["models"]

    symbol = "BTCUSDT"
    interval = "1h"
    logger.info(f"=== Training for {symbol} {interval} ===")
    logger.info(f"Target: horizon={HORIZON} candles, threshold={THRESHOLD*100}%")

    # داده
    X, y, feature_cols, regimes = prepare_dataset(
        symbol, interval, horizon=HORIZON, threshold=THRESHOLD
    )
    splits = train_val_test_split(X, y, regimes=regimes)

    X_tr, y_tr = splits["train"][0], splits["train"][1]
    X_va, y_va = splits["val"][0], splits["val"][1]
    X_te, y_te = splits["test"][0], splits["test"][1]

    logger.info(f"X_tr={X_tr.shape}, X_va={X_va.shape}, X_te={X_te.shape}")

    # ---------- XGBoost ----------
    logger.info("--- XGBoost ---")
    xgb_cfg = dict(mc["xgboost"])
    pos = (y_tr == 1).sum()
    neg = (y_tr == 0).sum()
    xgb_cfg["scale_pos_weight"] = float(neg / max(pos, 1))
    xgb_model = XGBoostModel(**xgb_cfg)
    xgb_model.fit(X_tr, y_tr, X_va, y_va)
    logger.info("Val:")
    xgb_model.evaluate(X_va, y_va)
    test_xgb = xgb_model.evaluate(X_te, y_te)
    xgb_model.save()

    # ---------- LightGBM ----------
    logger.info("--- LightGBM ---")
    lgb_cfg = dict(mc["lightgbm"])
    lgb_cfg["class_weight"] = "balanced"
    lgb_model = LightGBMModel(**lgb_cfg)
    lgb_model.fit(X_tr, y_tr, X_va, y_va)
    logger.info("Val:")
    lgb_model.evaluate(X_va, y_va)
    test_lgb = lgb_model.evaluate(X_te, y_te)
    lgb_model.save()

    # ---------- Random Forest ----------
    logger.info("--- Random Forest ---")
    rf_model = RandomForestModel(**mc["random_forest"])
    rf_model.fit(X_tr, y_tr)
    logger.info("Val:")
    rf_model.evaluate(X_va, y_va)
    test_rf = rf_model.evaluate(X_te, y_te)
    rf_model.save()

    # ---------- LSTM ----------
    logger.info("--- LSTM ---")
    lstm_cfg = mc["lstm"]
    lstm_model = LSTMModel(
        input_size=X_tr.shape[1],
        hidden_size=lstm_cfg["hidden_size"],
        num_layers=lstm_cfg["num_layers"],
        dropout=lstm_cfg["dropout"],
        seq_len=lstm_cfg["sequence_length"],
        epochs=lstm_cfg["epochs"],
        batch_size=lstm_cfg["batch_size"],
        learning_rate=lstm_cfg["learning_rate"],
    )
    lstm_model.fit(X_tr, y_tr, X_va, y_va)
    logger.info("Val:")
    lstm_model.evaluate(X_va, y_va)
    test_lstm = lstm_model.evaluate(X_te, y_te)
    lstm_model.save()

    # ---------- Ensemble ----------
    logger.info("--- Ensemble (Test) ---")
    ensemble = Ensemble({
        "xgboost": xgb_model,
        "lightgbm": lgb_model,
        "random_forest": rf_model,
        "lstm": lstm_model,
    })
    proba = ensemble.predict_proba(X_te)

    seq_len = lstm_cfg["sequence_length"]
    proba_v = proba[seq_len - 1:]
    y_v = y_te.values[seq_len - 1:]
    pred = (proba_v >= 0.5).astype(int)
    ens_acc = accuracy_score(y_v, pred)
    ens_auc = _safe_auc(y_v, proba_v)
    test_ens = {"accuracy": ens_acc, "auc": ens_auc}
    logger.info(f"[ensemble] acc={ens_acc:.4f} auc={ens_auc:.4f}")

    # ---------- خلاصه ----------
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Test Set Results (horizon={HORIZON}, threshold={THRESHOLD})")
    logger.info("=" * 60)
    for name, m in [("xgboost", test_xgb), ("lightgbm", test_lgb),
                    ("random_forest", test_rf), ("lstm", test_lstm),
                    ("ensemble", test_ens)]:
        logger.info(f"  {name:<15}: acc={m['accuracy']:.4f}  auc={m['auc']:.4f}")
    logger.info("=" * 60)
    logger.info("All models trained and saved!")


if __name__ == "__main__":
    main()