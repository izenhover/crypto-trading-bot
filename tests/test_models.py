"""تست مدل‌های آموزش‌دیده"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
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
from src.utils.logger import setup_logger

setup_logger("test")


def main():
    print()
    print("=" * 60)
    print("Testing All Models")
    print("=" * 60)

    X, y, _ = prepare_dataset("BTCUSDT", "1h")
    (_, _), (_, _), (X_te, y_te) = train_val_test_split(X, y)

    # بارگذاری
    xgb_model = XGBoostModel.load()
    lgb_model = LightGBMModel.load()
    rf_model = RandomForestModel.load()
    lstm_model = LSTMModel.load()

    print("[OK] All models loaded")

    # پیش‌بینی
    print()
    print("Individual models (test set):")
    for name, m in [("xgboost", xgb_model), ("lightgbm", lgb_model),
                    ("random_forest", rf_model), ("lstm", lstm_model)]:
        proba = m.predict_proba(X_te)
        if name == "lstm":
            seq = lstm_model.seq_len
            pv = proba[seq - 1:]
            yv = y_te.values[seq - 1:]
        else:
            pv = proba
            yv = y_te.values
        pred = (pv >= 0.5).astype(int)
        acc = accuracy_score(yv, pred)
        try:
            auc = roc_auc_score(yv, pv)
        except ValueError:
            auc = float("nan")
        print(f"  {name:<15}: acc={acc:.4f} auc={auc:.4f}")

    # Ensemble
    print()
    print("Ensemble (test set):")
    ens = Ensemble({
        "xgboost": xgb_model,
        "lightgbm": lgb_model,
        "random_forest": rf_model,
        "lstm": lstm_model,
    })
    proba = ens.predict_proba(X_te)
    seq = lstm_model.seq_len
    pv = proba[seq - 1:]
    yv = y_te.values[seq - 1:]
    pred = (pv >= 0.5).astype(int)
    acc = accuracy_score(yv, pred)
    try:
        auc = roc_auc_score(yv, pv)
    except ValueError:
        auc = float("nan")
    print(f"  {'ensemble':<15}: acc={acc:.4f} auc={auc:.4f}")

    print()
    print("=" * 60)
    print("All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
