"""ترکیب مدل‌ها با وزن‌دهی ساده"""
from pathlib import Path
import numpy as np
import pandas as pd
from loguru import logger

MODEL_DIR = Path("data/models")

DEFAULT_WEIGHTS = {
    "xgboost": 0.35,
    "lightgbm": 0.35,
    "random_forest": 0.15,
    "lstm": 0.15,
}


class Ensemble:
    def __init__(self, models: dict, weights: dict = None):
        """
        models: دیکشنری {name: model_instance}
        weights: دیکشنری وزن‌ها (اگر ندهید، از DEFAULT_WEIGHTS استفاده می‌شود)
        """
        self.models = models
        weights = weights or DEFAULT_WEIGHTS
        # نرمال‌سازی وزن‌ها
        total = sum(weights.get(n, 0) for n in models)
        self.weights = {n: weights.get(n, 0) / total for n in models}
        logger.info(f"Ensemble weights: {self.weights}")

    def predict_proba(self, X) -> np.ndarray:
        """پیش‌بینی ترکیبی"""
        result = np.zeros(len(X), dtype=np.float32)
        for name, model in self.models.items():
            p = model.predict_proba(X)
            w = self.weights[name]
            result += w * p
            logger.debug(f"  {name}: w={w:.3f}, mean={p.mean():.4f}")
        return result

    @classmethod
    def load_all(cls, weights: dict = None, include_lstm: bool = True):
        """بارگذاری همه مدل‌ها از دیسک"""
        from src.models.xgboost_model import XGBoostModel
        from src.models.lightgbm_model import LightGBMModel
        from src.models.random_forest_model import RandomForestModel
        from src.models.lstm_model import LSTMModel

        models = {}
        try:
            models["xgboost"] = XGBoostModel.load()
        except Exception as e:
            logger.warning(f"XGBoost load failed: {e}")
        try:
            models["lightgbm"] = LightGBMModel.load()
        except Exception as e:
            logger.warning(f"LightGBM load failed: {e}")
        try:
            models["random_forest"] = RandomForestModel.load()
        except Exception as e:
            logger.warning(f"RandomForest load failed: {e}")
        if include_lstm:
            try:
                models["lstm"] = LSTMModel.load()
            except Exception as e:
                logger.warning(f"LSTM load failed: {e}")

        if not models:
            raise RuntimeError("No models could be loaded")

        return cls(models, weights=weights)
