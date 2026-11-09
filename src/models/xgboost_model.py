"""مدل XGBoost"""
from pathlib import Path
import joblib
import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score
from loguru import logger

MODEL_DIR = Path("data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


class XGBoostModel:
    name = "xgboost"

    def __init__(self, **params):
        defaults = dict(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
        )
        defaults.update(params)
        self.model = xgb.XGBClassifier(**defaults)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        if X_val is not None and y_val is not None:
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            self.model.fit(X_train, y_train)
        logger.info(f"[{self.name}] Trained")
        return self

    def predict_proba(self, X) -> np.ndarray:
        """احتمال کلاس 1 (صعودی)"""
        return self.model.predict_proba(X)[:, 1]

    def evaluate(self, X, y):
        proba = self.predict_proba(X)
        pred = (proba >= 0.5).astype(int)
        acc = accuracy_score(y, pred)
        try:
            auc = roc_auc_score(y, proba)
        except ValueError:
            auc = float("nan")
        logger.info(f"[{self.name}] acc={acc:.4f} auc={auc:.4f}")
        return {"accuracy": acc, "auc": auc}

    def save(self, path: str = None):
        path = Path(path) if path else MODEL_DIR / f"{self.name}.pkl"
        joblib.dump(self.model, path)
        logger.info(f"[{self.name}] Saved -> {path}")

    @classmethod
    def load(cls, path: str = None):
        path = Path(path) if path else MODEL_DIR / f"{cls.name}.pkl"
        instance = cls()
        instance.model = joblib.load(path)
        logger.info(f"[{cls.name}] Loaded from {path}")
        return instance
