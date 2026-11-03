"""تشخیص ناهنجاری با Isolation Forest"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from loguru import logger

MODEL_DIR = Path("data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def build_flow_features(df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """ساخت ویژگی‌های جریان پول از OHLCV"""
    features = pd.DataFrame(index=df.index)

    features["volume"] = df["volume"]
    features["volume_ratio"] = df["volume"] / df["volume"].rolling(window).mean()
    features["volume_std"] = df["volume"].rolling(window).std()
    features["volume_z"] = (
        (df["volume"] - df["volume"].rolling(window).mean())
        / df["volume"].rolling(window).std()
    )
    features["candle_range"] = (df["high"] - df["low"]) / df["close"]
    features["return_abs"] = df["close"].pct_change().abs()
    features["high_low_ratio"] = df["high"] / df["low"]

    return features.dropna()


class FlowAnomalyDetector:
    """تشخیص ناهنجاری جریان پول با Isolation Forest"""

    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.model: IsolationForest = None
        self.scaler: StandardScaler = None
        self.feature_cols: list = None

    def fit(self, df: pd.DataFrame, window: int = 60):
        features = build_flow_features(df, window=window)
        self.feature_cols = list(features.columns)

        logger.info(f"Training Isolation Forest on {len(features)} samples, "
                    f"{len(self.feature_cols)} features")

        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(features.values)

        self.model = IsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=200,
            n_jobs=-1,
        )
        self.model.fit(X)

        preds = self.model.predict(X)
        anomaly_rate = (preds == -1).mean() * 100
        logger.info(f"Isolation Forest trained. Anomaly rate: {anomaly_rate:.2f}%")
        return self

    def score(self, df: pd.DataFrame, window: int = 60) -> pd.DataFrame:
        """امتیاز ناهنجاری برای هر ردیف"""
        features = build_flow_features(df, window=window)
        X = self.scaler.transform(features.values)
        raw = self.model.score_samples(X)          # بالاتر = نرمال‌تر
        anomaly = -raw                              # بالاتر = ناهنجارتر
        preds = self.model.predict(X)              # -1 یا 1

        result = pd.DataFrame({
            "anomaly_score": anomaly,
            "is_anomaly": preds == -1,
        }, index=features.index)
        return result

    def current_score(self, df: pd.DataFrame, window: int = 60) -> dict:
        """امتیاز ناهنجاری آخرین کندل"""
        result = self.score(df, window=window)
        if result.empty:
            return {"anomaly_score": 0.0, "is_anomaly": False}
        last = result.iloc[-1]
        return {
            "anomaly_score": float(last["anomaly_score"]),
            "is_anomaly": bool(last["is_anomaly"]),
        }

    def save(self, path: str = None):
        path = Path(path) if path else MODEL_DIR / "flow_anomaly.pkl"
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "feature_cols": self.feature_cols,
            "contamination": self.contamination,
        }, path)
        logger.info(f"Saved anomaly detector -> {path}")

    @classmethod
    def load(cls, path: str = None):
        path = Path(path) if path else MODEL_DIR / "flow_anomaly.pkl"
        data = joblib.load(path)
        instance = cls(contamination=data["contamination"])
        instance.model = data["model"]
        instance.scaler = data["scaler"]
        instance.feature_cols = data["feature_cols"]
        logger.info(f"Loaded anomaly detector from {path}")
        return instance
