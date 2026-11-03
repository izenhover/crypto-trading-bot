# setup_phase3_files.py
"""ساخت فایل‌های فاز ۳: تشخیص رژیم با HMM"""
from pathlib import Path

FILES = {}

# ---------- src/regime_detection/hmm_regime.py ----------
FILES["src/regime_detection/hmm_regime.py"] = '''"""تشخیص رژیم بازار با مدل پنهان مارکوف (HMM)"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
from loguru import logger

MODEL_DIR = Path("data/models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# برچسب‌های رژیم بر اساس میانگین بازدهی صعودی از BEAR به BULL
REGIME_LABELS = ["BEAR", "SIDEWAYS_LOW", "SIDEWAYS_HIGH", "BULL"]


class RegimeDetector:
    """تشخیص رژیم بازار با HMM گاوسی"""

    def __init__(self, n_regimes: int = 4, covariance_type: str = "full", n_iter: int = 1000, random_state: int = 42):
        self.n_regimes = n_regimes
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.model: hmm.GaussianHMM = None
        self.scaler: StandardScaler = None
        self.regime_stats: dict = {}
        self.label_map: dict = {}

    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """ساخت ۳ ویژگی استاندارد برای HMM"""
        returns = df["close"].pct_change()
        volatility = returns.rolling(20).std()
        volume_change = df["volume"].pct_change().replace([np.inf, -np.inf], np.nan)

        features = pd.DataFrame({
            "returns": returns,
            "volatility": volatility,
            "volume_change": volume_change,
        }, index=df.index)

        return features.dropna()

    def fit(self, df: pd.DataFrame):
        """آموزش HMM روی داده‌های تاریخی"""
        features = self._prepare_features(df)
        if features.empty:
            raise ValueError("No features after preparation")

        logger.info(f"Training HMM on {len(features)} samples, {self.n_regimes} regimes")

        # نرمال‌سازی
        self.scaler = StandardScaler()
        X = self.scaler.fit_transform(features.values)

        # آموزش با چند شروع تصادفی و انتخاب بهترین
        best_model = None
        best_score = -np.inf
        for seed in [42, 7, 123, 2024, 999]:
            try:
                model = hmm.GaussianHMM(
                    n_components=self.n_regimes,
                    covariance_type=self.covariance_type,
                    n_iter=self.n_iter,
                    random_state=seed,
                    tol=1e-4,
                )
                model.fit(X)
                score = model.score(X)
                if score > best_score:
                    best_score = score
                    best_model = model
                logger.debug(f"  seed={seed}: score={score:.2f}")
            except Exception as e:
                logger.warning(f"  seed={seed} failed: {e}")

        if best_model is None:
            raise RuntimeError("All HMM initializations failed")

        self.model = best_model
        logger.info(f"Best HMM score: {best_score:.2f}")

        # ذخیره آمار هر رژیم
        self._compute_regime_stats(features)
        self._build_label_map()
        return self

    def _compute_regime_stats(self, features: pd.DataFrame):
        """محاسبه آمار هر رژیم برای برچسب‌گذاری"""
        X = self.scaler.transform(features.values)
        states = self.model.predict(X)

        stats = {}
        for r in range(self.n_regimes):
            mask = states == r
            if mask.sum() == 0:
                stats[r] = {"mean_return": 0.0, "mean_vol": 0.0, "count": 0, "freq": 0.0}
                continue
            stats[r] = {
                "mean_return": float(features["returns"].values[mask].mean()),
                "mean_vol": float(features["volatility"].values[mask].mean()),
                "count": int(mask.sum()),
                "freq": float(mask.mean()),
            }
        self.regime_stats = stats
        logger.info("Regime stats:")
        for r, s in stats.items():
            logger.info(
                f"  state {r}: return={s['mean_return']:+.5f}, "
                f"vol={s['mean_vol']:.5f}, "
                f"freq={s['freq']*100:.1f}%"
            )

    def _build_label_map(self):
        """برچسب‌گذاری: BEAR (کم‌ترین بازدهی) تا BULL (بیشترین)"""
        items = sorted(self.regime_stats.items(), key=lambda x: x[1]["mean_return"])
        self.label_map = {state: REGIME_LABELS[i] for i, (state, _) in enumerate(items)}
        logger.info(f"Label map: {self.label_map}")

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """پیش‌بینی رژیم برای همه ردیف‌ها"""
        if self.model is None:
            raise RuntimeError("Model not trained")
        features = self._prepare_features(df)
        X = self.scaler.transform(features.values)
        states = self.model.predict(X)
        labels = [self.label_map[s] for s in states]
        return pd.Series(labels, index=features.index, name="regime")

    def predict_current(self, df: pd.DataFrame, window: int = 200) -> str:
        """پیش‌بینی رژیم فعلی بر اساس آخرین N کندل"""
        recent = df.tail(window)
        regimes = self.predict(recent)
        if regimes.empty:
            return "UNKNOWN"
        return regimes.iloc[-1]

    def save(self, path: str = None):
        path = Path(path) if path else MODEL_DIR / "hmm_regime.pkl"
        joblib.dump({
            "model": self.model,
            "scaler": self.scaler,
            "regime_stats": self.regime_stats,
            "label_map": self.label_map,
            "n_regimes": self.n_regimes,
        }, path)
        logger.info(f"Saved HMM model -> {path}")

    @classmethod
    def load(cls, path: str = None):
        path = Path(path) if path else MODEL_DIR / "hmm_regime.pkl"
        data = joblib.load(path)
        detector = cls(n_regimes=data["n_regimes"])
        detector.model = data["model"]
        detector.scaler = data["scaler"]
        detector.regime_stats = data["regime_stats"]
        detector.label_map = data["label_map"]
        logger.info(f"Loaded HMM model from {path}")
        return detector
'''

# ---------- src/regime_detection/regime_validator.py ----------
FILES["src/regime_detection/regime_validator.py"] = '''"""اعتبارسنجی و آمار رژیم‌ها"""
import numpy as np
import pandas as pd


def regime_distribution(regimes: pd.Series) -> pd.DataFrame:
    """توزیع رژیم‌ها"""
    counts = regimes.value_counts()
    pct = (counts / len(regimes) * 100).round(2)
    return pd.DataFrame({"count": counts, "pct": pct})


def regime_transitions(regimes: pd.Series) -> pd.DataFrame:
    """ماتریس انتقال بین رژیم‌ها"""
    labels = sorted(regimes.unique())
    matrix = pd.DataFrame(0, index=labels, columns=labels)

    prev = None
    for r in regimes:
        if prev is not None:
            matrix.loc[prev, r] += 1
        prev = r

    # نرمال‌سازی به احتمال
    row_sums = matrix.sum(axis=1).replace(0, np.nan)
    return matrix.div(row_sums, axis=0).fillna(0).round(3)


def regime_durations(regimes: pd.Series) -> pd.DataFrame:
    """میانگین طول هر رژیم (تعداد کندل متوالی)"""
    if regimes.empty:
        return pd.DataFrame()
    changes = (regimes != regimes.shift()).cumsum()
    durations = regimes.groupby(changes).agg(["first", "count"])
    durations.columns = ["regime", "duration"]
    return durations.groupby("regime")["duration"].agg(["mean", "min", "max", "count"]).round(2)
'''

# ---------- src/regime_detection/__init__.py ----------
FILES["src/regime_detection/__init__.py"] = '''from src.regime_detection.hmm_regime import RegimeDetector
from src.regime_detection.regime_validator import (
    regime_distribution,
    regime_transitions,
    regime_durations,
)

__all__ = [
    "RegimeDetector",
    "regime_distribution",
    "regime_transitions",
    "regime_durations",
]
'''

# ---------- scripts/train_regime.py ----------
FILES["scripts/train_regime.py"] = '''"""آموزش HMM برای تشخیص رژیم بازار"""
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
    logger.info("\\n" + regime_distribution(regimes).to_string())

    logger.info("")
    logger.info("=== Regime Transition Matrix ===")
    logger.info("\\n" + regime_transitions(regimes).to_string())

    logger.info("")
    logger.info("=== Regime Durations (in candles) ===")
    logger.info("\\n" + regime_durations(regimes).to_string())

    # رژیم فعلی
    current = regimes.iloc[-1]
    logger.info("")
    logger.info(f"Current regime: {current}")

    # ذخیره رژیم‌ها در فایل
    regimes.to_frame().to_parquet(Path("data/processed") / f"{symbol}_{interval}_regimes.parquet")
    logger.info(f"Saved regimes -> data/processed/{symbol}_{interval}_regimes.parquet")


if __name__ == "__main__":
    main()
'''

# ---------- tests/test_regime.py ----------
FILES["tests/test_regime.py"] = '''"""تست تشخیص رژیم"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.regime_detection import RegimeDetector
from src.feature_engineering.pipeline import get_processed_path
from src.utils.logger import setup_logger

setup_logger("test")


def main():
    print()
    print("=" * 60)
    print("Testing Regime Detector")
    print("=" * 60)

    detector = RegimeDetector.load()
    print("[OK] Model loaded")

    df = pd.read_parquet(get_processed_path("BTCUSDT", "1h"))
    regimes = detector.predict(df)

    print(f"[OK] Predicted regimes for {len(regimes)} candles")
    print()
    print("Distribution:")
    print(regimes.value_counts().to_string())
    print()
    print(f"Current regime: {regimes.iloc[-1]}")
    print(f"Previous 10 regimes: {list(regimes.tail(10).values)}")


if __name__ == "__main__":
    main()
'''


def main():
    root = Path(".")
    created = 0
    for file_path, content in FILES.items():
        path = root / file_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        created += 1
        print(f"  + {file_path}")
    print()
    print("=" * 50)
    print(f"Created {created} files successfully!")
    print("=" * 50)
    print()
    print("Next steps:")
    print("  1. python scripts/train_regime.py")
    print("  2. python tests/test_regime.py")
    print()


if __name__ == "__main__":
    main()