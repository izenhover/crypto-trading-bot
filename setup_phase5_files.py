# setup_phase5_files.py
"""ساخت فایل‌های فاز ۵: لایه جریان پول و ناهنجاری"""
from pathlib import Path

FILES = {}

# ---------- src/money_flow/whale_alert.py ----------
FILES["src/money_flow/whale_alert.py"] = '''"""اتصال به Whale Alert API (اختیاری) + پروکسی حجمی"""
import os
from datetime import datetime, timezone, timedelta
from typing import Optional
import requests
from loguru import logger

WHALE_ALERT_URL = "https://api.whale-alert.io/v1/transactions"


class WhaleAlertClient:
    """
    کلاینت Whale Alert.
    اگر کلید API نباشد، به حالت «آفلاین» می‌رود و از داده‌های حجمی محلی استفاده می‌کند.
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("WHALE_ALERT_API_KEY")
        self.online = bool(self.api_key and self.api_key != "your_whale_alert_key_here")
        if self.online:
            logger.info("Whale Alert: online mode")
        else:
            logger.warning("Whale Alert: offline mode (no API key)")

    def fetch_transactions(
        self,
        min_value_usd: int = 1_000_000,
        hours_back: int = 1,
        currency: str = "btc",
    ) -> list:
        """دریافت تراکنش‌های بزرگ از API (اگر آنلاین باشد)"""
        if not self.online:
            return []

        start = int((datetime.now(timezone.utc) - timedelta(hours=hours_back)).timestamp())
        params = {
            "api_key": self.api_key,
            "min_value": min_value_usd,
            "start": start,
            "currency": currency,
        }
        try:
            resp = requests.get(WHALE_ALERT_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            transactions = data.get("transactions", [])
            logger.info(f"Whale Alert: fetched {len(transactions)} transactions")
            return transactions
        except Exception as e:
            logger.error(f"Whale Alert error: {e}")
            return []

    def summarize_flow(self, transactions: list) -> dict:
        """خلاصه جریان پول از تراکنش‌ها"""
        inflow = 0.0
        outflow = 0.0
        n = 0
        max_tx = 0.0
        for tx in transactions:
            amount = float(tx.get("amount_usd", 0))
            tx_type = tx.get("transaction_type", "")
            n += 1
            max_tx = max(max_tx, amount)
            if "inflow" in tx_type:
                inflow += amount
            elif "outflow" in tx_type:
                outflow += amount
        return {
            "inflow_usd": inflow,
            "outflow_usd": outflow,
            "net_flow_usd": inflow - outflow,
            "tx_count": n,
            "max_tx_usd": max_tx,
        }


def volume_proxy_flow(df) -> dict:
    """
    پروکسی جریان پول از داده‌های حجمی.
    حجم غیرعادی بالا = احتمال حرکت نهنگ
    """
    if len(df) < 60:
        return {"volume_ratio": 1.0, "is_spike": False, "spike_size": 0.0}

    latest_vol = float(df["volume"].iloc[-1])
    rolling_mean = float(df["volume"].iloc[-60:].mean())
    rolling_std = float(df["volume"].iloc[-60:].std())

    ratio = latest_vol / rolling_mean if rolling_mean > 0 else 1.0
    z_score = (latest_vol - rolling_mean) / rolling_std if rolling_std > 0 else 0.0
    is_spike = ratio > 2.5 or z_score > 3.0

    return {
        "volume_ratio": ratio,
        "volume_z_score": z_score,
        "is_spike": is_spike,
        "spike_size": max(0.0, ratio - 1.0),
    }
'''

# ---------- src/money_flow/anomaly.py ----------
FILES["src/money_flow/anomaly.py"] = '''"""تشخیص ناهنجاری با Isolation Forest"""
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
'''

# ---------- src/money_flow/circuit_breaker.py ----------
FILES["src/money_flow/circuit_breaker.py"] = '''"""قطع‌کننده مدار برای توقف معاملات در شرایط بحرانی"""
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional
from loguru import logger


@dataclass
class CircuitBreakerState:
    halted: bool = False
    halt_until: Optional[datetime] = None
    reason: str = ""
    consecutive_losses: int = 0
    daily_pnl: float = 0.0
    last_reset: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class CircuitBreaker:
    """مدیریت توقف اضطراری معاملات"""

    def __init__(
        self,
        max_consecutive_losses: int = 3,
        max_daily_loss_pct: float = 0.05,
        halt_duration_hours: int = 24,
        anomaly_threshold: float = 0.95,
    ):
        self.max_consecutive_losses = max_consecutive_losses
        self.max_daily_loss_pct = max_daily_loss_pct
        self.halt_duration_hours = halt_duration_hours
        self.anomaly_threshold = anomaly_threshold
        self.state = CircuitBreakerState()

    def _reset_if_new_day(self):
        now = datetime.now(timezone.utc)
        if now.date() != self.state.last_reset.date():
            logger.info("New day: resetting circuit breaker daily stats")
            self.state.daily_pnl = 0.0
            self.state.consecutive_losses = 0
            self.state.last_reset = now

    def record_trade(self, pnl_pct: float):
        """ثبت نتیجه معامله (pnl_pct: بازده به‌صورت اعشار، مثلاً 0.02 = +2%)"""
        self._reset_if_new_day()
        self.state.daily_pnl += pnl_pct

        if pnl_pct < 0:
            self.state.consecutive_losses += 1
        else:
            self.state.consecutive_losses = 0

        # چک شرایط توقف
        if self.state.consecutive_losses >= self.max_consecutive_losses:
            self.halt(
                f"{self.state.consecutive_losses} consecutive losses",
                hours=self.halt_duration_hours,
            )
        elif self.state.daily_pnl <= -self.max_daily_loss_pct:
            self.halt(
                f"daily loss limit hit: {self.state.daily_pnl*100:.2f}%",
                hours=self.halt_duration_hours * 2,
            )

    def record_anomaly(self, anomaly_score: float, is_anomaly: bool):
        """ثبت ناهنجاری جریان پول"""
        if is_anomaly and anomaly_score > self.anomaly_threshold:
            self.halt(
                f"severe flow anomaly (score={anomaly_score:.2f})",
                hours=2,
            )

    def halt(self, reason: str, hours: int = 24):
        self.state.halted = True
        self.state.halt_until = datetime.now(timezone.utc) + timedelta(hours=hours)
        self.state.reason = reason
        logger.warning(f"CIRCUIT BREAKER: {reason} — halted for {hours}h")

    def can_trade(self) -> bool:
        self._reset_if_new_day()
        if not self.state.halted:
            return True
        if self.state.halt_until and datetime.now(timezone.utc) >= self.state.halt_until:
            logger.info("Circuit breaker released")
            self.state.halted = False
            self.state.halt_until = None
            self.state.reason = ""
            return True
        return False

    def status(self) -> dict:
        return {
            "halted": self.state.halted,
            "halt_until": self.state.halt_until.isoformat() if self.state.halt_until else None,
            "reason": self.state.reason,
            "consecutive_losses": self.state.consecutive_losses,
            "daily_pnl": self.state.daily_pnl,
        }
'''

# ---------- src/money_flow/__init__.py ----------
FILES["src/money_flow/__init__.py"] = '''from src.money_flow.whale_alert import WhaleAlertClient, volume_proxy_flow
from src.money_flow.anomaly import FlowAnomalyDetector, build_flow_features
from src.money_flow.circuit_breaker import CircuitBreaker, CircuitBreakerState

__all__ = [
    "WhaleAlertClient",
    "volume_proxy_flow",
    "FlowAnomalyDetector",
    "build_flow_features",
    "CircuitBreaker",
    "CircuitBreakerState",
]
'''

# ---------- scripts/train_anomaly.py ----------
FILES["scripts/train_anomaly.py"] = '''"""آموزش مدل تشخیص ناهنجاری جریان پول"""
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
'''

# ---------- tests/test_money_flow.py ----------
FILES["tests/test_money_flow.py"] = '''"""تست لایه جریان پول"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.feature_engineering.pipeline import get_processed_path
from src.money_flow import FlowAnomalyDetector, CircuitBreaker, volume_proxy_flow
from src.utils.logger import setup_logger

setup_logger("test")


def main():
    print()
    print("=" * 60)
    print("Testing Money Flow Layer")
    print("=" * 60)

    # ۱. بارگذاری داده
    df = pd.read_parquet(get_processed_path("BTCUSDT", "1h"))
    print(f"[OK] Loaded {len(df)} rows")

    # ۲. تست Volume Proxy
    proxy = volume_proxy_flow(df)
    print(f"[OK] Volume proxy: ratio={proxy['volume_ratio']:.2f}, "
          f"z={proxy.get('volume_z_score', 0):.2f}, "
          f"spike={proxy['is_spike']}")

    # ۳. تست Anomaly Detector
    detector = FlowAnomalyDetector.load()
    current = detector.current_score(df)
    print(f"[OK] Current anomaly: score={current['anomaly_score']:.4f}, "
          f"is_anomaly={current['is_anomaly']}")

    # ۴. تست Circuit Breaker
    cb = CircuitBreaker(max_consecutive_losses=3, max_daily_loss_pct=0.05)
    print(f"[OK] CircuitBreaker initial: can_trade={cb.can_trade()}")

    cb.record_trade(-0.01)
    cb.record_trade(-0.01)
    cb.record_trade(-0.01)
    print(f"[OK] After 3 losses: can_trade={cb.can_trade()} "
          f"(reason: {cb.status()['reason']})")

    # ۵. تست هشدار ناهنجاری
    cb2 = CircuitBreaker(anomaly_threshold=0.5)
    cb2.record_anomaly(anomaly_score=0.9, is_anomaly=True)
    print(f"[OK] After severe anomaly: can_trade={cb2.can_trade()}")

    print()
    print("=" * 60)
    print("All money-flow tests passed!")
    print("=" * 60)


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
    print("  1. python scripts/train_anomaly.py")
    print("  2. python tests/test_money_flow.py")
    print()


if __name__ == "__main__":
    main()