"""تست تشخیص رژیم"""
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
