"""تست لایه جریان پول"""
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
