"""تست موتور سیگنال"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from src.feature_engineering.pipeline import get_processed_path
from src.signal_engine import SignalGenerator, SignalType
from src.utils.logger import setup_logger

setup_logger("test")


def main():
    print()
    print("=" * 60)
    print("Testing Signal Generator")
    print("=" * 60)

    df = pd.read_parquet(get_processed_path("BTCUSDT", "1h"))
    generator = SignalGenerator(
        threshold_buy=0.80,
        threshold_sell=0.20,
        min_confirmations=2,
    )

    # تست سناریوهای مختلف
    scenarios = [
        {
            "name": "STRONG BUY",
            "ensemble_score": 0.90,
            "regime": "BULL",
            "anomaly_score": 0.3,
            "is_anomaly": False,
        },
        {
            "name": "MARGINAL (in neutral zone)",
            "ensemble_score": 0.55,
            "regime": "SIDEWAYS_LOW",
            "anomaly_score": 0.3,
            "is_anomaly": False,
        },
        {
            "name": "STRONG SELL",
            "ensemble_score": 0.10,
            "regime": "BEAR",
            "anomaly_score": 0.3,
            "is_anomaly": False,
        },
        {
            "name": "CIRCUIT BREAKER ACTIVE",
            "ensemble_score": 0.95,
            "regime": "BULL",
            "anomaly_score": 0.3,
            "is_anomaly": False,
            "circuit_breaker": True,
        },
        {
            "name": "SEVERE ANOMALY",
            "ensemble_score": 0.90,
            "regime": "BULL",
            "anomaly_score": 0.95,
            "is_anomaly": True,
        },
    ]

    for scenario in scenarios:
        print()
        print("-" * 60)
        print(f"Scenario: {scenario['name']}")
        print("-" * 60)

        signal = generator.generate(
            df=df,
            ensemble_score=scenario["ensemble_score"],
            regime=scenario["regime"],
            anomaly_score=scenario["anomaly_score"],
            is_anomaly=scenario["is_anomaly"],
            circuit_breaker_active=scenario.get("circuit_breaker", False),
        )

        print(f"  Signal: {signal.signal.value}")
        print(f"  Confidence: {signal.confidence:.3f} ({signal.confidence_level.value})")
        print(f"  Confirmations: {signal.confirmations}")
        if signal.stop_loss:
            print(f"  Stop Loss: {signal.stop_loss:.2f}")
            print(f"  Take Profit: {signal.take_profit:.2f}")
        print("  Reasons:")
        for r in signal.reasons:
            print(f"    - {r}")

    print()
    print("=" * 60)
    print("All signal tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
