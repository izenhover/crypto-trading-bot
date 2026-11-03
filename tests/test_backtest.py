"""تست ماژول‌های بک‌تست"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

from backtest import (
    compute_all_metrics, walk_forward_splits,
    SimConfig, Simulator,
)
from src.utils.logger import setup_logger

setup_logger("test")


def main():
    print()
    print("=" * 60)
    print("Testing Backtest Modules")
    print("=" * 60)

    # ۱. تست Walk-Forward
    windows = list(walk_forward_splits(n_samples=1000, train_size=500, test_size=100))
    print(f"[OK] Walk-Forward: {len(windows)} windows")

    # ۲. تست Metrics
    equity = pd.Series([10000, 10100, 10050, 10200, 10300, 10250, 10500])
    trades = pd.DataFrame({
        "pnl_amount": [100, -50, 150, 100, -50, 250],
        "regime": ["BULL", "BEAR", "BULL", "BULL", "BEAR", "BULL"],
    })
    metrics = compute_all_metrics(equity, trades)
    print(f"[OK] Metrics: {len(metrics)} values")
    for k, v in list(metrics.items())[:5]:
        print(f"     {k}: {v}")

    # ۳. تست Simulator
    cfg = SimConfig(initial_capital=10000)
    sim = Simulator(cfg)
    print(f"[OK] Simulator: capital={sim.portfolio.capital}")

    print()
    print("=" * 60)
    print("All backtest tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
