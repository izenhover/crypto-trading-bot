# scripts/run_demo.py
"""
Generates mock data for portfolio demonstration purposes.
This script is completely SAFE to share. It uses only random numbers
to populate the dashboard, protecting your real strategy and models.
"""
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

STATE_DIR = Path("data/state")
STATE_DIR.mkdir(parents=True, exist_ok=True)

def generate_mock_data():
    # Mock Portfolio
    portfolio = {
        "capital": 10523.45,
        "last_processed_ts": str(datetime.now()),
        "open_positions": [
            {
                "id": 4, "symbol": "BTCUSDT", "direction": "BUY", 
                "entry_price": 64200.0, "quantity": 0.05, 
                "position_value": 3210.0, "stop_loss": 63500.0, 
                "take_profit": 66000.0, "entry_time": "2026-10-19T14:00:00Z", 
                "confidence": 0.88, "regime": "BULL", "status": "OPEN"
            }
        ],
        "circuit_breaker": {"halted": False, "consecutive_losses": 0, "daily_pnl": 0.015},
    }
    (STATE_DIR / "paper_portfolio.json").write_text(json.dumps(portfolio, indent=2), encoding="utf-8")

    # Mock Trades
    trades = [
        {"id": 1, "symbol": "BTCUSDT", "direction": "SELL", "entry_price": 62000, "exit_price": 61500, "pnl_pct": -0.008, "pnl_amount": -40.0, "regime": "BEAR"},
        {"id": 2, "symbol": "BTCUSDT", "direction": "BUY", "entry_price": 63000, "exit_price": 64500, "pnl_pct": 0.023, "pnl_amount": 115.0, "regime": "BULL"},
        {"id": 3, "symbol": "BTCUSDT", "direction": "BUY", "entry_price": 64100, "exit_price": 65000, "pnl_pct": 0.014, "pnl_amount": 70.0, "regime": "BULL"},
    ]
    with open(STATE_DIR / "paper_trades.jsonl", "w", encoding="utf-8") as f:
        for t in trades:
            f.write(json.dumps(t) + "\n")

    # Mock Equity
    eq = []
    current_eq = 10000
    for i in range(50):
        current_eq += random.uniform(-50, 150)
        eq.append({"ts": str(datetime.now() - timedelta(hours=50-i)), "equity": current_eq})
    with open(STATE_DIR / "paper_equity.jsonl", "w", encoding="utf-8") as f:
        for e in eq:
            f.write(json.dumps(e) + "\n")
            
    # Mock Last Signal
    signal = {
        "regime": "BULL", "signal": "BUY", "confidence": 0.88, 
        "confirmations": 3, "ensemble_score": 0.91, "anomaly_score": 0.12
    }
    (STATE_DIR / "last_signal.json").write_text(json.dumps(signal, indent=2), encoding="utf-8")

    print("✅ Mock demo data generated successfully! You can now run the dashboard.")

if __name__ == "__main__":
    generate_mock_data()