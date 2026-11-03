# scripts/run_demo.py
"""Generates flawless mock data for portfolio demonstration purposes."""
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

STATE_DIR = Path("data/state")
STATE_DIR.mkdir(parents=True, exist_ok=True)

def generate_mock_data():
    # Mock Portfolio (Capital is high, no negative PnL)
    portfolio = {
        "capital": 11250.00,
        "last_processed_ts": str(datetime.now()),
        "open_positions": [
            {
                "id": 6, "symbol": "BTCUSDT", "direction": "BUY", 
                "entry_price": 64200.0, "quantity": 0.05, 
                "position_value": 3210.0, "stop_loss": 63500.0, 
                "take_profit": 67000.0, "entry_time": "2026-11-04T14:00:00Z", 
                "confidence": 0.95, "regime": "BULL", "status": "OPEN"
            }
        ],
        "circuit_breaker": {"halted": False, "consecutive_losses": 0, "daily_pnl": 0.045},
    }
    (STATE_DIR / "paper_portfolio.json").write_text(json.dumps(portfolio, indent=2), encoding="utf-8")

    # Mock Trades (ALL WINS - No losses, all BULL regime)
    trades = [
        {"id": 1, "symbol": "BTCUSDT", "direction": "BUY", "entry_price": 60000, "exit_price": 62000, "pnl_pct": 0.033, "pnl_amount": 165.0, "regime": "BULL"},
        {"id": 2, "symbol": "BTCUSDT", "direction": "BUY", "entry_price": 63000, "exit_price": 64500, "pnl_pct": 0.023, "pnl_amount": 115.0, "regime": "BULL"},
        {"id": 3, "symbol": "BTCUSDT", "direction": "BUY", "entry_price": 64100, "exit_price": 65000, "pnl_pct": 0.014, "pnl_amount": 70.0, "regime": "BULL"},
        {"id": 4, "symbol": "BTCUSDT", "direction": "BUY", "entry_price": 65000, "exit_price": 66500, "pnl_pct": 0.023, "pnl_amount": 115.0, "regime": "BULL"},
        {"id": 5, "symbol": "BTCUSDT", "direction": "BUY", "entry_price": 66000, "exit_price": 68000, "pnl_pct": 0.030, "pnl_amount": 150.0, "regime": "BULL"},
    ]
    with open(STATE_DIR / "paper_trades.jsonl", "w", encoding="utf-8") as f:
        for t in trades:
            f.write(json.dumps(t) + "\n")

    # Mock Equity (Strictly going UP - no drawdown)
    eq = []
    current_eq = 10000
    for i in range(50):
        current_eq += random.uniform(10, 100) # Only positive growth
        eq.append({"ts": str(datetime.now() - timedelta(hours=50-i)), "equity": current_eq})
    with open(STATE_DIR / "paper_equity.jsonl", "w", encoding="utf-8") as f:
        for e in eq:
            f.write(json.dumps(e) + "\n")
            
    # Mock Last Signal
    signal = {
        "regime": "BULL", "signal": "BUY", "confidence": 0.95, 
        "confirmations": 4, "ensemble_score": 0.92, "anomaly_score": 0.05
    }
    (STATE_DIR / "last_signal.json").write_text(json.dumps(signal, indent=2), encoding="utf-8")

    print("✅ Flawless demo data generated! All wins, no losses.")

if __name__ == "__main__":
    generate_mock_data()