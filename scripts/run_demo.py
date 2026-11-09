# scripts/run_demo.py
"""Generates highly realistic, yet profitable, mock data for portfolio demo."""
import json
import random
from datetime import datetime, timedelta
from pathlib import Path

STATE_DIR = Path("data/state")
STATE_DIR.mkdir(parents=True, exist_ok=True)

def generate_mock_data():
    # Mock Portfolio (Capital has grown from $10k to ~$11.5k)
    portfolio = {
        "capital": 11520.00,
        "last_processed_ts": str(datetime.now()),
        "open_positions": [
            {
                "id": 12, "symbol": "BTCUSDT", "direction": "BUY", 
                "entry_price": 64200.0, "quantity": 0.05, 
                "position_value": 3210.0, "stop_loss": 63500.0, 
                "take_profit": 67000.0, "entry_time": "2026-11-10T14:00:00Z", 
                "confidence": 0.88, "regime": "BULL", "status": "OPEN"
            }
        ],
        "circuit_breaker": {"halted": False, "consecutive_losses": 0, "daily_pnl": 0.025},
    }
    (STATE_DIR / "paper_portfolio.json").write_text(json.dumps(portfolio, indent=2), encoding="utf-8")

    # Mock Trades (Realistic: 8 Wins, 4 Losses = 66% Win Rate)
    # Includes trades from BULL, BEAR, and SIDEWAYS regimes
    trades = [
        {"id": 1, "symbol": "BTCUSDT", "direction": "BUY", "entry_price": 60000, "exit_price": 61200, "pnl_pct": 0.020, "pnl_amount": 120.0, "regime": "BULL"},
        {"id": 2, "symbol": "BTCUSDT", "direction": "SELL", "entry_price": 61500, "exit_price": 61900, "pnl_pct": -0.006, "pnl_amount": -60.0, "regime": "BEAR"},
        {"id": 3, "symbol": "BTCUSDT", "direction": "BUY", "entry_price": 62000, "exit_price": 62800, "pnl_pct": 0.012, "pnl_amount": 80.0, "regime": "SIDEWAYS_LOW"},
        {"id": 4, "symbol": "BTCUSDT", "direction": "BUY", "entry_price": 63000, "exit_price": 64500, "pnl_pct": 0.023, "pnl_amount": 150.0, "regime": "BULL"},
        {"id": 5, "symbol": "BTCUSDT", "direction": "SELL", "entry_price": 65000, "exit_price": 65400, "pnl_pct": -0.006, "pnl_amount": -40.0, "regime": "BEAR"},
        {"id": 6, "symbol": "BTCUSDT", "direction": "BUY", "entry_price": 64100, "exit_price": 65000, "pnl_pct": 0.014, "pnl_amount": 90.0, "regime": "BULL"},
        {"id": 7, "symbol": "BTCUSDT", "direction": "BUY", "entry_price": 65500, "exit_price": 67500, "pnl_pct": 0.030, "pnl_amount": 200.0, "regime": "BULL"},
        {"id": 8, "symbol": "BTCUSDT", "direction": "SELL", "entry_price": 67000, "exit_price": 67800, "pnl_pct": -0.011, "pnl_amount": -80.0, "regime": "BEAR"},
        {"id": 9, "symbol": "BTCUSDT", "direction": "BUY", "entry_price": 66000, "exit_price": 67100, "pnl_pct": 0.016, "pnl_amount": 110.0, "regime": "BULL"},
        {"id": 10, "symbol": "BTCUSDT", "direction": "BUY", "entry_price": 66200, "exit_price": 66900, "pnl_pct": 0.010, "pnl_amount": 70.0, "regime": "SIDEWAYS_LOW"},
        {"id": 11, "symbol": "BTCUSDT", "direction": "BUY", "entry_price": 65000, "exit_price": 66300, "pnl_pct": 0.020, "pnl_amount": 130.0, "regime": "BULL"},
    ]
    with open(STATE_DIR / "paper_trades.jsonl", "w", encoding="utf-8") as f:
        for t in trades:
            f.write(json.dumps(t) + "\n")

    # Mock Equity (Realistic growth with some dips)
    eq = []
    current_eq = 10000
    for i in range(60):
        # 70% chance of going up, 30% chance of a small dip
        if random.random() < 0.70:
            current_eq += random.uniform(20, 100)
        else:
            current_eq -= random.uniform(10, 50)
        eq.append({"ts": str(datetime.now() - timedelta(hours=60-i)), "equity": current_eq})
    with open(STATE_DIR / "paper_equity.jsonl", "w", encoding="utf-8") as f:
        for e in eq:
            f.write(json.dumps(e) + "\n")
            
    # Mock Last Signal
    signal = {
        "regime": "BULL", "signal": "BUY", "confidence": 0.88, 
        "confirmations": 3, "ensemble_score": 0.85, "anomaly_score": 0.12
    }
    (STATE_DIR / "last_signal.json").write_text(json.dumps(signal, indent=2), encoding="utf-8")

    print("✅ Realistic demo data generated! (66% Win Rate, mixed regimes)")

if __name__ == "__main__":
    generate_mock_data()