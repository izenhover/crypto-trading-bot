# 🤖 Crypto Algorithmic Trading Bot (Architecture Showcase)
A sophisticated, multi-layered algorithmic trading system designed for the cryptocurrency market. This repository serves as a **portfolio showcase** of the architecture, machine learning pipeline, risk management engine, and interactive dashboard.

## ⚠️ Disclaimer & Intellectual Property
**This is a proprietary system.** The trained ML models (weights), specific feature engineering logic, and strategy thresholds are **not included** in this repository to protect the intellectual property. 

To demonstrate the UI, a `run_demo.py` script is included. It generates **mock data**, allowing you to run the dashboard and see the interface without revealing the core predictive engine.

## 🧠 Architecture Highlights
- **Multi-Factor Data Pipeline**: Integrates OHLCV data, technical indicators, and macroeconomic data (via FRED API).
- **Regime Detection (HMM)**: Uses Hidden Markov Models to detect Bull, Bear, and Sideways market regimes, adapting the strategy dynamically.
- **Ensemble ML Engine**: Combines XGBoost, LightGBM, Random Forest, and LSTM (PyTorch) for high-confidence predictions.
- **Advanced Risk Management**: Implements Fractional Kelly Criterion for position sizing, dynamic ATR-based Stop-Losses, and a custom Circuit Breaker system that halts trading after consecutive losses.
- **Real-time Dashboard**: A custom cyberpunk-themed Streamlit dashboard for live monitoring.

## 🛠️ Tech Stack
- **Languages**: Python 3.12
- **Machine Learning**: XGBoost, LightGBM, Scikit-learn, PyTorch
- **Data & Analysis**: Pandas, NumPy, FRED API
- **UI/UX**: Streamlit, Plotly
- **Infrastructure**: Docker

## 🚀 How to Run the Demo

1. **Clone the repository:**
   ```bash
   git clone https://github.com/izenhover/crypto-trading-bot.git
   cd crypto-trading-bot
<img width="1440" height="900" alt="Screenshot (210)" src="https://github.com/user-attachments/assets/5c531b4f-e146-4df2-8755-982b210ad8f7" />
