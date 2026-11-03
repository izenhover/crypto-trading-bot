"""داشبورد Streamlit برای نمایش وضعیت ربات"""
import json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
import streamlit as st

STATE_DIR = Path("data/state")
TRADES_FILE = STATE_DIR / "paper_trades.jsonl"
EQUITY_FILE = STATE_DIR / "paper_equity.jsonl"
PORTFOLIO_FILE = STATE_DIR / "paper_portfolio.json"
REPORTS_DIR = Path("data/reports")


st.set_page_config(page_title="Crypto Bot Dashboard", layout="wide")
st.title("🤖 Crypto Prediction Bot — Dashboard")
st.caption(f"Last refresh: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")


# ============ ۱. وضعیت حساب ============
st.header("💼 Account Status")
col1, col2, col3, col4 = st.columns(4)

if PORTFOLIO_FILE.exists():
    try:
        data = json.loads(PORTFOLIO_FILE.read_text(encoding="utf-8"))
        capital = data.get("capital", 0)
        open_positions = data.get("open_positions", [])
        last_ts = data.get("last_processed_ts", "—")

        col1.metric("Capital (USDT)", f"{capital:,.2f}")
        col2.metric("Open Positions", len(open_positions))
        col3.metric("Last Candle", str(last_ts)[:19] if last_ts else "—")
        col4.metric("Status", "🟢 Active" if open_positions else "⚪ Idle")
    except Exception as e:
        st.error(f"Error reading portfolio: {e}")
else:
    st.warning("No state file found. Run `python scripts/run_live.py` first.")


# ============ ۲. منحنی سرمایه ============
st.header("📈 Equity Curve")
if EQUITY_FILE.exists():
    try:
        lines = EQUITY_FILE.read_text(encoding="utf-8").strip().split("\n")
        records = [json.loads(l) for l in lines if l]
        if records:
            df = pd.DataFrame(records)
            df["ts"] = pd.to_datetime(df["ts"])
            df = df.set_index("ts").sort_index()
            st.line_chart(df["equity"])
            st.caption(f"{len(df)} data points | "
                       f"From {df.index.min()} to {df.index.max()}")
        else:
            st.info("No equity data yet.")
    except Exception as e:
        st.error(f"Error reading equity: {e}")
else:
    st.info("No equity log yet.")


# ============ ۳. معاملات ============
st.header("📊 Trades Log")
if TRADES_FILE.exists():
    try:
        lines = TRADES_FILE.read_text(encoding="utf-8").strip().split("\n")
        trades = [json.loads(l) for l in lines if l]
        if trades:
            tdf = pd.DataFrame(trades)

            # آمار خلاصه
            if "pnl_pct" in tdf.columns and "pnl_amount" in tdf.columns:
                wins = (tdf["pnl_pct"] > 0).sum()
                losses = (tdf["pnl_pct"] <= 0).sum()
                total_pnl = tdf["pnl_amount"].sum()
                win_rate = wins / len(tdf) * 100 if len(tdf) > 0 else 0

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Total Trades", len(tdf))
                c2.metric("Wins / Losses", f"{wins} / {losses}")
                c3.metric("Win Rate", f"{win_rate:.1f}%")
                c4.metric("Total PnL", f"{total_pnl:+.2f} USDT")

            # جدول
            st.subheader("Recent Trades")
            cols_to_show = [c for c in ["id", "symbol", "direction", "entry_price",
                                        "exit_price", "pnl_pct", "pnl_amount",
                                        "entry_time", "exit_time", "regime"]
                            if c in tdf.columns]
            st.dataframe(tdf[cols_to_show].tail(50), use_container_width=True)
        else:
            st.info("No trades yet.")
    except Exception as e:
        st.error(f"Error reading trades: {e}")
else:
    st.info("No trades log yet.")


# ============ ۴. گزارش بک‌تست ============
st.header("🧪 Last Backtest Results")
metrics_file = REPORTS_DIR / "BTCUSDT_1h_backtest_clean_metrics.parquet"
if metrics_file.exists():
    try:
        m = pd.read_parquet(metrics_file).iloc[0].to_dict()
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Return", f"{m.get('total_return_pct', 0):.2f}%")
        c2.metric("Sharpe", f"{m.get('sharpe', 0):.3f}")
        c3.metric("Max DD", f"{m.get('max_drawdown_pct', 0):.2f}%")
        c4.metric("Trades", int(m.get("total_trades", 0)))

        with st.expander("All metrics"):
            st.json(m)
    except Exception as e:
        st.error(f"Error reading metrics: {e}")
else:
    st.info("No backtest report yet. Run `python scripts/run_backtest.py`.")


# ============ ۵. راهنما ============
st.divider()
st.caption(
    "**نکته**: این داشبورد به‌صورت زنده از فایل‌های `data/state/` می‌خواند. "
    "برای به‌روزرسانی، صفحه را refresh کنید (F5)."
)
