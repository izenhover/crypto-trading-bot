# monitoring/dashboard_pro.py
"""داشبورد حرفه‌ای Operations Center برای ربات کریپتو"""
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# ============================================================
# تنظیمات صفحه
# ============================================================
st.set_page_config(
    page_title="Crypto Bot | Ops Center",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# CSS — سبک Water Plant Dashboard
# ============================================================
st.markdown("""
<style>
    .stApp {
        background: radial-gradient(ellipse at 50% 0%, #071c33 0%, #030b16 45%, #000205 100%);
        color: #c8e6ff;
        font-family: 'Segoe UI', 'Tahoma', sans-serif;
    }
    #MainMenu, footer, header {visibility: hidden;}
    
    /* کارت اصلی */
    .panel {
        background: linear-gradient(135deg, rgba(0,60,110,0.30), rgba(0,20,45,0.55));
        border: 1px solid rgba(0,180,255,0.28);
        border-radius: 10px;
        padding: 12px 14px;
        box-shadow: 0 0 14px rgba(0,150,255,0.12), inset 0 0 12px rgba(0,150,255,0.04);
        margin-bottom: 10px;
        position: relative;
    }
    .panel::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, #00d4ff, transparent);
    }
    
    /* هدر */
    .main-header {
        text-align: center;
        padding: 8px 0 6px 0;
        border-bottom: 1px solid rgba(0,180,255,0.25);
        margin-bottom: 12px;
    }
    .main-header h1 {
        color: #00d4ff;
        font-size: 19px;
        letter-spacing: 6px;
        text-transform: uppercase;
        margin: 0;
        text-shadow: 0 0 22px rgba(0,200,255,0.8);
    }
    .main-header .subtitle {
        color: #5f8ba8;
        font-size: 10px;
        letter-spacing: 3px;
        margin-top: 4px;
    }
    
    /* KPI Cards */
    .kpi {
        background: linear-gradient(135deg, rgba(0,80,140,0.32), rgba(0,25,55,0.55));
        border: 1px solid rgba(0,180,255,0.32);
        border-radius: 8px;
        padding: 10px 12px;
        box-shadow: 0 0 14px rgba(0,150,255,0.12);
        text-align: center;
    }
    .kpi-label {
        color: #6ea8c9;
        font-size: 9.5px;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        margin-bottom: 4px;
    }
    .kpi-value {
        color: #e6f6ff;
        font-size: 20px;
        font-weight: 700;
        text-shadow: 0 0 12px rgba(0,200,255,0.55);
        font-family: 'Consolas', monospace;
    }
    .kpi-sub {
        color: #7fa8c8;
        font-size: 10px;
        margin-top: 3px;
    }
    .kpi-sub.pos { color: #00ff9d; }
    .kpi-sub.neg { color: #ff4d6d; }
    
    /* بخش‌بندی */
    .section {
        color: #00d4ff;
        font-size: 11px;
        letter-spacing: 2.5px;
        text-transform: uppercase;
        margin: 8px 0 6px 0;
        padding-bottom: 3px;
        border-bottom: 1px solid rgba(0,180,255,0.2);
        text-shadow: 0 0 8px rgba(0,200,255,0.5);
    }
    
    /* کارت سیگنال */
    .signal-box {
        text-align: center;
        padding: 18px 12px;
        border-radius: 12px;
    }
    .signal-box.buy {
        background: radial-gradient(circle, rgba(0,255,157,0.15), transparent 70%);
        border: 2px solid rgba(0,255,157,0.55);
        box-shadow: 0 0 26px rgba(0,255,157,0.28);
    }
    .signal-box.sell {
        background: radial-gradient(circle, rgba(255,77,109,0.15), transparent 70%);
        border: 2px solid rgba(255,77,109,0.55);
        box-shadow: 0 0 26px rgba(255,77,109,0.28);
    }
    .signal-box.hold {
        background: radial-gradient(circle, rgba(120,150,180,0.10), transparent 70%);
        border: 2px solid rgba(120,150,180,0.4);
    }
    .signal-box .txt {
        font-size: 30px;
        font-weight: 800;
        letter-spacing: 6px;
    }
    .txt.buy { color: #00ff9d; text-shadow: 0 0 22px #00ff9d; }
    .txt.sell { color: #ff4d6d; text-shadow: 0 0 22px #ff4d6d; }
    .txt.hold { color: #90a8c0; }
    
    /* کارت پوزیشن */
    .pos-card {
        background: linear-gradient(135deg, rgba(0,120,80,0.22), rgba(0,25,25,0.45));
        border-left: 3px solid #00ff9d;
        border-radius: 6px;
        padding: 8px 10px;
        margin: 5px 0;
        font-size: 12px;
    }
    .pos-card.sell {
        background: linear-gradient(135deg, rgba(140,20,50,0.22), rgba(30,0,10,0.45));
        border-left: 3px solid #ff4d6d;
    }
    
    /* Badge */
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 8px;
        font-size: 10px;
        font-weight: 600;
        letter-spacing: 1px;
    }
    .badge.green { background: rgba(0,255,157,0.18); color: #00ff9d; border: 1px solid #00ff9d; }
    .badge.red { background: rgba(255,77,109,0.18); color: #ff4d6d; border: 1px solid #ff4d6d; }
    .badge.yellow { background: rgba(255,200,0,0.18); color: #ffc800; border: 1px solid #ffc800; }
    .badge.blue { background: rgba(0,180,255,0.18); color: #00d4ff; border: 1px solid #00d4ff; }
    .badge.gray { background: rgba(150,180,200,0.12); color: #90a8c0; border: 1px solid #90a8c0; }
    
    /* Live dot */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.3; }
        100% { opacity: 1; }
    }
    .dot {
        display: inline-block;
        width: 8px; height: 8px;
        border-radius: 50%;
        background: #00ff9d;
        box-shadow: 0 0 10px #00ff9d;
        animation: pulse 1.4s infinite;
        margin-right: 6px;
        vertical-align: middle;
    }
    .dot.red { background: #ff4d6d; box-shadow: 0 0 10px #ff4d6d; }
    .dot.yellow { background: #ffc800; box-shadow: 0 0 10px #ffc800; }
    
    /* جدول */
    .stDataFrame { background-color: transparent !important; }
    
    /* Grid مرکزی */
    .center-diagram {
        text-align: center;
        padding: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# مسیرها
# ============================================================
STATE_DIR = Path("data/state")
TRADES_FILE = STATE_DIR / "paper_trades.jsonl"
EQUITY_FILE = STATE_DIR / "paper_equity.jsonl"
PORTFOLIO_FILE = STATE_DIR / "paper_portfolio.json"
REPORTS_DIR = Path("data/reports")
PROCESSED_DIR = Path("data/processed")


# ============================================================
# Loader
# ============================================================
def load_json_safe(path):
    if not path.exists(): return {}
    try: return json.loads(path.read_text(encoding="utf-8"))
    except Exception: return {}

def load_jsonl_safe(path):
    if not path.exists(): return []
    try:
        return [json.loads(l) for l in path.read_text(encoding="utf-8").strip().split("\n") if l]
    except Exception: return []

def load_parquet_safe(path):
    if not path.exists(): return None
    try: return pd.read_parquet(path)
    except Exception: return None


# ============================================================
# بارگذاری داده‌ها
# ============================================================
portfolio = load_json_safe(PORTFOLIO_FILE)
trades = load_jsonl_safe(TRADES_FILE)
equity = load_jsonl_safe(EQUITY_FILE)
features_df = load_parquet_safe(PROCESSED_DIR / "BTCUSDT_1h_features.parquet")
metrics_df = load_parquet_safe(REPORTS_DIR / "BTCUSDT_1h_backtest_clean_metrics.parquet")

capital = portfolio.get("capital", 10000)
open_positions = portfolio.get("open_positions", [])
last_ts = portfolio.get("last_processed_ts", "—")
cb_state = portfolio.get("circuit_breaker", {})

trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
equity_df = pd.DataFrame(equity) if equity else pd.DataFrame()


# ============================================================
# Helper
# ============================================================
def panel_start():
    st.markdown('<div class="panel">', unsafe_allow_html=True)

def panel_end():
    st.markdown('</div>', unsafe_allow_html=True)

def section(t):
    st.markdown(f'<div class="section">◆ {t}</div>', unsafe_allow_html=True)

def kpi(label, value, sub="", sub_kind=""):
    sub_cls = f"kpi-sub {sub_kind}" if sub_kind else "kpi-sub"
    sub_html = f'<div class="{sub_cls}">{sub}</div>' if sub else ''
    st.markdown(f"""
        <div class="kpi">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
            {sub_html}
        </div>
    """, unsafe_allow_html=True)


# ============================================================
# هدر اصلی
# ============================================================
live = "LIVE" if open_positions else "STANDBY"
dot_cls = "" if live == "LIVE" else "yellow"
st.markdown(f"""
    <div class="main-header">
        <h1>CRYPTO BOT &nbsp;·&nbsp; OPERATIONS CENTER</h1>
        <div class="subtitle">
            <span class="dot {dot_cls}"></span>{live}
            &nbsp;·&nbsp; BTC/USDT · 1H
            &nbsp;·&nbsp; {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC
        </div>
    </div>
""", unsafe_allow_html=True)


# ============================================================
# [۱] KPI Row — ۸ کارت بالا
# ============================================================
section("SYSTEM OVERVIEW")

# محاسبات
total_return_pct = 0.0
if not equity_df.empty and len(equity_df) > 1:
    total_return_pct = (equity_df["equity"].iloc[-1] / equity_df["equity"].iloc[0] - 1) * 100

wins = losses = 0
total_pnl = win_rate = 0.0
profit_factor = 0.0
if not trades_df.empty and "pnl_pct" in trades_df.columns:
    wins = int((trades_df["pnl_pct"] > 0).sum())
    losses = int((trades_df["pnl_pct"] <= 0).sum())
    if "pnl_amount" in trades_df.columns:
        total_pnl = float(trades_df["pnl_amount"].sum())
        gains = trades_df.loc[trades_df["pnl_amount"] > 0, "pnl_amount"].sum()
        loss_abs = abs(trades_df.loc[trades_df["pnl_amount"] < 0, "pnl_amount"].sum())
        profit_factor = gains / loss_abs if loss_abs > 0 else 0.0
    win_rate = wins / len(trades_df) * 100 if len(trades_df) else 0

c1, c2, c3, c4, c5, c6, c7, c8 = st.columns(8)
with c1: kpi("CAPITAL", f"${capital:,.0f}", f"{total_return_pct:+.2f}%",
             "pos" if total_return_pct > 0 else "neg" if total_return_pct < 0 else "")
with c2: kpi("TOTAL PnL", f"{total_pnl:+.1f}", "", "pos" if total_pnl > 0 else "neg")
with c3: kpi("WIN RATE", f"{win_rate:.0f}%", f"{wins}W / {losses}L",
             "pos" if win_rate >= 50 else "neg")
with c4: kpi("PROFIT FACTOR", f"{profit_factor:.2f}", "", "pos" if profit_factor >= 1 else "neg")
with c5: kpi("TRADES", f"{len(trades_df)}", f"open: {len(open_positions)}", "pos" if open_positions else "")
with c6:
    cb_halted = cb_state.get("halted", False)
    kpi("CIRCUIT BRK", "HALTED" if cb_halted else "ARMED",
        f"{cb_state.get('consecutive_losses', 0)} consec", "neg" if cb_halted else "pos")
with c7:
    regime = "—"
    if features_df is not None and not features_df.empty:
        try:
            from src.regime_detection import RegimeDetector
            det = RegimeDetector.load()
            regime = det.predict_current(features_df, window=200)
        except Exception:
            regime = "—"
    kpi("REGIME", regime, "", "pos" if regime in ("BULL", "SIDEWAYS_LOW") else "neg")
with c8:
    last_str = str(last_ts)[-8:-3] if last_ts and last_ts != "—" else "—"
    kpi("UPDATED", last_str, str(last_ts)[:10] if last_ts != "—" else "", "")

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# [۲] ردیف مرکزی: نمودار اصلی + سیگنال + شاخص‌ها
# ============================================================
col_left, col_center, col_right = st.columns([1, 2.5, 1])

# -------- چپ: شاخص‌های تکنیکال --------
with col_left:
    section("TECHNICAL")
    panel_start()
    if features_df is not None and not features_df.empty:
        last = features_df.iloc[-1]

        # RSI Gauge
        rsi = float(last.get("rsi_14", 50))
        fig_rsi = go.Figure(go.Indicator(
            mode="gauge+number",
            value=rsi,
            number={"font": {"color": "#00d4ff", "size": 20}, "suffix": ""},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#5f8ba8"},
                "bar": {"color": "#00d4ff"},
                "bgcolor": "rgba(0,0,0,0)",
                "steps": [
                    {"range": [0, 30], "color": "rgba(0,255,157,0.15)"},
                    {"range": [30, 70], "color": "rgba(0,180,255,0.10)"},
                    {"range": [70, 100], "color": "rgba(255,77,109,0.15)"},
                ],
                "threshold": {"value": 70, "line": {"color": "#ff4d6d", "width": 2}},
            },
        ))
        fig_rsi.update_layout(height=140, margin=dict(l=10, r=10, t=10, b=5),
                              paper_bgcolor="rgba(0,0,0,0)", font={"color": "#c8e6ff"},
                              title={"text": "RSI(14)", "font": {"size": 11, "color": "#6ea8c9"}})
        st.plotly_chart(fig_rsi, use_container_width=True, config={"displayModeBar": False})

        # MACD
        macd = float(last.get("macd", 0))
        macd_sig = float(last.get("macd_signal", 0))
        macd_hist = float(last.get("macd_hist", 0))
        macd_color = "#00ff9d" if macd_hist > 0 else "#ff4d6d"
        st.markdown(f"""
            <div style="text-align:center; padding:6px 0;">
                <div class="kpi-label">MACD</div>
                <div class="kpi-value" style="color:{macd_color}; font-size:16px;">
                    {macd_hist:+.4f}
                </div>
                <div class="kpi-sub">MACD: {macd:.4f} | SIG: {macd_sig:.4f}</div>
            </div>
        """, unsafe_allow_html=True)

        # Bollinger Position
        bb_pos = float(last.get("bb_position", 0.5)) * 100
        st.markdown(f"""
            <div style="text-align:center; padding:6px 0;">
                <div class="kpi-label">BOLLINGER POS</div>
                <div class="kpi-value" style="font-size:16px;">{bb_pos:.1f}%</div>
                <div class="kpi-sub">{'بالای میانگین' if bb_pos > 50 else 'زیر میانگین'}</div>
            </div>
        """, unsafe_allow_html=True)

        # ATR
        atr_pct = float(last.get("atr_14_pct", 0))
        st.markdown(f"""
            <div style="text-align:center; padding:6px 0;">
                <div class="kpi-label">VOLATILITY (ATR%)</div>
                <div class="kpi-value" style="font-size:16px;">{atr_pct:.3f}%</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.info("ویژگی‌ها بارگذاری نشده")
    panel_end()


# -------- مرکز: نمودار اصلی قیمت + پیش‌بینی --------
with col_center:
    section("MARKET VIEW & PRICE ACTION")
    panel_start()

    if features_df is not None and not features_df.empty:
        recent = features_df.tail(200).copy()

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            row_heights=[0.75, 0.25],
            subplot_titles=("", ""),
        )

        # کندل استیک
        fig.add_trace(go.Candlestick(
            x=recent.index,
            open=recent["open"], high=recent["high"],
            low=recent["low"], close=recent["close"],
            increasing=dict(line=dict(color="#00ff9d"), fillcolor="rgba(0,255,157,0.5)"),
            decreasing=dict(line=dict(color="#ff4d6d"), fillcolor="rgba(255,77,109,0.5)"),
            name="Price",
        ), row=1, col=1)

        # MA20 (bb_mid)
        fig.add_trace(go.Scatter(
            x=recent.index, y=recent["bb_mid"],
            line=dict(color="#ffc800", width=1.5),
            name="MA20",
        ), row=1, col=1)

        # باندهای بولینگر
        fig.add_trace(go.Scatter(
            x=recent.index, y=recent["bb_upper"],
            line=dict(color="rgba(0,180,255,0.4)", width=1, dash="dot"),
            name="BB Upper", showlegend=False,
        ), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=recent.index, y=recent["bb_lower"],
            line=dict(color="rgba(0,180,255,0.4)", width=1, dash="dot"),
            name="BB Lower", showlegend=False,
            fill="tonexty", fillcolor="rgba(0,120,200,0.06)",
        ), row=1, col=1)

        # Volume
        colors = ["rgba(0,255,157,0.6)" if c >= o else "rgba(255,77,109,0.6)"
                  for c, o in zip(recent["close"], recent["open"])]
        fig.add_trace(go.Bar(
            x=recent.index, y=recent["volume"],
            marker=dict(color=colors),
            name="Volume", showlegend=False,
        ), row=2, col=1)

        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,15,30,0.35)",
            height=420,
            margin=dict(l=10, r=10, t=10, b=10),
            showlegend=True,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02,
                font=dict(color="#c8e6ff", size=10),
                bgcolor="rgba(0,0,0,0)",
            ),
            xaxis_rangeslider_visible=False,
            xaxis=dict(gridcolor="rgba(0,120,200,0.12)", showline=False),
            yaxis=dict(gridcolor="rgba(0,120,200,0.12)", showline=False, title="Price"),
            xaxis2=dict(gridcolor="rgba(0,120,200,0.12)"),
            yaxis2=dict(gridcolor="rgba(0,120,200,0.12)", title="Vol"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.info("داده‌ای برای نمایش نیست")
    panel_end()


# -------- راست: سیگنال + رژیم + جریان پول --------
with col_right:
    section("SIGNAL STATUS")
    panel_start()

    # آخرین سیگنال
    last_sig = "HOLD"
    last_conf = 0.0
    last_regime = "—"
    if not trades_df.empty:
        if "direction" in trades_df.columns:
            last_sig = str(trades_df["direction"].iloc[-1]).upper()
        if "confidence" in trades_df.columns:
            last_conf = float(trades_df["confidence"].iloc[-1])
        if "regime" in trades_df.columns:
            last_regime = str(trades_df["regime"].iloc[-1])

    cls = "buy" if last_sig == "BUY" else "sell" if last_sig == "SELL" else "hold"
    st.markdown(f"""
        <div class="signal-box {cls}">
            <div class="txt {cls}">{last_sig}</div>
            <div style="color:#7fa8c8; font-size:10px; letter-spacing:2px; margin-top:6px;">
                CONFIDENCE: {last_conf*100:.1f}%
            </div>
        </div>
    """, unsafe_allow_html=True)

    panel_end()

    section("MARKET REGIME")
    panel_start()
    regime_badge = "green" if regime in ("BULL", "SIDEWAYS_LOW") else "red" if regime == "BEAR" else "yellow"
    st.markdown(f"""
        <div style="text-align:center; padding:8px 0;">
            <span class="badge {regime_badge}" style="font-size:14px; padding:6px 16px;">
                {regime}
            </span>
        </div>
    """, unsafe_allow_html=True)
    panel_end()

    section("RISK STATE")
    panel_start()
    cb_halted = cb_state.get("halted", False)
    st.markdown(f"""
        <div style="text-align:center; padding:4px 0;">
            <span class="badge {'red' if cb_halted else 'green'}">
                {'⛔ HALTED' if cb_halted else '✓ ARMED'}
            </span>
        </div>
        <div class="kpi-sub" style="text-align:center; margin-top:6px;">
            Consecutive losses: <b>{cb_state.get('consecutive_losses', 0)}</b><br>
            Daily PnL: <b style="color:{'#00ff9d' if cb_state.get('daily_pnl', 0) >= 0 else '#ff4d6d'};">
                {cb_state.get('daily_pnl', 0)*100:+.2f}%
            </b>
        </div>
    """, unsafe_allow_html=True)
    panel_end()

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# [۳] ردیف میانی: Equity Curve + Open Positions + Regime Performance
# ============================================================
col_eq, col_pos, col_reg = st.columns([1.5, 1, 1])

# -------- Equity Curve --------
with col_eq:
    section("EQUITY CURVE")
    panel_start()
    if not equity_df.empty and len(equity_df) > 1:
        eq = equity_df.copy()
        eq["ts"] = pd.to_datetime(eq["ts"])
        eq = eq.set_index("ts").sort_index()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=eq.index, y=eq["equity"],
            mode="lines", line=dict(color="#00d4ff", width=2),
            fill="tozeroy", fillcolor="rgba(0,180,255,0.12)",
            name="Equity",
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,15,30,0.35)",
            height=240, margin=dict(l=10, r=10, t=10, b=10),
            showlegend=False,
            xaxis=dict(gridcolor="rgba(0,120,200,0.12)"),
            yaxis=dict(gridcolor="rgba(0,120,200,0.12)"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        # نمایشی
        now = pd.Timestamp.utcnow()
        demo = pd.DataFrame({
            "ts": pd.date_range(end=now, periods=100, freq="h"),
            "equity": 10000 + np.cumsum(np.random.randn(100) * 15),
        }).set_index("ts")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=demo.index, y=demo["equity"],
            mode="lines", line=dict(color="#00d4ff", width=2),
            fill="tozeroy", fillcolor="rgba(0,180,255,0.12)",
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,15,30,0.35)",
            height=240, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
            xaxis=dict(gridcolor="rgba(0,120,200,0.12)"),
            yaxis=dict(gridcolor="rgba(0,120,200,0.12)"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
        st.caption("⚠️ داده نمایشی")
    panel_end()


# -------- Open Positions --------
with col_pos:
    section(f"OPEN POSITIONS ({len(open_positions)})")
    panel_start()
    if open_positions:
        for p in open_positions:
            d = str(p.get("direction", "?")).upper()
            cls = "sell" if d == "SELL" else ""
            st.markdown(f"""
                <div class="pos-card {cls}">
                    <b>#{p.get('id','?')} · {d}</b><br>
                    Entry: <code>{p.get('entry_price',0):.2f}</code><br>
                    SL: <code>{p.get('stop_loss',0):.2f}</code> · TP: <code>{p.get('take_profit',0):.2f}</code>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown('<div style="text-align:center; color:#5f8ba8; padding:20px;">بدون پوزیشن باز</div>',
                    unsafe_allow_html=True)
    panel_end()


# -------- Regime Performance --------
with col_reg:
    section("PnL BY REGIME")
    panel_start()
    if not trades_df.empty and "regime" in trades_df.columns and "pnl_amount" in trades_df.columns:
        grp = trades_df.groupby("regime").agg(
            trades=("pnl_amount", "size"),
            pnl=("pnl_amount", "sum"),
        ).reset_index()
        colors = ["#00ff9d" if v > 0 else "#ff4d6d" for v in grp["pnl"]]
        fig = go.Figure(go.Bar(
            x=grp["regime"], y=grp["pnl"],
            marker=dict(color=colors),
            text=grp["pnl"].round(1), textposition="outside",
            textfont=dict(color="#c8e6ff"),
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,15,30,0.35)",
            height=240, margin=dict(l=10, r=10, t=10, b=10), showlegend=False,
            xaxis=dict(gridcolor="rgba(0,120,200,0.12)"),
            yaxis=dict(gridcolor="rgba(0,120,200,0.12)"),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.markdown('<div style="text-align:center; color:#5f8ba8; padding:20px;">بدون داده</div>',
                    unsafe_allow_html=True)
    panel_end()

st.markdown("<br>", unsafe_allow_html=True)


# ============================================================
# [۴] ردیف پایین: Recent Trades + Win/Loss + Backtest Metrics
# ============================================================
col_tr, col_wl, col_bt = st.columns([2, 1, 1])

# -------- Recent Trades --------
with col_tr:
    section("RECENT TRADES")
    panel_start()
    if not trades_df.empty:
        show_cols = [c for c in ["id", "direction", "entry_price", "exit_price",
                                  "pnl_pct", "pnl_amount", "regime"]
                     if c in trades_df.columns]
        df_show = trades_df[show_cols].tail(10).copy()
        if "pnl_pct" in df_show.columns:
            df_show["pnl_pct"] = df_show["pnl_pct"].apply(lambda x: f"{x*100:+.2f}%")
        if "pnl_amount" in df_show.columns:
            df_show["pnl_amount"] = df_show["pnl_amount"].apply(lambda x: f"{x:+.2f}")
        st.dataframe(df_show, use_container_width=True, hide_index=True, height=250)
    else:
        st.markdown('<div style="text-align:center; color:#5f8ba8; padding:30px;">بدون معامله</div>',
                    unsafe_allow_html=True)
    panel_end()


# -------- Win/Loss Donut --------
with col_wl:
    section("WIN/LOSS")
    panel_start()
    if wins + losses > 0:
        fig = go.Figure(go.Pie(
            labels=["Wins", "Losses"],
            values=[wins, losses],
            hole=0.72,
            marker=dict(colors=["#00ff9d", "#ff4d6d"]),
            textinfo="none", hoverinfo="label+value",
        ))
        fig.update_layout(
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            height=230, margin=dict(l=5, r=5, t=5, b=5),
            showlegend=True,
            legend=dict(font=dict(color="#c8e6ff", size=10)),
            annotations=[dict(
                text=f"<b>{win_rate:.0f}%</b>", x=0.5, y=0.5,
                font_size=22, showarrow=False, font_color="#00d4ff",
            )],
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    else:
        st.markdown('<div style="text-align:center; color:#5f8ba8; padding:30px;">—</div>',
                    unsafe_allow_html=True)
    panel_end()


# -------- Backtest Metrics --------
with col_bt:
    section("BACKTEST")
    panel_start()
    if metrics_df is not None and not metrics_df.empty:
        m = metrics_df.iloc[0].to_dict()
        st.markdown(f"""
            <div style="padding:6px 0;">
                <div class="kpi-label">TOTAL RETURN</div>
                <div class="kpi-value" style="font-size:15px; color:{'#00ff9d' if m.get('total_return_pct', 0) > 0 else '#ff4d6d'};">
                    {m.get('total_return_pct', 0):+.2f}%
                </div>
            </div>
            <div style="padding:6px 0;">
                <div class="kpi-label">SHARPE</div>
                <div class="kpi-value" style="font-size:15px;">{m.get('sharpe', 0):.3f}</div>
            </div>
            <div style="padding:6px 0;">
                <div class="kpi-label">MAX DD</div>
                <div class="kpi-value" style="font-size:15px; color:#ff4d6d;">
                    {m.get('max_drawdown_pct', 0):.2f}%
                </div>
            </div>
            <div style="padding:6px 0;">
                <div class="kpi-label">WIN RATE</div>
                <div class="kpi-value" style="font-size:15px;">{m.get('win_rate', 0)*100:.1f}%</div>
            </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div style="text-align:center; color:#5f8ba8; padding:30px;">—</div>',
                    unsafe_allow_html=True)
    panel_end()


# ============================================================
# فوتر
# ============================================================
st.markdown("---")
st.caption(
    f"⚡ Crypto Prediction Bot · Operations Center v2.0 · "
    f"Auto-refresh 30s · {datetime.now(timezone.utc).strftime('%H:%M:%S')} UTC"
)

# Auto-refresh
try:
    from streamlit_autorefresh import st_autorefresh
    st_autorefresh(interval=30000, key="autorefresh")
except ImportError:
    pass