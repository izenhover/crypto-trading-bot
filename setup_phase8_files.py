# setup_phase8_files.py
"""ساخت فایل‌های فاز ۸: بک‌تست"""
from pathlib import Path

FILES = {}

# ---------- backtest/metrics.py ----------
FILES["backtest/metrics.py"] = '''"""معیارهای ارزیابی عملکرد"""
import numpy as np
import pandas as pd


def total_return(equity: pd.Series) -> float:
    if len(equity) < 2:
        return 0.0
    return float(equity.iloc[-1] / equity.iloc[0] - 1)


def cagr(equity: pd.Series, periods_per_year: int = 8760) -> float:
    """بازده سالانه مرکب (برای تایم‌فریم 1h: 8760 ساعت در سال)"""
    if len(equity) < 2:
        return 0.0
    n = len(equity)
    total = equity.iloc[-1] / equity.iloc[0]
    years = n / periods_per_year
    if years <= 0 or total <= 0:
        return 0.0
    return float(total ** (1 / years) - 1)


def sharpe_ratio(returns: pd.Series, risk_free: float = 0.0,
                 periods_per_year: int = 8760) -> float:
    """نسبت شارپ"""
    if returns.std() == 0 or len(returns) < 2:
        return 0.0
    excess = returns - risk_free / periods_per_year
    return float(np.sqrt(periods_per_year) * excess.mean() / returns.std())


def sortino_ratio(returns: pd.Series, risk_free: float = 0.0,
                  periods_per_year: int = 8760) -> float:
    """نسبت سورتینو (فقط نوسان منفی)"""
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    excess = returns.mean() - risk_free / periods_per_year
    return float(np.sqrt(periods_per_year) * excess / downside.std())


def max_drawdown(equity: pd.Series) -> float:
    """حداکثر افت سرمایه"""
    if len(equity) < 2:
        return 0.0
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    return float(drawdown.min())


def profit_factor(trades_df: pd.DataFrame) -> float:
    """نسبت مجموع سود به مجموع زیان"""
    if trades_df.empty or "pnl_amount" not in trades_df.columns:
        return 0.0
    gains = trades_df[trades_df["pnl_amount"] > 0]["pnl_amount"].sum()
    losses = abs(trades_df[trades_df["pnl_amount"] < 0]["pnl_amount"].sum())
    if losses == 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def win_rate(trades_df: pd.DataFrame) -> float:
    if trades_df.empty:
        return 0.0
    return float((trades_df["pnl_amount"] > 0).mean())


def avg_win_loss_ratio(trades_df: pd.DataFrame) -> float:
    if trades_df.empty:
        return 0.0
    wins = trades_df[trades_df["pnl_amount"] > 0]["pnl_amount"]
    losses = trades_df[trades_df["pnl_amount"] < 0]["pnl_amount"]
    if len(wins) == 0 or len(losses) == 0:
        return 0.0
    return float(abs(wins.mean() / losses.mean()))


def calmar_ratio(equity: pd.Series, periods_per_year: int = 8760) -> float:
    """نسبت کالمار: CAGR / MaxDD"""
    mdd = abs(max_drawdown(equity))
    if mdd == 0:
        return 0.0
    return float(cagr(equity, periods_per_year) / mdd)


def expectancy(trades_df: pd.DataFrame) -> float:
    """امید ریاضی هر معامله (میانگین PnL)"""
    if trades_df.empty:
        return 0.0
    return float(trades_df["pnl_amount"].mean())


def compute_all_metrics(equity: pd.Series, trades_df: pd.DataFrame,
                        periods_per_year: int = 8760) -> dict:
    """محاسبه تمام معیارها"""
    returns = equity.pct_change().dropna()
    return {
        "total_return_pct": round(total_return(equity) * 100, 2),
        "cagr_pct": round(cagr(equity, periods_per_year) * 100, 2),
        "sharpe": round(sharpe_ratio(returns, periods_per_year=periods_per_year), 3),
        "sortino": round(sortino_ratio(returns, periods_per_year=periods_per_year), 3),
        "max_drawdown_pct": round(max_drawdown(equity) * 100, 2),
        "calmar": round(calmar_ratio(equity, periods_per_year), 3),
        "total_trades": int(len(trades_df)),
        "win_rate": round(win_rate(trades_df), 4),
        "profit_factor": round(profit_factor(trades_df), 3),
        "avg_win_loss_ratio": round(avg_win_loss_ratio(trades_df), 3),
        "expectancy_usdt": round(expectancy(trades_df), 4),
    }
'''

# ---------- backtest/simulator.py ----------
FILES["backtest/simulator.py"] = '''"""شبیه‌ساز معاملات"""
from dataclasses import dataclass
from typing import Optional
import pandas as pd
from loguru import logger

from src.signal_engine import Signal, SignalType
from src.risk_management import (
    fractional_kelly, confidence_based_size,
    compute_position_value, Portfolio,
)


@dataclass
class SimConfig:
    initial_capital: float = 10000.0
    commission_pct: float = 0.001
    slippage_pct: float = 0.0005
    max_open_positions: int = 3
    kelly_fraction: float = 0.25
    max_risk_per_trade: float = 0.02
    min_confidence: float = 0.75


class Simulator:
    """شبیه‌ساز معامله با سیگنال‌های تولیدشده"""

    def __init__(self, config: SimConfig):
        self.config = config
        self.portfolio = Portfolio(
            initial_capital=config.initial_capital,
            max_open=config.max_open_positions,
        )
        self.equity_curve = []  # (timestamp, capital)

    def _apply_costs(self, price: float, direction: str, is_entry: bool) -> float:
        c = self.config.commission_pct
        s = self.config.slippage_pct
        total = c + s
        if direction.upper() == "BUY":
            return price * (1 + total) if is_entry else price * (1 - total)
        return price * (1 - total) if is_entry else price * (1 + total)

    def process_candle(self, timestamp, candle: pd.Series, signal: Signal):
        """پردازش یک کندل"""
        # ۱. اول چک بسته شدن پوزیشن‌های باز
        self.portfolio.check_exits(
            current_high=candle["high"],
            current_low=candle["low"],
            current_close=candle["close"],
        )

        # ۲. اگر سیگنال BUY/SELL است
        if signal.signal in (SignalType.BUY, SignalType.SELL):
            can, reason = self.portfolio.can_open(signal.signal.value)
            if can and signal.confidence >= self.config.min_confidence:
                risk_pct = self._compute_risk_pct(signal)
                if risk_pct > 0:
                    entry_price = self._apply_costs(
                        candle["close"], signal.signal.value, is_entry=True
                    )
                    sizing = compute_position_value(
                        capital=self.portfolio.available_capital,
                        risk_pct=risk_pct,
                        entry_price=entry_price,
                        stop_loss=signal.stop_loss,
                    )
                    if sizing["quantity"] > 0:
                        self.portfolio.open_position(
                            symbol="BTCUSDT",
                            direction=signal.signal.value,
                            entry_price=entry_price,
                            quantity=sizing["quantity"],
                            position_value=sizing["position_value"],
                            stop_loss=signal.stop_loss,
                            take_profit=signal.take_profit,
                            confidence=signal.confidence,
                            regime=signal.regime,
                        )

        # ۳. ثبت ارزش کل
        self.equity_curve.append((timestamp, self.portfolio.capital))

    def _compute_risk_pct(self, signal: Signal) -> float:
        """محاسبه درصد ریسک با Kelly یا Confidence"""
        stats = self.portfolio.stats()
        if stats["total_trades"] >= 10 and stats["win_loss_ratio"] > 0:
            return fractional_kelly(
                win_rate=stats["win_rate"],
                win_loss_ratio=stats["win_loss_ratio"],
                fraction=self.config.kelly_fraction,
                max_risk=self.config.max_risk_per_trade,
            )
        return confidence_based_size(
            signal.confidence, base_size=0.005,
            max_size=self.config.max_risk_per_trade,
        )

    def finalize(self, last_price: float):
        """بستن پوزیشن‌های باز در پایان"""
        for pos in list(self.portfolio.open_positions):
            self.portfolio.close_position(pos, last_price)

    def get_equity_series(self) -> pd.Series:
        if not self.equity_curve:
            return pd.Series(dtype=float)
        idx, vals = zip(*self.equity_curve)
        return pd.Series(vals, index=pd.Index(idx, name="timestamp"))

    def get_trades_df(self) -> pd.DataFrame:
        if not self.portfolio.closed_positions:
            return pd.DataFrame()
        return pd.DataFrame([p.to_dict() for p in self.portfolio.closed_positions])
'''

# ---------- backtest/walk_forward.py ----------
FILES["backtest/walk_forward.py"] = '''"""Walk-Forward Validation"""
import pandas as pd
from loguru import logger


def walk_forward_splits(
    n_samples: int,
    train_size: int,
    test_size: int,
    step_size: int = None,
):
    """
    تولید ایندکس‌های بازه‌های train/test به صورت متوالی
    train: پنجره آموزش (ثابت)
    test: پنجره تست (پس از train)
    step: گام جلو رفتن (پیش‌فرض = test_size)
    """
    if step_size is None:
        step_size = test_size
    start = 0
    while start + train_size + test_size <= n_samples:
        train_start = start
        train_end = start + train_size
        test_start = train_end
        test_end = test_start + test_size
        yield {
            "train": (train_start, train_end),
            "test": (test_start, test_end),
        }
        start += step_size


def generate_walk_forward_report(n_samples: int,
                                  train_size: int,
                                  test_size: int,
                                  step_size: int = None) -> dict:
    windows = list(walk_forward_splits(n_samples, train_size, test_size, step_size))
    logger.info(f"Walk-Forward: {len(windows)} windows, "
                f"train={train_size}, test={test_size}, step={step_size or test_size}")
    return {"n_windows": len(windows), "windows": windows}
'''

# ---------- backtest/report.py ----------
FILES["backtest/report.py"] = '''"""تولید گزارش بک‌تست"""
from pathlib import Path
import pandas as pd
from loguru import logger


REPORT_DIR = Path("data/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def print_metrics(metrics: dict, title: str = "Backtest Results"):
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"=== {title} ===")
    logger.info("=" * 60)
    for k, v in metrics.items():
        logger.info(f"  {k:<25}: {v}")
    logger.info("=" * 60)


def save_report(metrics: dict, trades_df: pd.DataFrame, equity: pd.Series,
                name: str = "backtest"):
    # ذخیره معیارها
    metrics_df = pd.DataFrame([metrics])
    metrics_path = REPORT_DIR / f"{name}_metrics.parquet"
    metrics_df.to_parquet(metrics_path)

    # ذخیره معاملات
    trades_path = None
    if not trades_df.empty:
        trades_path = REPORT_DIR / f"{name}_trades.parquet"
        trades_df.to_parquet(trades_path)

    # ذخیره equity
    equity_path = REPORT_DIR / f"{name}_equity.parquet"
    equity.to_frame("equity").to_parquet(equity_path)

    logger.info(f"Saved report: {metrics_path}")
    if trades_path:
        logger.info(f"Saved trades: {trades_path}")
    logger.info(f"Saved equity: {equity_path}")


def summarize_trades(trades_df: pd.DataFrame) -> pd.DataFrame:
    """خلاصه معاملات بر اساس رژیم"""
    if trades_df.empty or "regime" not in trades_df.columns:
        return pd.DataFrame()
    grouped = trades_df.groupby("regime").agg(
        trades=("pnl_amount", "size"),
        wins=("pnl_amount", lambda x: (x > 0).sum()),
        total_pnl=("pnl_amount", "sum"),
        avg_pnl=("pnl_amount", "mean"),
    ).round(4)
    grouped["win_rate"] = (grouped["wins"] / grouped["trades"]).round(4)
    return grouped
'''

# ---------- backtest/__init__.py ----------
FILES["backtest/__init__.py"] = '''from backtest.metrics import compute_all_metrics
from backtest.simulator import Simulator, SimConfig
from backtest.walk_forward import walk_forward_splits, generate_walk_forward_report
from backtest.report import print_metrics, save_report, summarize_trades

__all__ = [
    "compute_all_metrics",
    "Simulator",
    "SimConfig",
    "walk_forward_splits",
    "generate_walk_forward_report",
    "print_metrics",
    "save_report",
    "summarize_trades",
]
'''

# ---------- scripts/run_backtest.py ----------
FILES["scripts/run_backtest.py"] = '''"""اجرای بک‌تست کامل"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger

from src.feature_engineering.pipeline import get_processed_path
from src.models import Ensemble
from src.models.dataset import get_feature_columns
from src.regime_detection import RegimeDetector
from src.money_flow import FlowAnomalyDetector, CircuitBreaker
from src.signal_engine import SignalGenerator
from src.risk_management import Portfolio
from backtest import (
    Simulator, SimConfig, compute_all_metrics,
    print_metrics, save_report, summarize_trades,
    generate_walk_forward_report,
)
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger


def main():
    setup_logger("backtest")
    config = load_config()

    symbol, interval = "BTCUSDT", "1h"
    df = pd.read_parquet(get_processed_path(symbol, interval))
    logger.info(f"Loaded {len(df)} rows")

    # ماژول‌ها
    ensemble = Ensemble.load_all(include_lstm=False)
    regime_detector = RegimeDetector.load()
    anomaly_detector = FlowAnomalyDetector.load()

    sig_cfg = config["signal"]
    risk_cfg = config["risk"]
    bt_cfg = config["backtest"]

    generator = SignalGenerator(
        threshold_buy=sig_cfg["threshold_buy"],
        threshold_sell=sig_cfg["threshold_sell"],
        min_confirmations=sig_cfg["min_confirmations"],
        atr_stop_multiplier=risk_cfg["atr_stop_multiplier"],
        atr_take_profit_multiplier=risk_cfg["atr_take_profit_multiplier"],
    )

    sim_config = SimConfig(
        initial_capital=bt_cfg["initial_capital"],
        commission_pct=bt_cfg["commission_pct"],
        slippage_pct=bt_cfg["slippage_pct"],
        max_open_positions=risk_cfg["max_open_positions"],
        kelly_fraction=risk_cfg["kelly_fraction"],
        max_risk_per_trade=risk_cfg["max_risk_per_trade"],
    )
    simulator = Simulator(sim_config)

    cb_cfg = risk_cfg["circuit_breaker"]
    cb = CircuitBreaker(
        max_consecutive_losses=cb_cfg["max_consecutive_losses"],
        max_daily_loss_pct=cb_cfg["max_daily_loss_pct"],
        halt_duration_hours=cb_cfg["halt_duration_hours"],
    )

    feature_cols = get_feature_columns(df)
    start_idx = 500

    # Walk-Forward گزارش
    wf = generate_walk_forward_report(
        n_samples=len(df) - start_idx,
        train_size=int((len(df) - start_idx) * 0.6),
        test_size=int((len(df) - start_idx) * 0.1),
    )
    logger.info(f"Walk-Forward windows: {wf['n_windows']}")

    # شبیه‌سازی
    logger.info(f"Simulating {len(df) - start_idx} candles...")
    signals_generated = {"BUY": 0, "SELL": 0, "HOLD": 0}

    for i in range(start_idx, len(df)):
        current_df = df.iloc[:i+1]
        candle = df.iloc[i]

        if not cb.can_trade():
            simulator.process_candle(candle.name, candle, None)
            continue

        X_current = current_df[feature_cols].iloc[-1:]
        ensemble_score = float(ensemble.predict_proba(X_current)[0])

        try:
            regime = regime_detector.predict_current(current_df, window=150)
        except Exception:
            regime = "UNKNOWN"

        try:
            anomaly = anomaly_detector.current_score(current_df)
        except Exception:
            anomaly = {"anomaly_score": 0.0, "is_anomaly": False}

        cb.record_anomaly(anomaly["anomaly_score"], anomaly["is_anomaly"])

        signal = generator.generate(
            df=current_df,
            ensemble_score=ensemble_score,
            regime=regime,
            anomaly_score=anomaly["anomaly_score"],
            is_anomaly=anomaly["is_anomaly"],
            circuit_breaker_active=not cb.can_trade(),
        )

        signals_generated[signal.signal.value] += 1
        simulator.process_candle(candle.name, candle, signal)

    simulator.finalize(df.iloc[-1]["close"])

    # معیارها
    equity = simulator.get_equity_series()
    trades = simulator.get_trades_df()

    metrics = compute_all_metrics(equity, trades, periods_per_year=8760)
    metrics["signals_buy"] = signals_generated["BUY"]
    metrics["signals_sell"] = signals_generated["SELL"]
    metrics["signals_hold"] = signals_generated["HOLD"]

    print_metrics(metrics, title="Backtest Results (BTCUSDT 1h)")
    save_report(metrics, trades, equity, name=f"{symbol}_{interval}_backtest")

    # خلاصه بر اساس رژیم
    if not trades.empty:
        logger.info("")
        logger.info("=== Performance by Regime ===")
        logger.info("\\n" + summarize_trades(trades).to_string())


if __name__ == "__main__":
    main()
'''

# ---------- tests/test_backtest.py ----------
FILES["tests/test_backtest.py"] = '''"""تست ماژول‌های بک‌تست"""
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
'''


def main():
    root = Path(".")
    created = 0
    for file_path, content in FILES.items():
        path = root / file_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        created += 1
        print(f"  + {file_path}")
    print()
    print("=" * 50)
    print(f"Created {created} files successfully!")
    print("=" * 50)
    print()
    print("Next steps:")
    print("  1. python tests/test_backtest.py")
    print("  2. python scripts/run_backtest.py")
    print()


if __name__ == "__main__":
    main()