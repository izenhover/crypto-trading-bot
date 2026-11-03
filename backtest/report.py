"""تولید گزارش بک‌تست"""
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
