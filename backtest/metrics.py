"""معیارهای ارزیابی عملکرد"""
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
