# src/regime_detection/regime_validator.py
"""اعتبارسنجی و آمار رژیم‌ها"""
import numpy as np
import pandas as pd


def regime_distribution(regimes: pd.Series) -> pd.DataFrame:
    """توزیع رژیم‌ها"""
    counts = regimes.value_counts()
    pct = (counts / len(regimes) * 100).round(2)
    return pd.DataFrame({"count": counts, "pct": pct})


def regime_transitions(regimes: pd.Series) -> pd.DataFrame:
    """ماتریس انتقال بین رژیم‌ها"""
    labels = sorted(regimes.unique())
    matrix = pd.DataFrame(0, index=labels, columns=labels)

    prev = None
    for r in regimes:
        if prev is not None:
            matrix.loc[prev, r] += 1
        prev = r

    row_sums = matrix.sum(axis=1).replace(0, np.nan)
    return matrix.div(row_sums, axis=0).fillna(0).round(3)


def regime_durations(regimes: pd.Series) -> pd.DataFrame:
    """میانگین طول هر رژیم (تعداد کندل متوالی)"""
    if regimes.empty:
        return pd.DataFrame(columns=["mean", "min", "max", "count"])

    # کپی و تغییر نام برای جلوگیری از ابهام index/column
    s = regimes.copy()
    s.name = "state"
    s = s.reset_index(drop=True)

    # شناسه هر بلوک متوالی از رژیم
    block_id = (s != s.shift()).cumsum()

    # برای هر بلوک: نام رژیم و طول
    blocks = pd.DataFrame({
        "block_id": block_id.values,
        "state": s.values,
    })
    durations = blocks.groupby("block_id").agg(
        state=("state", "first"),
        duration=("state", "size"),
    )

    # گروه‌بندی بر اساس نام رژیم و آماره‌ها
    result = durations.groupby("state")["duration"].agg(
        ["mean", "min", "max", "count"]
    ).round(2)
    result.index.name = "regime"
    return result