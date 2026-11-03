# fix_monitoring.py
"""بازسازی فایل‌های monitoring که ناقص مانده‌اند"""
from pathlib import Path

FILES = {}

FILES["monitoring/telegram_alerts.py"] = '''"""ارسال هشدار به تلگرام (اختیاری)"""
import os
import requests
from loguru import logger


class TelegramAlerter:
    def __init__(self, bot_token: str = None, chat_id: str = None):
        self.bot_token = bot_token or os.getenv("TELEGRAM_BOT_TOKEN")
        self.chat_id = chat_id or os.getenv("TELEGRAM_CHAT_ID")
        self.enabled = (
            self.bot_token
            and self.chat_id
            and self.bot_token != "your_bot_token_here"
        )
        if self.enabled:
            logger.info("Telegram alerter enabled")
        else:
            logger.info("Telegram alerter disabled (no credentials)")

    def send(self, message: str):
        if not self.enabled:
            logger.debug(f"[TELEGRAM-OFF] {message}")
            return False
        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            resp = requests.post(
                url,
                json={"chat_id": self.chat_id, "text": message, "parse_mode": "Markdown"},
                timeout=10,
            )
            return resp.ok
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    def send_signal(self, signal):
        emoji = "🟢" if signal.signal.value == "BUY" else "🔴" if signal.signal.value == "SELL" else "⚪"
        text = (
            f"{emoji} *{signal.signal.value}* BTCUSDT\\n"
            f"قیمت: `{signal.current_price:.2f}`\\n"
            f"اطمینان: `{signal.confidence*100:.1f}%` ({signal.confidence_level.value})\\n"
            f"رژیم: `{signal.regime}`\\n"
            f"SL: `{signal.stop_loss:.2f}` | TP: `{signal.take_profit:.2f}`\\n"
            f"تأییدها: `{signal.confirmations}/4`"
        )
        return self.send(text)
'''

FILES["monitoring/drift_detection.py"] = '''"""تشخیص Drift در توزیع داده‌ها"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from loguru import logger

DRIFT_DIR = Path("data/drift")
DRIFT_DIR.mkdir(parents=True, exist_ok=True)

REFERENCE_FILE = DRIFT_DIR / "reference_stats.json"


def compute_reference(df: pd.DataFrame, feature_cols: list, name: str = "reference"):
    ref = {}
    for col in feature_cols:
        if col in df.columns:
            vals = df[col].dropna().values
            if len(vals) == 0:
                continue
            ref[col] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "quantiles": [float(q) for q in np.quantile(vals, [0.1, 0.25, 0.5, 0.75, 0.9])],
                "sample": vals[-500:].tolist() if len(vals) >= 500 else vals.tolist(),
            }
    path = DRIFT_DIR / f"{name}_stats.json"
    path.write_text(json.dumps(ref), encoding="utf-8")
    logger.info(f"Saved reference stats: {path} ({len(ref)} features)")


def detect_drift(df: pd.DataFrame, feature_cols: list,
                 reference_name: str = "reference",
                 threshold: float = 0.01) -> dict:
    ref_path = DRIFT_DIR / f"{reference_name}_stats.json"
    if not ref_path.exists():
        return {"has_drift": False, "reason": "no reference"}

    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    drifts = []
    details = {}

    for col in feature_cols:
        if col not in ref or col not in df.columns:
            continue
        ref_sample = np.array(ref[col]["sample"])
        curr_sample = df[col].dropna().values[-500:]
        if len(curr_sample) < 50 or len(ref_sample) < 50:
            continue
        try:
            stat, p = ks_2samp(ref_sample, curr_sample)
            details[col] = {"ks_stat": float(stat), "p_value": float(p)}
            if p < threshold:
                drifts.append(col)
        except Exception:
            pass

    has_drift = len(drifts) > 0
    return {
        "has_drift": has_drift,
        "drifted_features": drifts,
        "drift_ratio": len(drifts) / max(len(feature_cols), 1),
        "details": details,
    }


def check_performance_drift(current_accuracy: float,
                            reference_accuracy: float,
                            threshold: float = 0.05) -> dict:
    drop = reference_accuracy - current_accuracy
    return {
        "has_drift": drop > threshold,
        "current_accuracy": current_accuracy,
        "reference_accuracy": reference_accuracy,
        "drop": drop,
        "threshold": threshold,
    }
'''

FILES["monitoring/performance_tracker.py"] = '''"""پیگیری عملکرد مدل‌ها در طول زمان"""
from pathlib import Path
from datetime import datetime, timezone
import json
from loguru import logger

PERF_DIR = Path("data/performance")
PERF_DIR.mkdir(parents=True, exist_ok=True)

PERF_FILE = PERF_DIR / "performance_log.jsonl"


def log_performance(metrics: dict, tag: str = "daily"):
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
        **metrics,
    }
    with open(PERF_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\\n")


def get_recent(n: int = 30) -> list:
    if not PERF_FILE.exists():
        return []
    lines = PERF_FILE.read_text(encoding="utf-8").strip().split("\\n")
    records = [json.loads(l) for l in lines if l]
    return records[-n:]


def compute_trend_metrics(n: int = 30) -> dict:
    records = get_recent(n)
    if len(records) < 2:
        return {}
    accs = [r.get("accuracy", 0) for r in records if "accuracy" in r]
    if len(accs) < 2:
        return {}
    return {
        "accuracy_mean": sum(accs) / len(accs),
        "accuracy_trend": accs[-1] - accs[0],
        "samples": len(accs),
    }
'''

FILES["monitoring/__init__.py"] = '''from monitoring.telegram_alerts import TelegramAlerter
from monitoring.drift_detection import (
    compute_reference, detect_drift, check_performance_drift,
)
from monitoring.performance_tracker import (
    log_performance, get_recent, compute_trend_metrics,
)

__all__ = [
    "TelegramAlerter",
    "compute_reference",
    "detect_drift",
    "check_performance_drift",
    "log_performance",
    "get_recent",
    "compute_trend_metrics",
]
'''


def main():
    root = Path(".")
    for path_str, content in FILES.items():
        path = root / path_str
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        size = len(content)
        print(f"  OK  {path_str:<45} ({size:>6} chars)")
    print()
    print("=" * 50)
    print(f"Fixed {len(FILES)} monitoring files successfully!")
    print("=" * 50)
    print()
    print("Next: run tests")
    print("  python tests/test_versioning.py")


if __name__ == "__main__":
    main()