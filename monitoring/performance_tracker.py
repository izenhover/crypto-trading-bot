"""پیگیری عملکرد مدل‌ها در طول زمان"""
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
        f.write(json.dumps(record, default=str) + "\n")


def get_recent(n: int = 30) -> list:
    if not PERF_FILE.exists():
        return []
    lines = PERF_FILE.read_text(encoding="utf-8").strip().split("\n")
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
