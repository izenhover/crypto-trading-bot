# setup_phase10_files.py
"""ساخت فایل‌های فاز ۱۰: یادگیری مستمر"""
from pathlib import Path

FILES = {}

# ---------- src/utils/versioning.py ----------
FILES["src/utils/versioning.py"] = '''"""مدیریت نسخه مدل‌ها"""
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from loguru import logger

MODEL_DIR = Path("data/models")
ARCHIVE_DIR = MODEL_DIR / "archive"
ARCHIVE_DIR.mkdir(parents=True, exist_ok=True)


def archive_current_models(tag: str = None) -> str:
    """بک‌آپ گرفتن از مدل‌های فعلی با timestamp"""
    if tag is None:
        tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    dest = ARCHIVE_DIR / tag
    dest.mkdir(parents=True, exist_ok=True)

    archived = []
    for file in MODEL_DIR.glob("*"):
        if file.is_file() and file.suffix in (".pkl", ".pt"):
            shutil.copy2(file, dest / file.name)
            archived.append(file.name)

    logger.info(f"Archived {len(archived)} models to {dest}")
    return tag


def save_metrics_snapshot(metrics: dict, tag: str):
    """ذخیره snapshot از معیارها"""
    path = ARCHIVE_DIR / tag / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    logger.info(f"Saved metrics snapshot: {path}")


def load_metrics_snapshot(tag: str) -> dict:
    """بارگذاری معیارهای یک نسخه"""
    path = ARCHIVE_DIR / tag / "metrics.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def list_archives() -> list:
    """لیست نسخه‌های ذخیره‌شده"""
    return sorted([d.name for d in ARCHIVE_DIR.iterdir() if d.is_dir()])


def rollback(tag: str):
    """بازگردانی به نسخه قبلی"""
    src = ARCHIVE_DIR / tag
    if not src.exists():
        raise ValueError(f"Archive not found: {tag}")
    for f in src.glob("*"):
        if f.is_file() and f.suffix in (".pkl", ".pt"):
            shutil.copy2(f, MODEL_DIR / f.name)
    logger.info(f"Rolled back models to: {tag}")
'''

# ---------- monitoring/drift_detection.py ----------
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
    """محاسبه توزیع مرجع از داده آموزش"""
    ref = {}
    for col in feature_cols:
        if col in df.columns:
            vals = df[col].dropna().values
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
    logger.info(f"Saved reference stats: {path} ({len(feature_cols)} features)")


def detect_drift(df: pd.DataFrame, feature_cols: list,
                 reference_name: str = "reference",
                 threshold: float = 0.01) -> dict:
    """
    تشخیص drift با KS-test.
    p_value < threshold → drift detected
    """
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
    """مقایسه عملکرد فعلی با مرجع"""
    drop = reference_accuracy - current_accuracy
    return {
        "has_drift": drop > threshold,
        "current_accuracy": current_accuracy,
        "reference_accuracy": reference_accuracy,
        "drop": drop,
        "threshold": threshold,
    }
'''

# ---------- monitoring/performance_tracker.py ----------
FILES["monitoring/performance_tracker.py"] = '''"""پیگیری عملکرد مدل‌ها در طول زمان"""
from pathlib import Path
from datetime import datetime, timezone
import json
from loguru import logger

PERF_DIR = Path("data/performance")
PERF_DIR.mkdir(parents=True, exist_ok=True)

PERF_FILE = PERF_DIR / "performance_log.jsonl"


def log_performance(metrics: dict, tag: str = "daily"):
    """ثبت عملکرد"""
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tag": tag,
        **metrics,
    }
    with open(PERF_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\\n")


def get_recent(n: int = 30) -> list:
    """آخرین N رکورد"""
    if not PERF_FILE.exists():
        return []
    lines = PERF_FILE.read_text(encoding="utf-8").strip().split("\\n")
    records = [json.loads(l) for l in lines if l]
    return records[-n:]


def compute_trend_metrics(n: int = 30) -> dict:
    """محاسبه روند عملکرد"""
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

# ---------- monitoring/__init__.py ----------
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

# ---------- scripts/retrain_models.py ----------
FILES["scripts/retrain_models.py"] = '''"""بازآموزی خودکار مدل‌ها"""
import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

import pandas as pd
from loguru import logger

from src.utils.versioning import archive_current_models, save_metrics_snapshot
from monitoring import log_performance
from src.utils.logger import setup_logger


def main():
    setup_logger("retrain")
    logger.info("=" * 60)
    logger.info("🤖 Starting automated retraining")
    logger.info("=" * 60)

    # ۱. بک‌آپ از مدل‌های فعلی
    tag = archive_current_models()
    logger.info(f"Archived current models with tag: {tag}")

    # ۲. دریافت داده جدید (اختیاری — اگر آنلاین هست)
    try:
        from src.execution.live_data import LiveDataFetcher
        from src.data_collection.storage import save_candles
        fetcher = LiveDataFetcher("BTC/USDT", "1h")
        new_data = fetcher.fetch_recent(limit=500)
        if not new_data.empty:
            save_candles(new_data, "BTCUSDT", "1h")
            logger.info(f"Added {len(new_data)} new candles")
    except Exception as e:
        logger.warning(f"Live data update skipped: {e}")

    # ۳. بازسازی ویژگی‌ها
    logger.info("Rebuilding features...")
    from src.feature_engineering.pipeline import build_all
    build_all()

    # ۴. بازآموزی رژیم
    logger.info("Retraining regime detector...")
    from src.regime_detection import RegimeDetector
    from src.feature_engineering.pipeline import get_processed_path
    df = pd.read_parquet(get_processed_path("BTCUSDT", "1h"))
    detector = RegimeDetector(n_regimes=4)
    detector.fit(df)
    detector.save()

    # ۵. بازآموزی anomaly
    logger.info("Retraining anomaly detector...")
    from src.money_flow import FlowAnomalyDetector
    anomaly = FlowAnomalyDetector(contamination=0.05)
    anomaly.fit(df)
    anomaly.save()

    # ۶. بازآموزی مدل‌های Ensemble
    logger.info("Retraining ensemble models...")
    import subprocess
    result = subprocess.run(
        [sys.executable, "scripts/train_models.py"],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        logger.error(f"train_models failed:\\n{result.stderr}")
        return
    logger.info("Ensemble retrained successfully")

    # ۷. ذخیره snapshot
    save_metrics_snapshot({"tag": tag, "retrained_at": str(pd.Timestamp.now())}, tag)
    log_performance({"event": "retrain", "tag": tag})

    logger.info("=" * 60)
    logger.info(f"✅ Retraining complete. Archive tag: {tag}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
'''

# ---------- scripts/monitor_drift.py ----------
FILES["scripts/monitor_drift.py"] = '''"""پایش Drift"""
import sys
import warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")

import pandas as pd
from loguru import logger

from monitoring import compute_reference, detect_drift, log_performance
from src.feature_engineering.pipeline import get_processed_path
from src.models.dataset import get_feature_columns
from src.utils.logger import setup_logger


def main():
    setup_logger("drift")
    df = pd.read_parquet(get_processed_path("BTCUSDT", "1h"))
    feature_cols = get_feature_columns(df)

    # اگر reference موجود نیست، از ۷۰٪ اول به‌عنوان مرجع استفاده کن
    ref_path = Path("data/drift/reference_stats.json")
    if not ref_path.exists():
        logger.info("Creating reference from first 70% of data")
        split = int(len(df) * 0.7)
        compute_reference(df.iloc[:split], feature_cols, name="reference")

    # چک drift روی ۳۰٪ آخر
    recent = df.tail(500)
    drift = detect_drift(recent, feature_cols, reference_name="reference")

    logger.info("=" * 60)
    logger.info("=== DRIFT REPORT ===")
    logger.info("=" * 60)
    logger.info(f"Has drift: {drift['has_drift']}")
    logger.info(f"Drifted features: {drift.get('drifted_features', [])}")
    logger.info(f"Drift ratio: {drift.get('drift_ratio', 0):.2%}")

    log_performance({
        "event": "drift_check",
        "has_drift": drift["has_drift"],
        "drift_ratio": drift.get("drift_ratio", 0),
    })

    if drift["has_drift"]:
        logger.warning("⚠️ Drift detected! Consider retraining.")
    else:
        logger.info("✅ No significant drift")


if __name__ == "__main__":
    main()
'''

# ---------- scripts/run_daily.py ----------
FILES["scripts/run_daily.py"] = '''"""اجرای روزانه: پایش drift + بازآموزی در صورت نیاز"""
import sys
import subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from loguru import logger
from src.utils.logger import setup_logger


def run(script):
    logger.info(f"Running {script}...")
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"{script} failed:\\n{result.stderr}")
        return False
    logger.info(f"{script} OK")
    return True


def main():
    setup_logger("daily")
    logger.info("=== Daily Maintenance ===")

    # ۱. چک drift
    run("scripts/monitor_drift.py")

    # ۲. اگر drift تشخیص داده شد، بازآموزی
    drift_file = Path("data/performance/performance_log.jsonl")
    if drift_file.exists():
        lines = drift_file.read_text(encoding="utf-8").strip().split("\\n")
        last = lines[-1] if lines else ""
        if "drift_check" in last and '"has_drift": true' in last.lower():
            logger.warning("Drift detected — retraining")
            run("scripts/retrain_models.py")
        else:
            logger.info("No drift — retraining skipped")


if __name__ == "__main__":
    main()
'''

# ---------- tests/test_versioning.py ----------
FILES["tests/test_versioning.py"] = '''"""تست نسخه‌بندی"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.versioning import archive_current_models, list_archives, load_metrics_snapshot
from monitoring import detect_drift, compute_reference, log_performance, get_recent
from src.utils.logger import setup_logger

setup_logger("test")


def main():
    print()
    print("=" * 60)
    print("Testing Versioning & Drift")
    print("=" * 60)

    # بک‌آپ
    tag = archive_current_models(tag="test_backup")
    print(f"[OK] Archived with tag: {tag}")

    archives = list_archives()
    print(f"[OK] Archives: {archives[-3:]}")

    # Performance log
    log_performance({"accuracy": 0.55, "tag": "test"})
    recent = get_recent(5)
    print(f"[OK] Recent perf records: {len(recent)}")

    print()
    print("=" * 60)
    print("Versioning test passed!")
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
    print("  1. python tests/test_versioning.py")
    print("  2. python scripts/monitor_drift.py")
    print("  3. python scripts/run_daily.py")
    print()


if __name__ == "__main__":
    main()