"""مدیریت نسخه مدل‌ها"""
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
