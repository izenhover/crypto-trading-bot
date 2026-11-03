"""تست نسخه‌بندی"""
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
