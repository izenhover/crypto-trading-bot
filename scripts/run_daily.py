"""اجرای روزانه: پایش drift + بازآموزی در صورت نیاز"""
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
        logger.error(f"{script} failed:\n{result.stderr}")
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
        lines = drift_file.read_text(encoding="utf-8").strip().split("\n")
        last = lines[-1] if lines else ""
        if "drift_check" in last and '"has_drift": true' in last.lower():
            logger.warning("Drift detected — retraining")
            run("scripts/retrain_models.py")
        else:
            logger.info("No drift — retraining skipped")


if __name__ == "__main__":
    main()
