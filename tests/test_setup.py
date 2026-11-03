"""تست راه‌اندازی فاز ۰"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils import setup_logger, load_config
from src.utils.database import DatabaseManager


def test_config_load():
    config = load_config("config/settings.yaml")
    assert config["project"]["name"] == "Crypto Prediction Bot"
    assert config["regime"]["n_regimes"] == 4
    print("[OK] Config loaded")


def test_logger():
    logger = setup_logger("test")
    logger.info("Test log message")
    assert Path("logs/all.log").exists()
    print("[OK] Logger works")


def test_imports():
    import pandas, numpy, torch, xgboost, lightgbm, sklearn, hmmlearn, optuna
    print(f"[OK] Core imports work (torch {torch.__version__})")


def test_database():
    try:
        db = DatabaseManager()
        if db.test_connection():
            print("[OK] Database connection works")
        else:
            print("[SKIP] Database not running - will use file storage for now")
    except Exception as e:
        print(f"[SKIP] Database not available: {e}")


if __name__ == "__main__":
    print()
    print("=" * 50)
    print("Running Phase 0 Tests")
    print("=" * 50)
    print()
    test_config_load()
    test_logger()
    test_imports()
    test_database()
    print()
    print("=" * 50)
    print("Phase 0 Tests Complete!")
    print("=" * 50)
    print()
