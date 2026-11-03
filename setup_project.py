# setup_project.py
"""
اسکریپت راه‌اندازی اولیه پروژه ربات پیش‌بینی ارز دیجیتال
اجرا: python setup_project.py
"""
from pathlib import Path

PROJECT_NAME = "crypto_bot"

# ============ ساختار پوشه‌ها ============
FOLDERS = [
    "config",
    "data/raw",
    "data/processed",
    "data/models",
    "data/cache",
    "src/data_collection",
    "src/feature_engineering",
    "src/regime_detection",
    "src/models",
    "src/money_flow",
    "src/signal_engine",
    "src/risk_management",
    "src/execution",
    "src/utils",
    "backtest",
    "monitoring",
    "notebooks",
    "tests",
    "logs",
    "scripts",
]

# ============ فایل‌های خالی (touch) ============
EMPTY_FILES = [
    "src/__init__.py",
    "src/data_collection/__init__.py",
    "src/data_collection/binance_stream.py",
    "src/data_collection/historical.py",
    "src/data_collection/onchain.py",
    "src/data_collection/derivatives.py",
    "src/data_collection/sentiment.py",
    "src/data_collection/scheduler.py",
    "src/feature_engineering/__init__.py",
    "src/feature_engineering/technical.py",
    "src/feature_engineering/regime_features.py",
    "src/feature_engineering/flow_features.py",
    "src/feature_engineering/pipeline.py",
    "src/regime_detection/__init__.py",
    "src/regime_detection/hmm_regime.py",
    "src/regime_detection/regime_validator.py",
    "src/models/__init__.py",
    "src/models/dataset.py",
    "src/models/xgboost_model.py",
    "src/models/lightgbm_model.py",
    "src/models/lstm_model.py",
    "src/models/random_forest_model.py",
    "src/models/ensemble.py",
    "src/models/meta_learner.py",
    "src/money_flow/__init__.py",
    "src/money_flow/whale_alert.py",
    "src/money_flow/anomaly.py",
    "src/money_flow/circuit_breaker.py",
    "src/signal_engine/__init__.py",
    "src/signal_engine/signal_generator.py",
    "src/signal_engine/event_detector.py",
    "src/risk_management/__init__.py",
    "src/risk_management/position_sizing.py",
    "src/risk_management/stop_loss.py",
    "src/risk_management/portfolio.py",
    "src/execution/__init__.py",
    "src/execution/paper_trader.py",
    "src/execution/live_trader.py",
    "src/execution/order_manager.py",
    "src/utils/__init__.py",
    "src/utils/logger.py",
    "src/utils/database.py",
    "src/utils/config_loader.py",
    "src/utils/time_utils.py",
    "backtest/__init__.py",
    "backtest/walk_forward.py",
    "backtest/simulator.py",
    "backtest/metrics.py",
    "backtest/report.py",
    "monitoring/__init__.py",
    "monitoring/dashboard.py",
    "monitoring/telegram_alerts.py",
    "monitoring/drift_detection.py",
    "tests/__init__.py",
    "tests/test_data_collection.py",
    "tests/test_features.py",
    "tests/test_models.py",
    "tests/test_risk.py",
    "scripts/__init__.py",
    "scripts/run_backtest.py",
    "scripts/run_live.py",
    "scripts/retrain_models.py",
    "notebooks/.gitkeep",
    "data/raw/.gitkeep",
    "data/processed/.gitkeep",
    "data/models/.gitkeep",
    "data/cache/.gitkeep",
    "logs/.gitkeep",
]

# ============ محتوای فایل‌ها ============
FILES = {}

FILES["README.md"] = (
    "# Crypto Prediction Bot\n\n"
    "ربات پیش‌بینی قیمت ارز دیجیتال با معماری چندلایه تطبیقی.\n\n"
    "## نصب\n\n"
    "pip install -r requirements.txt\n"
    "cp config/api_keys.env.example config/api_keys.env\n"
    "# کلیدهای API را در config/api_keys.env وارد کنید\n\n"
    "## اجرا\n\n"
    "python scripts/run_live.py\n"
)

FILES[".gitignore"] = (
    "# Environment\n"
    "config/api_keys.env\n"
    ".env\n\n"
    "# Data\n"
    "data/raw/*\n"
    "data/processed/*\n"
    "data/models/*\n"
    "data/cache/*\n"
    "!data/**/.gitkeep\n\n"
    "# Logs\n"
    "logs/*\n"
    "!logs/.gitkeep\n\n"
    "# Python\n"
    "__pycache__/\n"
    "*.py[cod]\n"
    "*.egg-info/\n"
    ".venv/\n"
    "venv/\n"
    ".pytest_cache/\n\n"
    "# IDE\n"
    ".vscode/\n"
    ".idea/\n"
    "*.swp\n\n"
    "# Notebooks\n"
    ".ipynb_checkpoints/\n"
)

FILES["main.py"] = (
    '"""نقطه ورود اصلی ربات"""\n'
    "from src.utils.logger import setup_logger\n"
    "from src.utils.config_loader import load_config\n\n"
    'logger = setup_logger("main")\n\n\n'
    "def main():\n"
    '    logger.info("Starting Crypto Prediction Bot...")\n'
    '    config = load_config("config/settings.yaml")\n'
    '    logger.info(f"Loaded config: {config[\'project\'][\'name\']} v{config[\'project\'][\'version\']}")\n'
    '    logger.info("Ready for Phase 1: Data Pipeline")\n\n\n'
    'if __name__ == "__main__":\n'
    "    main()\n"
)


def main():
    base = Path(PROJECT_NAME)
    if base.exists():
        print(f"⚠️  پوشه {PROJECT_NAME} از قبل وجود دارد. فایل‌های جدید اضافه می‌شوند.")
    base.mkdir(exist_ok=True)

    # ساخت پوشه‌ها
    for folder in FOLDERS:
        (base / folder).mkdir(parents=True, exist_ok=True)
    print(f"✅ {len(FOLDERS)} پوشه ساخته شد")

    # ساخت فایل‌های خالی
    for file_path in EMPTY_FILES:
        (base / file_path).touch(exist_ok=True)
    print(f"✅ {len(EMPTY_FILES)} فایل خالی ساخته شد")

    # نوشتن محتوای فایل‌ها
    for file_path, content in FILES.items():
        (base / file_path).write_text(content, encoding="utf-8")
    print(f"✅ {len(FILES)} فایل محتوایی نوشته شد")

    print(f"\n🎉 پروژه در مسیر زیر ساخته شد: {base.absolute()}")
    print("\n📋 مراحل بعدی:")
    print("   1. cd crypto_bot")
    print("   2. pip install -r requirements.txt")
    print("   3. cp config/api_keys.env.example config/api_keys.env")
    print("   4. ویرایش config/api_keys.env با کلیدهای واقعی")


if __name__ == "__main__":
    main()