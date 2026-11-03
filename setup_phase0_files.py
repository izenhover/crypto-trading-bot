# setup_phase0_files.py
"""
ساخت تمام فایل‌های محتوایی فاز ۰
اجرا: python setup_phase0_files.py
"""
from pathlib import Path

# ============================================================
# محتوای فایل‌ها
# ============================================================
FILES = {}

# ---------- config/settings.yaml ----------
FILES["config/settings.yaml"] = '''# config/settings.yaml
project:
  name: "Crypto Prediction Bot"
  version: "0.1.0"
  environment: "development"

exchange:
  name: "binance"
  testnet: true
  symbols:
    - "BTCUSDT"
    - "ETHUSDT"
  timeframes:
    - "1m"
    - "5m"
    - "30m"
    - "1h"
    - "4h"

data:
  storage:
    type: "postgresql"
    host: "localhost"
    port: 5432
    database: "crypto_bot"
    user: "crypto_user"
  redis:
    host: "localhost"
    port: 6379
    db: 0
  cache_ttl_seconds: 60
  historical_days: 180

features:
  technical:
    rsi_period: 14
    macd_fast: 12
    macd_slow: 26
    macd_signal: 9
    bb_period: 20
    bb_std: 2
    atr_period: 14
  lookback_windows: [5, 20, 60]
  target_horizon: 1

regime:
  n_regimes: 4
  hmm_iterations: 1000
  hmm_covariance: "full"
  features: ["returns", "volatility", "volume_change"]

models:
  xgboost:
    n_estimators: 200
    max_depth: 6
    learning_rate: 0.05
    subsample: 0.8
    colsample_bytree: 0.8
  lightgbm:
    n_estimators: 200
    max_depth: 6
    learning_rate: 0.05
    num_leaves: 31
  lstm:
    hidden_size: 128
    num_layers: 3
    dropout: 0.3
    sequence_length: 60
    epochs: 50
    batch_size: 64
    learning_rate: 0.001
  random_forest:
    n_estimators: 200
    max_depth: 10
    min_samples_split: 5
  ensemble:
    method: "meta_learner"
    weights_by_regime:
      BULL: {xgboost: 0.4, lightgbm: 0.3, lstm: 0.2, rf: 0.1}
      BEAR: {xgboost: 0.5, lightgbm: 0.3, lstm: 0.1, rf: 0.1}
      SIDEWAYS_HIGH: {xgboost: 0.3, lightgbm: 0.3, lstm: 0.3, rf: 0.1}
      SIDEWAYS_LOW: {xgboost: 0.3, lightgbm: 0.2, lstm: 0.3, rf: 0.2}

signal:
  threshold_buy: 0.80
  threshold_sell: 0.20
  min_confirmations: 2
  require_closed_candle: true

money_flow:
  whale_alert:
    min_value_usd: 1000000
    check_interval_seconds: 60
  anomaly:
    contamination: 0.05
    window_minutes: 30
  circuit_breaker:
    max_whale_outflow_usd: 100000000
    anomaly_threshold: 0.95

risk:
  kelly_fraction: 0.25
  max_risk_per_trade: 0.02
  atr_stop_multiplier: 1.5
  atr_take_profit_multiplier: 3.0
  max_open_positions: 3
  circuit_breaker:
    max_consecutive_losses: 3
    max_daily_loss_pct: 0.05
    halt_duration_hours: 24

backtest:
  train_window_days: 30
  test_window_days: 7
  step_days: 7
  commission_pct: 0.001
  slippage_pct: 0.0005
  initial_capital: 10000

monitoring:
  log_level: "INFO"
  telegram:
    enabled: false
  dashboard:
    port: 8501
'''

# ---------- config/api_keys.env.example ----------
FILES["config/api_keys.env.example"] = '''# config/api_keys.env.example
# این فایل را کپی کنید به config/api_keys.env و مقادیر را پر کنید

# ============ Binance ============
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_API_SECRET=your_binance_secret_here
BINANCE_TESTNET=true

# ============ CoinGecko (رایگان) ============
COINGECKO_API_KEY=your_coingecko_key_here

# ============ Whale Alert (فاز ۵) ============
WHALE_ALERT_API_KEY=your_whale_alert_key_here

# ============ Glassnode (اختیاری) ============
GLASSNODE_API_KEY=your_glassnode_key_here

# ============ Telegram (هشدارها) ============
TELEGRAM_BOT_TOKEN=your_bot_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# ============ Database ============
DB_PASSWORD=crypto_pass_change_me
'''

# ---------- config/logging.yaml ----------
FILES["config/logging.yaml"] = '''# config/logging.yaml
version: 1
disable_existing_loggers: false

formats:
  default:
    format: "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}"

handlers:
  console:
    sink: "sys.stdout"
    level: "INFO"
    colorize: true

  file_all:
    sink: "logs/all.log"
    level: "DEBUG"
    rotation: "10 MB"
    retention: "30 days"
    compression: "zip"
    encoding: "utf-8"

  file_errors:
    sink: "logs/errors.log"
    level: "ERROR"
    rotation: "10 MB"
    retention: "90 days"
    compression: "zip"
    encoding: "utf-8"
'''

# ---------- docker-compose.yml ----------
FILES["docker-compose.yml"] = '''version: "3.9"

services:
  postgres:
    image: postgres:16-alpine
    container_name: crypto_bot_postgres
    restart: unless-stopped
    environment:
      POSTGRES_USER: crypto_user
      POSTGRES_PASSWORD: crypto_pass_change_me
      POSTGRES_DB: crypto_bot
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine
    container_name: crypto_bot_redis
    restart: unless-stopped
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    command: redis-server --appendonly yes

volumes:
  postgres_data:
  redis_data:
'''

# ---------- src/utils/logger.py ----------
FILES["src/utils/logger.py"] = '''"""ماژول لاگ‌گیری مرکزی با Loguru"""
import sys
from pathlib import Path
from loguru import logger

LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)


def setup_logger(name: str = "root", level: str = "INFO"):
    """راه‌اندازی لاگر با فرمت یکنواخت"""
    logger.remove()

    logger.add(
        sys.stdout,
        level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan> - <level>{message}</level>",
        colorize=True,
    )

    logger.add(
        LOG_DIR / "all.log",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="30 days",
        compression="zip",
        encoding="utf-8",
    )

    logger.add(
        LOG_DIR / "errors.log",
        level="ERROR",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
        rotation="10 MB",
        retention="90 days",
        compression="zip",
        backtrace=True,
        diagnose=True,
    )

    return logger.bind(logger_name=name)


def get_trade_logger():
    """لاگر مخصوص معاملات"""
    return logger.bind(logger_name="trades")


default_logger = setup_logger("root", "INFO")
'''

# ---------- src/utils/config_loader.py ----------
FILES["src/utils/config_loader.py"] = '''"""بارگذاری تنظیمات از YAML و متغیرهای محیطی"""
import os
from pathlib import Path
from typing import Any, Dict
import yaml
from dotenv import load_dotenv

ENV_FILE = Path("config/api_keys.env")
if ENV_FILE.exists():
    load_dotenv(ENV_FILE)


def load_config(config_path: str = "config/settings.yaml") -> Dict[str, Any]:
    """بارگذاری تنظیمات اصلی"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config


def get_api_key(key_name: str, required: bool = True):
    """دریافت کلید API از متغیرهای محیطی"""
    value = os.getenv(key_name)
    if value is None and required:
        raise ValueError(f"API key \'{key_name}\' not found in environment")
    return value


class Config:
    """دسترسی نقطه‌ای به تنظیمات"""
    def __init__(self, config_dict: Dict):
        self._config = config_dict

    def __getattr__(self, name: str):
        if name in self._config:
            value = self._config[name]
            if isinstance(value, dict):
                return Config(value)
            return value
        raise AttributeError(f"Config has no attribute \'{name}\'")

    def get(self, key: str, default=None):
        return self._config.get(key, default)
'''

# ---------- src/utils/time_utils.py ----------
FILES["src/utils/time_utils.py"] = '''"""ابزارهای زمان"""
from datetime import datetime, timezone, timedelta


def now_utc() -> datetime:
    return datetime.now(timezone.utc)


def to_timestamp(dt: datetime) -> int:
    """تبدیل datetime به timestamp میلی‌ثانیه"""
    return int(dt.timestamp() * 1000)


def from_timestamp(ts: int) -> datetime:
    """تبدیل timestamp میلی‌ثانیه به datetime"""
    return datetime.fromtimestamp(ts / 1000, tz=timezone.utc)


def align_to_candle(dt: datetime, interval_minutes: int) -> datetime:
    """هم‌ترازی زمان با شروع کندل"""
    minutes = (dt.minute // interval_minutes) * interval_minutes
    return dt.replace(minute=minutes, second=0, microsecond=0)


def is_candle_closed(dt: datetime, interval_minutes: int) -> bool:
    """بررسی بسته شدن کندل"""
    next_candle = align_to_candle(dt, interval_minutes) + timedelta(minutes=interval_minutes)
    return dt >= next_candle
'''

# ---------- src/utils/database.py ----------
FILES["src/utils/database.py"] = '''"""مدیریت دیتابیس PostgreSQL و Redis (اختیاری)"""
from contextlib import contextmanager
from typing import Optional
from loguru import logger

try:
    import redis
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker, Session
    HAS_DB_LIBS = True
except ImportError:
    HAS_DB_LIBS = False
    logger.warning("Database libraries not installed")

from src.utils.config_loader import load_config


class DatabaseManager:
    def __init__(self, config: Optional[dict] = None):
        self.config = config or load_config()
        self.engine = None
        self.SessionLocal = None
        self.redis = None
        if HAS_DB_LIBS:
            self._setup_postgres()
            self._setup_redis()

    def _setup_postgres(self):
        try:
            db_config = self.config["data"]["storage"]
            url = (
                f"postgresql://{db_config[\'user\']}:crypto_pass_change_me"
                f"@{db_config[\'host\']}:{db_config[\'port\']}/{db_config[\'database\']}"
            )
            self.engine = create_engine(url, pool_size=5, max_overflow=10)
            self.SessionLocal = sessionmaker(bind=self.engine)
            logger.info("PostgreSQL engine created")
        except Exception as e:
            logger.warning(f"PostgreSQL setup failed: {e}")

    def _setup_redis(self):
        try:
            redis_config = self.config["data"]["redis"]
            self.redis = redis.Redis(
                host=redis_config["host"],
                port=redis_config["port"],
                db=redis_config["db"],
                decode_responses=True,
            )
            self.redis.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis = None

    @contextmanager
    def session(self) -> "Session":
        if self.SessionLocal is None:
            raise RuntimeError("Database not configured")
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error(f"DB session error: {e}")
            raise
        finally:
            session.close()

    def test_connection(self) -> bool:
        if self.engine is None:
            return False
        try:
            with self.engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            return True
        except Exception as e:
            logger.error(f"DB connection test failed: {e}")
            return False
'''

# ---------- src/utils/__init__.py ----------
FILES["src/utils/__init__.py"] = '''from src.utils.logger import setup_logger, get_trade_logger
from src.utils.config_loader import load_config, get_api_key, Config
from src.utils.time_utils import now_utc, to_timestamp, from_timestamp, align_to_candle
from src.utils.database import DatabaseManager

__all__ = [
    "setup_logger",
    "get_trade_logger",
    "load_config",
    "get_api_key",
    "Config",
    "now_utc",
    "to_timestamp",
    "from_timestamp",
    "align_to_candle",
    "DatabaseManager",
]
'''

# ---------- main.py ----------
FILES["main.py"] = '''"""نقطه ورود اصلی ربات"""
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config

logger = setup_logger("main")


def main():
    logger.info("Starting Crypto Prediction Bot...")
    config = load_config("config/settings.yaml")
    logger.info(f"Loaded config: {config[\'project\'][\'name\']} v{config[\'project\'][\'version\']}")
    logger.info("Phase 0 complete. Ready for Phase 1: Data Pipeline")


if __name__ == "__main__":
    main()
'''

# ---------- tests/test_setup.py ----------
FILES["tests/test_setup.py"] = '''"""تست راه‌اندازی فاز ۰"""
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
'''


# ============================================================
# اجرای ساخت فایل‌ها
# ============================================================
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
    print("  1. python tests/test_setup.py")
    print("  2. python main.py")
    print()


if __name__ == "__main__":
    main()