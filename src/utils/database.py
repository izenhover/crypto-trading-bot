"""مدیریت دیتابیس PostgreSQL و Redis (اختیاری)"""
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
                f"postgresql://{db_config['user']}:crypto_pass_change_me"
                f"@{db_config['host']}:{db_config['port']}/{db_config['database']}"
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
