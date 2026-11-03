"""ماژول لاگ‌گیری مرکزی با Loguru"""
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
