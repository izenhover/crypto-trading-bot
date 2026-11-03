"""زمان‌بند به‌روزرسانی داده"""
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.interval import IntervalTrigger
from loguru import logger


def create_scheduler():
    return BlockingScheduler(timezone="UTC")


def add_historical_refresh_job(scheduler, job_func, interval_minutes=30):
    """هر N دقیقه داده تاریخی را به‌روزرسانی کن"""
    scheduler.add_job(
        job_func,
        trigger=IntervalTrigger(minutes=interval_minutes),
        id="historical_refresh",
        replace_existing=True,
        max_instances=1,
    )
    logger.info(f"Scheduled historical refresh every {interval_minutes} min")
