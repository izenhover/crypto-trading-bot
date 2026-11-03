"""Walk-Forward Validation"""
import pandas as pd
from loguru import logger


def walk_forward_splits(
    n_samples: int,
    train_size: int,
    test_size: int,
    step_size: int = None,
):
    """
    تولید ایندکس‌های بازه‌های train/test به صورت متوالی
    train: پنجره آموزش (ثابت)
    test: پنجره تست (پس از train)
    step: گام جلو رفتن (پیش‌فرض = test_size)
    """
    if step_size is None:
        step_size = test_size
    start = 0
    while start + train_size + test_size <= n_samples:
        train_start = start
        train_end = start + train_size
        test_start = train_end
        test_end = test_start + test_size
        yield {
            "train": (train_start, train_end),
            "test": (test_start, test_end),
        }
        start += step_size


def generate_walk_forward_report(n_samples: int,
                                  train_size: int,
                                  test_size: int,
                                  step_size: int = None) -> dict:
    windows = list(walk_forward_splits(n_samples, train_size, test_size, step_size))
    logger.info(f"Walk-Forward: {len(windows)} windows, "
                f"train={train_size}, test={test_size}, step={step_size or test_size}")
    return {"n_windows": len(windows), "windows": windows}
