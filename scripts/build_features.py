"""اسکریپت ساخت ویژگی‌ها"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.feature_engineering.pipeline import build_all
from src.utils.logger import setup_logger


def main():
    setup_logger("features")
    build_all()


if __name__ == "__main__":
    main()
