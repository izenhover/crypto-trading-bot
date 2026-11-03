"""نقطه ورود اصلی ربات"""
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config

logger = setup_logger("main")


def main():
    logger.info("Starting Crypto Prediction Bot...")
    config = load_config("config/settings.yaml")
    logger.info(f"Loaded config: {config['project']['name']} v{config['project']['version']}")
    logger.info("Phase 0 complete. Ready for Phase 1: Data Pipeline")


if __name__ == "__main__":
    main()
