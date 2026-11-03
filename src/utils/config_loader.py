"""بارگذاری تنظیمات از YAML و متغیرهای محیطی"""
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
        raise ValueError(f"API key '{key_name}' not found in environment")
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
        raise AttributeError(f"Config has no attribute '{name}'")

    def get(self, key: str, default=None):
        return self._config.get(key, default)
