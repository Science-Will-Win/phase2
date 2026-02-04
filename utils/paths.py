"""
Path configuration module.
Loads paths from config.yaml and provides accessor functions.
"""
import os
import yaml
from typing import Optional

_config: Optional[dict] = None
_environment: str = "server"


def load_config() -> dict:
    """Load config.yaml"""
    global _config
    if _config is None:
        config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            _config = yaml.safe_load(f)
    return _config


def set_environment(env: str):
    """Set environment (local/server)"""
    global _environment
    _environment = env


def set_local_mode(is_local: bool):
    """Handle --local argument"""
    set_environment("local" if is_local else "server")


def is_local_mode() -> bool:
    """Check if running in local mode"""
    return _environment == "local"


def get_path(key: str) -> str:
    """Get path by key"""
    config = load_config()
    return config["paths"][_environment][key]


def get_model_dir() -> str:
    """Get model directory path"""
    return get_path("model")


def get_data_dir() -> str:
    """Get data directory path"""
    return get_path("data")


def get_result_dir() -> str:
    """Get result directory path"""
    return get_path("result")


def get_log_dir() -> str:
    """Get log directory path"""
    return get_path("log")


def get_temp_data_dir() -> str:
    """Get temp_data directory path"""
    return get_path("temp_data")


def ensure_dirs():
    """Create required directories if they don't exist"""
    for key in ["model", "data", "result", "log", "temp_data"]:
        os.makedirs(get_path(key), exist_ok=True)
