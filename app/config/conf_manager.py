import json
import os

CONFIG_FILE = os.path.join(os.path.dirname(__file__), "settings.conf")


def load_config() -> dict:
    if not os.path.exists(CONFIG_FILE):
        default_config = {"data_file": None, "has_header": None}
        save_config(default_config)
        return default_config
    with open(CONFIG_FILE, "r") as f:
        return json.load(f)


def save_config(config_data: dict) -> None:
    with open(CONFIG_FILE, "w") as f:
        json.dump(config_data, f, indent=4)


def get_value(key: str):
    config_data = load_config()
    return config_data.get(key)


def set_value(key: str, value) -> None:
    config_data = load_config()
    config_data[key] = value
    save_config(config_data)
