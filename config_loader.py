from os import PathLike

import yaml

def load_config_yaml(path: str | PathLike) -> dict:
    with open(path, "r", encoding='utf-8') as f:
        config = yaml.safe_load(f) or {}

    return config