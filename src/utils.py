import json
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(config_path: str):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def clean_text(text: str, lowercase=True, remove_urls=True, remove_extra_whitespace=True):
    if not isinstance(text, str):
        text = ""

    if lowercase:
        text = text.lower()

    if remove_urls:
        text = re.sub(r"http\S+|www\.\S+", " ", text)

    if remove_extra_whitespace:
        text = re.sub(r"\s+", " ", text).strip()

    return text


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)