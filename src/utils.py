"""Shared helpers: config loading, seeding, paths and text cleaning."""

import json
import random
import re
from pathlib import Path

import numpy as np
import yaml


def load_config(config_path):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dirs(paths):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
    except ImportError:  # torch is optional for the scikit-learn baseline
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def experiment_paths(config):
    """Every experiment writes to models/<name>/ and outputs/<name>/."""
    name = config["experiment"]
    out = Path("outputs") / name
    return {
        "model_dir": Path("models") / name,
        "metrics_dir": out / "metrics",
        "figures_dir": out / "figures",
        "predictions_dir": out / "predictions",
        "ood_dir": out / "ood",
    }


def clean_text(text, lowercase=False, remove_urls=True, remove_extra_whitespace=True):
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
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
