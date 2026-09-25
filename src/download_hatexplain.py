"""Download HateXplain from the authors' GitHub release.

We use the original release (not the HuggingFace loader script, which newer
versions of `datasets` no longer run) and pin it to a commit so results are
reproducible. The release includes the official train/val/test split.
"""

import hashlib
import urllib.request
from pathlib import Path

PINNED_COMMIT = "01d742279dac941981f53806154481c0e15ee686"
BASE_URL = f"https://raw.githubusercontent.com/hate-alert/HateXplain/{PINNED_COMMIT}/Data/"
FILES = ["dataset.json", "post_id_divisions.json"]
RAW_DIR = Path("data/raw")


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        dest = RAW_DIR / name
        if dest.exists():
            print(f"{dest} already exists, skipping")
        else:
            print(f"Downloading {name} ...")
            urllib.request.urlretrieve(BASE_URL + name, dest)
        print(f"{dest}  sha256={sha256(dest)[:16]}")


if __name__ == "__main__":
    main()
