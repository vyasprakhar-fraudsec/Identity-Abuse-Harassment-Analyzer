"""Quick EDA on the processed splits: label balance and target-group coverage."""

import argparse
from pathlib import Path

import pandas as pd

from utils import load_config


def main(config_path):
    config = load_config(config_path)
    processed = Path(config["data"]["processed_dir"])
    for split in ["train", "val", "test"]:
        df = pd.read_csv(processed / f"{split}.csv", keep_default_na=False)
        print(f"\n=== {split} ({len(df)} posts) ===")
        print(df["label_text"].value_counts(normalize=True).round(3).to_string())
        groups = df["target_groups"].str.split("|").explode()
        print("top target groups:")
        print(groups[groups != ""].value_counts().head(12).to_string())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    main(parser.parse_args().config)
