from pathlib import Path

from datasets import load_dataset

from utils import ensure_dirs


def main():
    # Create directories for raw data storage
    ensure_dirs(["data/raw"])

    # Download the HateXplain dataset from Hugging Face datasets
    dataset = load_dataset("Hate-speech-CNERG/hatexplain")

    # Save each split to raw JSON files for reproducibility and inspection
    for split_name in dataset.keys():
        output_path = Path("data/raw") / f"hatexplain_{split_name}.json"
        dataset[split_name].to_json(str(output_path))
        print(f"Saved {split_name} split to {output_path}")


if __name__ == "__main__":
    main()