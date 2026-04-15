import argparse
from collections import Counter

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from label_maps import LABEL_TO_ID_3CLASS, LABEL_TO_ID_BINARY
from utils import clean_text, ensure_dirs, load_config, save_json, set_seed


def normalize_majority_label(annotators_dict):
    """
    Extract the majority label from HateXplain's annotator metadata.
    This assumes a list of labels exists under the 'label' key.
    """
    labels = annotators_dict.get("label", [])
    if not labels:
        return None

    counts = Counter(labels)
    majority_label = counts.most_common(1)[0][0]

    # Map common numeric forms used in HateXplain to text labels if needed.
    numeric_map = {0: "hatespeech", 1: "normal", 2: "offensive"}
    if isinstance(majority_label, int):
        majority_label = numeric_map.get(majority_label, None)

    if isinstance(majority_label, str):
        majority_label = majority_label.strip().lower().replace("hate speech", "hatespeech")

    return majority_label


def extract_target_groups(targets):
    """
    Convert target annotations into a readable subgroup string.
    Stores one or more groups joined by '|'.
    """
    if targets is None:
        return "none"

    if isinstance(targets, list):
        flat_targets = []
        for item in targets:
            if isinstance(item, list):
                flat_targets.extend(item)
            else:
                flat_targets.append(item)

        flat_targets = [str(t).strip().lower() for t in flat_targets if str(t).strip()]
        return "|".join(sorted(set(flat_targets))) if flat_targets else "none"

    return str(targets).strip().lower() if str(targets).strip() else "none"


def build_dataframe(dataset_split, config):
    rows = []

    for example in dataset_split:
        text = " ".join(example.get("post_tokens", []))
        text = clean_text(
            text,
            lowercase=config["preprocessing"]["lowercase"],
            remove_urls=config["preprocessing"]["remove_urls"],
            remove_extra_whitespace=config["preprocessing"]["remove_extra_whitespace"],
        )

        if len(text.split()) < config["data"]["min_text_len"]:
            continue

        annotators = example.get("annotators", {})
        raw_label = normalize_majority_label(annotators)

        if raw_label is None:
            continue

        # Binary mode merges offensive + hatespeech into abusive
        if config["data"]["label_mode"] == "binary":
            final_label = "non_abusive" if raw_label == "normal" else "abusive"
        else:
            final_label = raw_label

        target_group = extract_target_groups(example.get("target"))

        rows.append(
            {
                "post_id": example.get("id"),
                "text": text,
                "label_text": final_label,
                "target_group": target_group,
            }
        )

    return pd.DataFrame(rows)


def main(config_path):
    config = load_config(config_path)
    set_seed(config["data"]["random_seed"])

    ensure_dirs([config["data"]["processed_dir"]])

    dataset = load_dataset(config["data"]["dataset_name"])
    all_df = pd.concat(
        [build_dataframe(dataset[split], config) for split in dataset.keys()],
        ignore_index=True,
    ).drop_duplicates(subset=["post_id"])

    if config["data"]["label_mode"] == "binary":
        label_map = LABEL_TO_ID_BINARY
    else:
        label_map = LABEL_TO_ID_3CLASS

    all_df = all_df[all_df["label_text"].isin(label_map.keys())].copy()
    all_df["label"] = all_df["label_text"].map(label_map)

    train_df, temp_df = train_test_split(
        all_df,
        test_size=config["data"]["test_size"] + config["data"]["val_size"],
        random_state=config["data"]["random_seed"],
        stratify=all_df["label"],
    )

    relative_val_size = config["data"]["val_size"] / (
        config["data"]["test_size"] + config["data"]["val_size"]
    )

    val_df, test_df = train_test_split(
        temp_df,
        test_size=1 - relative_val_size,
        random_state=config["data"]["random_seed"],
        stratify=temp_df["label"],
    )

    processed_dir = config["data"]["processed_dir"]
    train_df.to_csv(f"{processed_dir}/train.csv", index=False)
    val_df.to_csv(f"{processed_dir}/val.csv", index=False)
    test_df.to_csv(f"{processed_dir}/test.csv", index=False)

    save_json(label_map, config["paths"]["label_map_path"])

    print("Saved processed splits:")
    print(train_df["label_text"].value_counts())
    print(val_df["label_text"].value_counts())
    print(test_df["label_text"].value_counts())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)