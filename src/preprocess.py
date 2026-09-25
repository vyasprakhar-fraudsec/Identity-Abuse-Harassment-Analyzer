"""Turn raw HateXplain into train/val/test CSVs.

Decisions (following the HateXplain paper, Mathew et al. 2021):
- Label = majority vote of the 3 annotators. Posts where all 3 disagree
  ("undecided", ~900 posts) are dropped.
- Target groups = communities named by at least 2 of the 3 annotators.
  A post can target several groups; they are stored as "A|B".
- The official split in post_id_divisions.json is used, so results are
  comparable with published numbers.
"""

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd

from label_maps import LABEL_TO_ID
from utils import clean_text, ensure_dirs, load_config, load_json

NO_TARGET = {"None", "none", ""}


def majority_label(annotators):
    """Return the label chosen by >= 2 annotators, or None if undecided."""
    labels = [a["label"] for a in annotators]
    if not labels:
        return None
    label, votes = Counter(labels).most_common(1)[0]
    return label if votes >= 2 else None


def majority_targets(annotators, min_votes=2):
    """Return sorted target groups named by at least `min_votes` annotators."""
    counts = Counter()
    for a in annotators:
        for group in set(a.get("target") or []):
            if group not in NO_TARGET:
                counts[group] += 1
    return sorted(g for g, c in counts.items() if c >= min_votes)


def build_frames(dataset, divisions, config):
    pre = config["preprocessing"]
    frames = {}
    stats = Counter()
    for split, post_ids in divisions.items():
        rows = []
        for post_id in post_ids:
            post = dataset[post_id]
            label = majority_label(post["annotators"])
            if label is None:
                stats["dropped_undecided"] += 1
                continue
            text = clean_text(
                " ".join(post["post_tokens"]),
                lowercase=pre["lowercase"],
                remove_urls=pre["remove_urls"],
                remove_extra_whitespace=pre["remove_extra_whitespace"],
            )
            if len(text.split()) < pre["min_tokens"]:
                stats["dropped_short"] += 1
                continue
            rows.append(
                {
                    "post_id": post_id,
                    "text": text,
                    "label_text": label,
                    "label": LABEL_TO_ID[label],
                    "target_groups": "|".join(majority_targets(post["annotators"])),
                }
            )
        frames[split] = pd.DataFrame(rows)
    return frames, stats


def main(config_path):
    config = load_config(config_path)
    raw_dir = Path(config["data"]["raw_dir"])
    dataset = load_json(raw_dir / "dataset.json")
    divisions = load_json(raw_dir / "post_id_divisions.json")

    frames, stats = build_frames(dataset, divisions, config)

    out_dir = Path(config["data"]["processed_dir"])
    ensure_dirs([out_dir])
    for split, df in frames.items():
        df.to_csv(out_dir / f"{split}.csv", index=False)
        dist = df["label_text"].value_counts().to_dict()
        print(f"{split:>5}: {len(df):>6} posts  {dist}")
    print(f"dropped: {dict(stats)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data.yaml")
    main(parser.parse_args().config)
