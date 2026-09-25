"""Label the sampled Wikipedia comments in the terminal (works in Colab too).

    python src/label_wiki.py --annotator prakhar
    python src/label_wiki.py --annotator friend --limit 150   # second annotator for agreement

Each annotator gets their own file in data/external/labels/<annotator>.csv holding
rev ids and labels only (no text), so it is safe to commit. Progress is saved
after every item; re-running continues where you stopped.
Guidelines: docs/LABELING_GUIDELINES.md
"""

import argparse
import textwrap
from pathlib import Path

import pandas as pd

from utils import load_config

KEYS = {"1": "normal", "2": "offensive", "3": "hatespeech", "u": "unclear"}


def main(args):
    config = load_config(args.config)
    items = pd.read_csv(config["to_label_path"])
    out = Path(config["labels_dir"]) / f"{args.annotator}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    cols = ["rev_id", "old_rev_id", "stratum", "label", "annotator"]
    done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=cols)
    todo = items[~items["rev_id"].isin(done["rev_id"])]
    if args.limit:
        todo = todo.head(max(0, args.limit - len(done)))

    print(f"{len(done)} labelled, {len(todo)} to go.  1=normal 2=offensive 3=hate u=unclear q=quit")
    for n, row in enumerate(todo.itertuples(), 1):
        print("\n" + "-" * 70 + f"\n[{n}/{len(todo)}]\n" + textwrap.fill(row.text, 70))
        key = ""
        while key not in KEYS and key != "q":
            key = input("> ").strip().lower()
        if key == "q":
            break
        new = {
            "rev_id": row.rev_id,
            "old_rev_id": row.old_rev_id,
            "stratum": row.stratum,
            "label": KEYS[key],
            "annotator": args.annotator,
        }
        done = pd.concat([done, pd.DataFrame([new])], ignore_index=True)
        done.to_csv(out, index=False)
    print(f"\nsaved {len(done)} labels -> {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotator", required=True)
    parser.add_argument("--limit", type=int, default=0, help="stop after this many labels in total")
    parser.add_argument("--config", default="configs/collect_wiki.yaml")
    main(parser.parse_args())
