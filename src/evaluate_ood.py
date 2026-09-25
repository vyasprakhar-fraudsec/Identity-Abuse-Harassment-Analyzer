"""Score trained models on the hand-labelled Wikipedia set and measure label agreement.

    python src/evaluate_ood.py --gold prakhar --models configs/tfidf_logreg.yaml configs/distilroberta.yaml

Writes outputs/<experiment>/ood/summary.json per model and outputs/ood_agreement.json.
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score

from evaluate import summarize
from label_maps import LABEL_TO_ID
from predict import load_predictor, predict_labels
from utils import experiment_paths, load_config, save_json


def agreement(labels_dir, gold):
    """Cohen's kappa between the gold annotator and every other annotator on shared items."""
    gold_df = pd.read_csv(Path(labels_dir) / f"{gold}.csv")
    results = {}
    for f in sorted(Path(labels_dir).glob("*.csv")):
        if f.stem == gold:
            continue
        both = gold_df.merge(pd.read_csv(f), on="rev_id", suffixes=("_a", "_b"))
        both = both[(both["label_a"] != "unclear") & (both["label_b"] != "unclear")]
        if len(both) >= 20:
            results[f.stem] = {
                "n": int(len(both)),
                "cohen_kappa": float(cohen_kappa_score(both["label_a"], both["label_b"])),
                "raw_agreement": float((both["label_a"] == both["label_b"]).mean()),
            }
    return results


def main(args):
    wiki = load_config(args.config)
    labels = pd.read_csv(Path(wiki["labels_dir"]) / f"{args.gold}.csv")
    texts = pd.read_csv(wiki["to_label_path"])[["rev_id", "text"]]
    data = labels.merge(texts, on="rev_id")
    data = data[data["label"].isin(LABEL_TO_ID)].reset_index(drop=True)
    y_true = data["label"].map(LABEL_TO_ID).to_numpy()
    print(f"{len(data)} labelled items: {data['label'].value_counts().to_dict()}")

    for cfg_path in args.models:
        config = load_config(cfg_path)
        y_pred, _ = predict_labels(load_predictor(config), data["text"])
        summary, _ = summarize(y_true, y_pred)
        summary["experiment"] = config["experiment"]
        summary["label_distribution"] = data["label"].value_counts().to_dict()
        summary["by_stratum"] = {}
        for stratum, idx in data.groupby("stratum").groups.items():
            s, _ = summarize(y_true[idx], y_pred[idx])
            summary["by_stratum"][stratum] = {k: s[k] for k in ["n", "accuracy", "macro_f1"]}
        save_json(summary, experiment_paths(config)["ood_dir"] / "summary.json")
        print(f"[{config['experiment']}] Wikipedia macro F1 = {summary['macro_f1']:.4f}")

    agree = agreement(wiki["labels_dir"], args.gold)
    save_json(agree, "outputs/ood_agreement.json")
    for name, a in agree.items():
        print(f"agreement {args.gold} vs {name}: kappa={a['cohen_kappa']:.3f} on {a['n']} items")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", required=True, help="annotator whose labels are the reference")
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--config", default="configs/collect_wiki.yaml")
    main(parser.parse_args())
