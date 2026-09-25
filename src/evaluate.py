"""Evaluate a trained model on the official HateXplain test split.

Writes to outputs/<experiment>/:
  metrics/summary.json            headline numbers
  metrics/classification_report.csv
  metrics/subgroup_metrics.csv    per target group fairness metrics
  figures/confusion_matrix.png
  predictions/test_predictions.csv
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from label_maps import ABUSIVE_IDS, LABEL_TO_ID, LABELS  # noqa: E402
from predict import load_predictor, predict_labels  # noqa: E402
from utils import ensure_dirs, experiment_paths, load_config, save_json  # noqa: E402


def summarize(y_true, y_pred):
    report = classification_report(
        y_true, y_pred, labels=[0, 1, 2], target_names=LABELS, output_dict=True, zero_division=0
    )
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro")),
        "per_class": {
            label: {k: float(report[label][k]) for k in ["precision", "recall", "f1-score"]}
            for label in LABELS
        },
    }, report


def subgroup_metrics(df, y_true, y_pred, min_group_size=30):
    """Metrics for each target group a post belongs to.

    macro_f1        3-class macro F1 within the group
    abusive_recall  share of truly abusive posts (offensive or hate) that were flagged
    hate_recall     share of true hate speech predicted as hate speech
    normal_fpr      share of normal posts about this group wrongly flagged as abusive
                    (few posts are "normal" AND have a target, so n_normal is reported)
    """
    d = df[["target_groups"]].copy()
    d["y_true"], d["y_pred"] = y_true, y_pred
    d["group"] = d["target_groups"].fillna("").str.split("|")
    d = d.explode("group")
    d = d[d["group"] != ""]

    abusive = list(ABUSIVE_IDS)
    hate = LABEL_TO_ID["hatespeech"]
    rows = []
    for group, g in d.groupby("group"):
        if len(g) < min_group_size:
            continue
        is_ab, pred_ab = g["y_true"].isin(abusive), g["y_pred"].isin(abusive)
        is_hate, is_normal = g["y_true"] == hate, ~is_ab
        rows.append(
            {
                "target_group": group,
                "n": len(g),
                "macro_f1": f1_score(g["y_true"], g["y_pred"], labels=[0, 1, 2], average="macro"),
                "abusive_recall": pred_ab[is_ab].mean() if is_ab.any() else np.nan,
                "hate_recall": (g["y_pred"][is_hate] == hate).mean() if is_hate.any() else np.nan,
                "n_normal": int(is_normal.sum()),
                "normal_fpr": pred_ab[is_normal].mean() if is_normal.any() else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("n", ascending=False).reset_index(drop=True)


def plot_confusion(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    plt.figure(figsize=(5.5, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
    return cm


def main(config_path):
    config = load_config(config_path)
    paths = experiment_paths(config)
    ensure_dirs([paths["metrics_dir"], paths["figures_dir"], paths["predictions_dir"]])

    processed = load_config(config["data_config"])["data"]["processed_dir"]
    test_df = pd.read_csv(f"{processed}/test.csv", keep_default_na=False)
    predictor = load_predictor(config)
    y_pred, probs = predict_labels(predictor, test_df["text"])
    y_true = test_df["label"].to_numpy()

    summary, report = summarize(y_true, y_pred)
    summary.update({"experiment": config["experiment"], "model_type": config["model"]["type"]})
    cm = plot_confusion(
        y_true, y_pred, f"{config['experiment']} (test)", paths["figures_dir"] / "confusion_matrix.png"
    )
    summary["confusion_matrix"] = cm.tolist()
    save_json(summary, paths["metrics_dir"] / "summary.json")
    pd.DataFrame(report).transpose().to_csv(paths["metrics_dir"] / "classification_report.csv")

    groups = subgroup_metrics(test_df, y_true, y_pred, config["eval"]["min_group_size"])
    groups.to_csv(paths["metrics_dir"] / "subgroup_metrics.csv", index=False)

    preds = test_df[["post_id", "text", "label_text", "target_groups"]].copy()
    preds["pred_label_text"] = [LABELS[i] for i in y_pred]
    for i, label in enumerate(LABELS):
        preds[f"p_{label}"] = probs[:, i].round(4)
    preds.to_csv(paths["predictions_dir"] / "test_predictions.csv", index=False)

    print(
        f"[{config['experiment']}] test macro F1 = {summary['macro_f1']:.4f}  "
        f"accuracy = {summary['accuracy']:.4f}"
    )
    print(groups.round(3).to_string(index=False))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    main(parser.parse_args().config)
