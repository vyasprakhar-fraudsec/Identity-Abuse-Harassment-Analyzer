import argparse
import json
import pickle

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

from train_baseline import MLPClassifier
from utils import ensure_dirs, load_config


def evaluate_subgroups(df, y_true, y_pred, id_to_label):
    """
    Compute simple subgroup metrics by target_group.
    Only reports groups with enough examples to be meaningful.
    """
    subgroup_rows = []
    eval_df = df.copy()
    eval_df["y_true"] = y_true
    eval_df["y_pred"] = y_pred

    for group, group_df in eval_df.groupby("target_group"):
        if len(group_df) < 20:
            continue

        precision, recall, f1, _ = precision_recall_fscore_support(
            group_df["y_true"],
            group_df["y_pred"],
            average="macro",
            zero_division=0,
        )

        subgroup_rows.append(
            {
                "target_group": group,
                "n_samples": len(group_df),
                "macro_precision": precision,
                "macro_recall": recall,
                "macro_f1": f1,
            }
        )

    return pd.DataFrame(subgroup_rows).sort_values("macro_f1")


def main(config_path):
    config = load_config(config_path)

    ensure_dirs(
        [
            config["paths"]["metrics_dir"],
            config["paths"]["figures_dir"],
            config["paths"]["predictions_dir"],
        ]
    )

    test_df = pd.read_csv(f"{config['data']['processed_dir']}/test.csv")

    with open(config["paths"]["vectorizer_path"], "rb") as f:
        vectorizer = pickle.load(f)

    with open(config["paths"]["label_map_path"], "r", encoding="utf-8") as f:
        label_map = json.load(f)

    id_to_label = {int(v): k for k, v in label_map.items()}

    X_test = vectorizer.transform(test_df["text"])
    y_test = test_df["label"].values

    input_dim = X_test.shape[1]
    output_dim = len(label_map)

    model = MLPClassifier(
        input_dim=input_dim,
        hidden_dim=config["training"]["hidden_dim"],
        output_dim=output_dim,
        dropout=config["training"]["dropout"],
    )

    model.load_state_dict(torch.load(config["paths"]["model_path"], map_location="cpu"))
    model.eval()

    with torch.no_grad():
        X_tensor = torch.tensor(X_test.toarray(), dtype=torch.float32)
        logits = model(X_tensor)
        y_pred = torch.argmax(logits, dim=1).numpy()

    report = classification_report(
        y_test,
        y_pred,
        target_names=[id_to_label[i] for i in sorted(id_to_label.keys())],
        zero_division=0,
        output_dict=True,
    )

    report_df = pd.DataFrame(report).transpose()
    report_df.to_csv(f"{config['paths']['metrics_dir']}/classification_report.csv")

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[id_to_label[i] for i in sorted(id_to_label.keys())],
        yticklabels=[id_to_label[i] for i in sorted(id_to_label.keys())],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(f"{config['paths']['figures_dir']}/confusion_matrix.png", dpi=200)
    plt.close()

    predictions_df = test_df.copy()
    predictions_df["pred_label"] = y_pred
    predictions_df["pred_label_text"] = predictions_df["pred_label"].map(id_to_label)
    predictions_df.to_csv(f"{config['paths']['predictions_dir']}/test_predictions.csv", index=False)

    subgroup_df = evaluate_subgroups(test_df, y_test, y_pred, id_to_label)
    subgroup_df.to_csv(f"{config['paths']['metrics_dir']}/subgroup_metrics.csv", index=False)

    print("Saved classification report, subgroup metrics, confusion matrix, and predictions.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)