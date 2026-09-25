"""Train a TF-IDF baseline: logistic regression (scikit-learn) or an MLP (PyTorch).

Usage: python src/train_baseline.py --config configs/tfidf_logreg.yaml
"""

import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight

from utils import ensure_dirs, experiment_paths, load_config, save_json, set_seed


def load_splits(config):
    processed = load_config(config["data_config"])["data"]["processed_dir"]
    return (
        pd.read_csv(f"{processed}/train.csv", keep_default_na=False),
        pd.read_csv(f"{processed}/val.csv", keep_default_na=False),
    )


def build_vectorizer(config):
    feats = config["features"]
    return TfidfVectorizer(
        max_features=feats["max_features"],
        ngram_range=(1, feats["ngram_max"]),
        min_df=feats["min_df"],
        sublinear_tf=True,
        lowercase=True,
    )


def train_logreg(config, X_train, y_train):
    m = config["model"]
    clf = LogisticRegression(
        C=m["C"],
        max_iter=2000,
        class_weight="balanced" if m["class_weighting"] else None,
    )
    return clf.fit(X_train, y_train)


def train_mlp(config, X_train, y_train, X_val, y_val, model_dir):
    import torch
    import torch.nn as nn

    from models import MLPClassifier

    m = config["model"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLPClassifier(X_train.shape[1], m["hidden_dim"], 3, m["dropout"]).to(device)

    weight = None
    if m["class_weighting"]:
        w = compute_class_weight("balanced", classes=np.arange(3), y=y_train)
        weight = torch.tensor(w, dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=m["learning_rate"])

    def to_tensor(X_sparse):
        # Densify one batch at a time; densifying the whole matrix needs ~2 GB.
        return torch.tensor(X_sparse.toarray(), dtype=torch.float32, device=device)

    def predict(X):
        model.eval()
        preds = []
        with torch.no_grad():
            for start in range(0, X.shape[0], 512):
                preds.append(model(to_tensor(X[start : start + 512])).argmax(1).cpu().numpy())
        return np.concatenate(preds)

    best_f1, bad_epochs, history = -1.0, 0, []
    y_train_t = torch.tensor(y_train, dtype=torch.long, device=device)
    for epoch in range(1, m["epochs"] + 1):
        model.train()
        order = np.random.permutation(X_train.shape[0])
        total = 0.0
        for start in range(0, len(order), m["batch_size"]):
            idx = order[start : start + m["batch_size"]]
            loss = criterion(model(to_tensor(X_train[idx])), y_train_t[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * len(idx)
        val_f1 = f1_score(y_val, predict(X_val), average="macro")
        history.append({"epoch": epoch, "train_loss": total / len(order), "val_macro_f1": val_f1})
        print(f"epoch {epoch:>2}  train_loss={total / len(order):.4f}  val_macro_f1={val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1, bad_epochs = val_f1, 0
            torch.save(model.state_dict(), model_dir / "mlp.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= m["patience"]:
                print(f"early stopping (best val macro F1 {best_f1:.4f})")
                break
    save_json(history, model_dir / "history.json")
    return best_f1


def main(config_path):
    config = load_config(config_path)
    set_seed(config["seed"])
    paths = experiment_paths(config)
    ensure_dirs([paths["model_dir"]])

    train_df, val_df = load_splits(config)
    vectorizer = build_vectorizer(config)
    X_train = vectorizer.fit_transform(train_df["text"])
    X_val = vectorizer.transform(val_df["text"])
    y_train, y_val = train_df["label"].to_numpy(), val_df["label"].to_numpy()
    joblib.dump(vectorizer, paths["model_dir"] / "vectorizer.joblib")

    model_type = config["model"]["type"]
    if model_type == "logreg":
        clf = train_logreg(config, X_train, y_train)
        joblib.dump(clf, paths["model_dir"] / "logreg.joblib")
        val_f1 = f1_score(y_val, clf.predict(X_val), average="macro")
    elif model_type == "mlp":
        val_f1 = train_mlp(config, X_train, y_train, X_val, y_val, paths["model_dir"])
    else:
        raise ValueError(f"unknown model type: {model_type}")

    print(f"[{config['experiment']}] validation macro F1 = {val_f1:.4f}")
    print(f"saved to {paths['model_dir']}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    main(parser.parse_args().config)
