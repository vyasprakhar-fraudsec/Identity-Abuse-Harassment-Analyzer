import argparse
import json
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, TensorDataset

from utils import ensure_dirs, load_config, set_seed


class MLPClassifier(nn.Module):
    """
    Simple baseline model:
    TF-IDF features -> small hidden layer -> output logits
    """
    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)


def make_loader(X, y, batch_size, shuffle=False):
    X_tensor = torch.tensor(X.toarray(), dtype=torch.float32)
    y_tensor = torch.tensor(y.values, dtype=torch.long)
    dataset = TensorDataset(X_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_epoch(model, loader, criterion, optimizer=None, device="cpu"):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()

    total_loss = 0.0
    total_correct = 0
    total_items = 0

    with torch.set_grad_enabled(is_train):
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            preds = torch.argmax(logits, dim=1)
            total_loss += loss.item() * y_batch.size(0)
            total_correct += (preds == y_batch).sum().item()
            total_items += y_batch.size(0)

    return total_loss / total_items, total_correct / total_items


def main(config_path):
    config = load_config(config_path)
    set_seed(config["data"]["random_seed"])

    ensure_dirs(["models"])

    train_df = pd.read_csv(f"{config['data']['processed_dir']}/train.csv")
    val_df = pd.read_csv(f"{config['data']['processed_dir']}/val.csv")

    vectorizer = TfidfVectorizer(
        max_features=config["training"]["max_features"],
        ngram_range=(1, 2),
        min_df=2,
    )

    X_train = vectorizer.fit_transform(train_df["text"])
    X_val = vectorizer.transform(val_df["text"])

    y_train = train_df["label"]
    y_val = val_df["label"]

    train_loader = make_loader(X_train, y_train, config["training"]["batch_size"], shuffle=True)
    val_loader = make_loader(X_val, y_val, config["training"]["batch_size"], shuffle=False)

    input_dim = X_train.shape[1]
    output_dim = len(sorted(train_df["label"].unique()))
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = MLPClassifier(
        input_dim=input_dim,
        hidden_dim=config["training"]["hidden_dim"],
        output_dim=output_dim,
        dropout=config["training"]["dropout"],
    ).to(device)

    if config["training"]["class_weighting"]:
        classes = np.array(sorted(y_train.unique()))
        weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train.values)
        class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])

    best_val_loss = float("inf")

    for epoch in range(config["training"]["epochs"]):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None, device=device)

        print(
            f"Epoch {epoch+1}/{config['training']['epochs']} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), config["paths"]["model_path"])

    with open(config["paths"]["vectorizer_path"], "wb") as f:
        pickle.dump(vectorizer, f)

    print(f"Saved best model to {config['paths']['model_path']}")
    print(f"Saved vectorizer to {config['paths']['vectorizer_path']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)