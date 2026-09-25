"""Fine-tune a pretrained transformer (default: distilroberta-base) on HateXplain.

Usage: python src/train_transformer.py --config configs/distilroberta.yaml
Keeps the checkpoint with the best validation macro F1 in models/<experiment>/.
"""

import argparse

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from label_maps import ID_TO_LABEL, LABEL_TO_ID
from utils import ensure_dirs, experiment_paths, load_config, save_json, set_seed


def make_loader(df, tokenizer, max_length, batch_size, shuffle):
    records = list(zip(df["text"].tolist(), df["label"].tolist()))

    def collate(batch):
        texts, labels = zip(*batch)
        enc = tokenizer(
            list(texts), truncation=True, max_length=max_length, padding=True, return_tensors="pt"
        )
        enc["labels"] = torch.tensor(labels, dtype=torch.long)
        return enc

    return DataLoader(records, batch_size=batch_size, shuffle=shuffle, collate_fn=collate)


def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            y = batch.pop("labels")
            logits = model(**batch).logits
            preds.append(logits.argmax(1).cpu().numpy())
            labels.append(y.cpu().numpy())
    return f1_score(np.concatenate(labels), np.concatenate(preds), average="macro")


def main(config_path):
    config = load_config(config_path)
    m = config["model"]
    set_seed(config["seed"])
    paths = experiment_paths(config)
    ensure_dirs([paths["model_dir"]])

    processed = load_config(config["data_config"])["data"]["processed_dir"]
    train_df = pd.read_csv(f"{processed}/train.csv", keep_default_na=False)
    val_df = pd.read_csv(f"{processed}/val.csv", keep_default_na=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: no GPU found, this will be slow. Use Colab with a T4 runtime.")

    tokenizer = AutoTokenizer.from_pretrained(m["pretrained_name"])
    model = AutoModelForSequenceClassification.from_pretrained(
        m["pretrained_name"], num_labels=3, id2label=ID_TO_LABEL, label2id=LABEL_TO_ID
    ).to(device)

    train_loader = make_loader(train_df, tokenizer, m["max_length"], m["batch_size"], True)
    val_loader = make_loader(val_df, tokenizer, m["max_length"], m["batch_size"] * 2, False)

    weight = None
    if m["class_weighting"]:
        w = compute_class_weight("balanced", classes=np.arange(3), y=train_df["label"].to_numpy())
        weight = torch.tensor(w, dtype=torch.float32, device=device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=m["learning_rate"], weight_decay=m["weight_decay"]
    )
    total_steps = len(train_loader) * m["epochs"]
    scheduler = get_linear_schedule_with_warmup(
        optimizer, int(m["warmup_ratio"] * total_steps), total_steps
    )
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best_f1, history = -1.0, []
    for epoch in range(1, m["epochs"] + 1):
        model.train()
        running = 0.0
        for step, batch in enumerate(train_loader, 1):
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                loss = loss_fn(model(**batch).logits.float(), labels)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += loss.item()
            if step % 100 == 0:
                print(f"epoch {epoch} step {step}/{len(train_loader)} loss={running / step:.4f}")

        val_f1 = evaluate(model, val_loader, device)
        history.append({"epoch": epoch, "train_loss": running / len(train_loader), "val_macro_f1": val_f1})
        print(f"epoch {epoch}: val macro F1 = {val_f1:.4f}")
        if val_f1 > best_f1:
            best_f1 = val_f1
            model.save_pretrained(paths["model_dir"])
            tokenizer.save_pretrained(paths["model_dir"])

    save_json(history, paths["model_dir"] / "history.json")
    print(f"[{config['experiment']}] best validation macro F1 = {best_f1:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    main(parser.parse_args().config)
