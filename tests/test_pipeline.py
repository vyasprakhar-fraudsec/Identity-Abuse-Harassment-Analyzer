"""End-to-end smoke test on a tiny synthetic dataset (scikit-learn path only, no torch)."""

import json

import pandas as pd
import yaml

import make_report
from evaluate import main as evaluate_main
from predict import load_predictor
from train_baseline import main as train_main

TEXTS = {
    0: ["have a lovely day", "thanks for the help", "great article today", "nice work on this"],
    1: ["you are an idiot", "shut up you fool", "what a stupid idiot", "you fool stop it"],
    2: ["group x are vermin", "remove group x now", "group x are subhuman", "vermin group x out"],
}


def make_project(tmp_path):
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    rows = [
        {
            "post_id": f"{label}-{i}-{rep}",
            "text": t,
            "label": label,
            "label_text": ["normal", "offensive", "hatespeech"][label],
            "target_groups": "X" if label == 2 else "",
        }
        for label, texts in TEXTS.items()
        for i, t in enumerate(texts)
        for rep in range(3)
    ]
    for split in ["train", "val", "test"]:
        pd.DataFrame(rows).to_csv(processed / f"{split}.csv", index=False)
    (tmp_path / "configs").mkdir()
    yaml.safe_dump({"data": {"processed_dir": "data/processed"}}, open(tmp_path / "configs/data.yaml", "w"))
    config = {
        "experiment": "tfidf_logreg",
        "seed": 0,
        "data_config": "configs/data.yaml",
        "features": {"max_features": 500, "ngram_max": 2, "min_df": 1},
        "model": {"type": "logreg", "C": 10.0, "class_weighting": True},
        "eval": {"min_group_size": 1},
    }
    yaml.safe_dump(config, open(tmp_path / "configs/tfidf_logreg.yaml", "w"))
    (tmp_path / "README.md").write_text("intro\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\nend\n")


def test_train_evaluate_report(tmp_path, monkeypatch):
    make_project(tmp_path)
    monkeypatch.chdir(tmp_path)
    train_main("configs/tfidf_logreg.yaml")
    evaluate_main("configs/tfidf_logreg.yaml")

    summary = json.loads((tmp_path / "outputs/tfidf_logreg/metrics/summary.json").read_text())
    assert summary["macro_f1"] > 0.9

    predictor = load_predictor("configs/tfidf_logreg.yaml")
    probs = predictor.predict_proba(["you are an idiot"])
    assert probs.shape == (1, 3) and probs[0].argmax() == 1
    assert predictor.explain("you are an idiot")[0][0] in {"idiot", "an idiot", "are an"}

    make_report.build()
    readme = (tmp_path / "README.md").read_text()
    assert "old" not in readme and "TF-IDF + Logistic Regression" in readme and readme.endswith("end\n")
    assert (tmp_path / "reports/RESULTS.md").exists()
    assert (tmp_path / "reports/figures/tfidf_logreg_confusion_matrix.png").exists()
