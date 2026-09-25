import pandas as pd

from make_report import bias_comparison, findings, main_table


def run(f1, cm, fpr, recall):
    sub = pd.DataFrame(
        {
            "target_group": ["A", "B"],
            "n": [100, 80],
            "macro_f1": [0.5, 0.6],
            "abusive_recall": [0.8, 0.8],
            "hate_recall": recall,
            "n_normal": [20, 15],
            "normal_fpr": fpr,
        }
    )
    per_class = {k: {"f1-score": 0.5} for k in ["normal", "offensive", "hatespeech"]}
    summary = {"macro_f1": f1, "accuracy": f1, "n": 10, "per_class": per_class, "confusion_matrix": cm}
    return {"summary": summary, "subgroups": sub, "ood": None}


CM = [[10, 7, 1], [2, 10, 5], [0, 3, 10]]
RUNS = {
    "tfidf_logreg": run(0.60, CM, [0.2, 0.3], [0.5, 0.5]),
    "distilroberta": run(0.70, CM, [0.4, 0.1], [0.7, 0.6]),
}


def test_only_best_model_is_bold():
    table = main_table(RUNS)
    assert table.count("**") == 2 and "**0.700**" in table


def test_findings_report_largest_actual_errors():
    text = findings(RUNS)
    assert "7 normal posts predicted as offensive" in text
    assert "5 offensive posts predicted as hate-speech" in text


def test_bias_comparison_counts_groups():
    summary, table = bias_comparison(RUNS)
    assert "more hate speech in 2 of 2 groups" in summary
    assert "more often in 1 of 2 groups" in summary
    assert table.count("\n") == 3
