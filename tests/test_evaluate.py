import numpy as np
import pandas as pd

from evaluate import subgroup_metrics, summarize


def test_summarize_perfect_predictions():
    y = np.array([0, 1, 2, 2])
    s, _ = summarize(y, y)
    assert s["macro_f1"] == 1.0 and s["n"] == 4


def test_subgroup_metrics_counts_multi_target_posts_in_each_group():
    df = pd.DataFrame({"target_groups": ["A|B", "A", "", "B"]})
    y_true = np.array([2, 0, 0, 1])
    y_pred = np.array([2, 1, 0, 1])
    out = subgroup_metrics(df, y_true, y_pred, min_group_size=1).set_index("target_group")
    assert out.loc["A", "n"] == 2 and out.loc["B", "n"] == 2
    assert "" not in out.index
    assert out.loc["A", "normal_fpr"] == 1.0  # the one normal post about A was flagged
    assert out.loc["B", "abusive_recall"] == 1.0
