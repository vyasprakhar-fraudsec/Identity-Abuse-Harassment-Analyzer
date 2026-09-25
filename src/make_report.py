"""Generate reports/RESULTS.md and the README results section from real outputs.

Nothing in the results is typed by hand: this script reads outputs/*/metrics and
outputs/*/ood, copies the key files into reports/ (which is committed), and writes
the tables. Run it after training/evaluating: `make report`.
"""

import json
import shutil
from pathlib import Path

import pandas as pd

ORDER = ["tfidf_logreg", "mlp_base", "mlp_tuned", "distilroberta"]
NAMES = {
    "tfidf_logreg": "TF-IDF + Logistic Regression",
    "mlp_base": "TF-IDF + MLP (untuned)",
    "mlp_tuned": "TF-IDF + MLP (dropout 0.3, class weights)",
    "distilroberta": "DistilRoBERTa (fine-tuned)",
}
LABEL_NAMES = ["normal", "offensive", "hate-speech"]
START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
MIN_NORMAL_FOR_FPR = 10


def fmt(x, digits=3):
    return "–" if x is None or pd.isna(x) else f"{x:.{digits}f}"


def load_runs(outputs):
    runs = {}
    for d in sorted(Path(outputs).iterdir()) if Path(outputs).exists() else []:
        summary = d / "metrics" / "summary.json"
        if not summary.exists():
            continue
        run = {"summary": json.loads(summary.read_text())}
        sub = d / "metrics" / "subgroup_metrics.csv"
        run["subgroups"] = pd.read_csv(sub) if sub.exists() else None
        ood = d / "ood" / "summary.json"
        run["ood"] = json.loads(ood.read_text()) if ood.exists() else None
        run["dir"] = d
        runs[d.name] = run
    return dict(sorted(runs.items(), key=lambda kv: ORDER.index(kv[0]) if kv[0] in ORDER else 99))


def name(exp):
    return NAMES.get(exp, exp)


def main_table(runs):
    lines = [
        "| Model | Macro F1 | Accuracy | Normal F1 | Offensive F1 | Hate F1 |",
        "|---|---|---|---|---|---|",
    ]
    best = best_run(runs)
    for exp, r in runs.items():
        s, pc = r["summary"], r["summary"]["per_class"]
        f1 = f"**{fmt(s['macro_f1'])}**" if exp == best else fmt(s["macro_f1"])
        lines.append(
            f"| {name(exp)} | {f1} | {fmt(s['accuracy'])} | "
            f"{fmt(pc['normal']['f1-score'])} | {fmt(pc['offensive']['f1-score'])} | "
            f"{fmt(pc['hatespeech']['f1-score'])} |"
        )
    return "\n".join(lines)


def subgroup_table(df):
    lines = [
        "| Target group | n | Macro F1 | Hate recall | Abusive recall | Normal posts "
        "| False-flag rate on normal |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in df.itertuples():
        fpr = r.normal_fpr if r.n_normal >= MIN_NORMAL_FOR_FPR else None
        lines.append(
            f"| {r.target_group} | {r.n} | {fmt(r.macro_f1)} | {fmt(r.hate_recall)} | "
            f"{fmt(r.abusive_recall)} | {r.n_normal} | {fmt(fpr)} |"
        )
    return "\n".join(lines)


def ood_table(runs):
    rows = [(exp, r) for exp, r in runs.items() if r["ood"]]
    if not rows:
        return None
    lines = [
        "| Model | HateXplain test | Wikipedia (all) | Wikipedia (random half) | Change |",
        "|---|---|---|---|---|",
    ]
    for exp, r in rows:
        in_f1, ood = r["summary"]["macro_f1"], r["ood"]
        rand = ood.get("by_stratum", {}).get("random", {}).get("macro_f1")
        lines.append(
            f"| {name(exp)} | {fmt(in_f1)} | {fmt(ood['macro_f1'])} | {fmt(rand)} | "
            f"{ood['macro_f1'] - in_f1:+.3f} |"
        )
    return "\n".join(lines)


def best_run(runs):
    return max(runs, key=lambda e: runs[e]["summary"]["macro_f1"])


def findings(runs):
    out = []
    best = best_run(runs)
    s = runs[best]["summary"]
    out.append(f"Best model on the HateXplain test set: **{name(best)}**, macro F1 {fmt(s['macro_f1'])}.")
    if "tfidf_logreg" in runs and best != "tfidf_logreg":
        gain = s["macro_f1"] - runs["tfidf_logreg"]["summary"]["macro_f1"]
        out.append(f"That is {gain:+.3f} macro F1 over the logistic-regression baseline.")
    cm = s.get("confusion_matrix")
    if cm:
        errors = sorted(
            ((cm[t][p], LABEL_NAMES[t], LABEL_NAMES[p]) for t in range(3) for p in range(3) if t != p),
            reverse=True,
        )
        (n1, t1, p1), (n2, t2, p2) = errors[:2]
        out.append(
            f"Most common errors: {n1} {t1} posts predicted as {p1}, and {n2} {t2} posts predicted as {p2}."
        )
    sub = runs[best]["subgroups"]
    if sub is not None and len(sub):
        big = sub[sub["n"] >= 50]
        if len(big) >= 2:
            lo, hi = big.loc[big["macro_f1"].idxmin()], big.loc[big["macro_f1"].idxmax()]
            out.append(
                f"Among groups with at least 50 test posts, macro F1 ranges from {fmt(lo.macro_f1)} "
                f"({lo.target_group}) to {fmt(hi.macro_f1)} ({hi.target_group})."
            )
        fpr = sub[sub["n_normal"] >= MIN_NORMAL_FOR_FPR].dropna(subset=["normal_fpr"])
        if len(fpr):
            w = fpr.loc[fpr["normal_fpr"].idxmax()]
            out.append(
                f"Highest false-flag rate: {fmt(w.normal_fpr * 100, 0)}% of normal posts about the "
                f"**{w.target_group}** group were flagged as abusive (only {w.n_normal} such posts, "
                "so treat as indicative). This is identity-term bias: the model learns the group "
                "name itself as a signal of abuse."
            )
    bias = bias_comparison(runs)
    if bias:
        out.append(
            f"**Accuracy vs. over-flagging:** {bias[0]} Group counts are small, so read this as a signal."
        )
    return "\n".join(f"- {x}" for x in out)


def bias_comparison(runs, baseline="tfidf_logreg"):
    """False-flag rate on normal posts per group: baseline vs best model."""
    best = best_run(runs)
    if best == baseline or baseline not in runs:
        return None
    a, b = runs[baseline]["subgroups"], runs[best]["subgroups"]
    if a is None or b is None:
        return None
    m = a.merge(b, on="target_group", suffixes=("_base", "_best"))
    m = m[m["n_normal_best"] >= MIN_NORMAL_FOR_FPR].sort_values("n_best", ascending=False)
    if m.empty:
        return None
    rows = [
        f"| Target group | Normal posts | False-flag rate: {name(baseline)} | False-flag rate: {name(best)} "
        f"| Hate recall: {name(baseline)} | Hate recall: {name(best)} |",
        "|---|---|---|---|---|---|",
    ]
    for r in m.itertuples():
        rows.append(
            f"| {r.target_group} | {r.n_normal_best} | {fmt(r.normal_fpr_base)} | {fmt(r.normal_fpr_best)} "
            f"| {fmt(r.hate_recall_base)} | {fmt(r.hate_recall_best)} |"
        )
    worse = int((m["normal_fpr_best"] > m["normal_fpr_base"]).sum())
    better_recall = int((m["hate_recall_best"] > m["hate_recall_base"]).sum())
    summary = (
        f"The more accurate model catches more hate speech in {better_recall} of {len(m)} groups, "
        f"but wrongly flags harmless posts more often in {worse} of {len(m)} groups."
    )
    return summary, "\n".join(rows)


def copy_artifacts(runs, reports):
    (reports / "figures").mkdir(parents=True, exist_ok=True)
    (reports / "metrics").mkdir(parents=True, exist_ok=True)
    for exp, r in runs.items():
        d = r["dir"]
        fig = d / "figures" / "confusion_matrix.png"
        if fig.exists():
            shutil.copy(fig, reports / "figures" / f"{exp}_confusion_matrix.png")
        for src, dst in [
            (d / "metrics" / "summary.json", f"{exp}_summary.json"),
            (d / "metrics" / "subgroup_metrics.csv", f"{exp}_subgroups.csv"),
            (d / "ood" / "summary.json", f"{exp}_wikipedia_summary.json"),
        ]:
            if src.exists():
                shutil.copy(src, reports / "metrics" / dst)


def build(outputs="outputs", reports="reports", readme="README.md"):
    runs = load_runs(outputs)
    if not runs:
        raise SystemExit("no evaluated runs in outputs/, run `make baseline` first")
    reports = Path(reports)
    copy_artifacts(runs, reports)
    best = best_run(runs)
    agreement_path = Path(outputs) / "ood_agreement.json"
    agreement = json.loads(agreement_path.read_text()) if agreement_path.exists() else {}

    n_test = runs[best]["summary"]["n"]
    ood = ood_table(runs)
    parts = [
        "# Results",
        "",
        "> Generated by `src/make_report.py` from the files in `reports/metrics/`. "
        "Do not edit by hand; rerun `make report`.",
        "",
        f"All numbers are on the **official HateXplain test split** ({n_test} posts, "
        "majority-vote labels, posts without a majority removed).",
        "",
        "## 1. Model comparison",
        "",
        main_table(runs),
        "",
        "## 2. Key findings",
        "",
        findings(runs),
        "",
        f"## 3. Confusion matrix: {name(best)}",
        "",
        f"![confusion matrix](figures/{best}_confusion_matrix.png)",
        "",
    ]
    if runs[best]["subgroups"] is not None:
        parts += [
            f"## 4. Fairness by target group: {name(best)}",
            "",
            "A post counts towards every group that at least 2 of 3 annotators said it targets. "
            f"False-flag rate is shown only when a group has at least {MIN_NORMAL_FOR_FPR} normal posts.",
            "",
            subgroup_table(runs[best]["subgroups"]),
            "",
        ]
        bias = bias_comparison(runs)
        if bias:
            parts += [
                "### Does the better model over-flag more?",
                "",
                bias[0]
                + " Over-flagging harmless posts about a group silences that group, so an accuracy gain "
                "can come with a fairness cost. Counts are small (see the Normal posts column).",
                "",
                bias[1],
                "",
            ]
    parts += ["## 5. Fresh data: Wikipedia talk pages", ""]
    if ood:
        first = next(r["ood"] for r in runs.values() if r["ood"])
        parts += [
            f"{first['n']} recent comments, hand-labelled (distribution: {first['label_distribution']}). "
            "See [the datasheet](../docs/DATASHEET_wiki_talk.md). The random half is the unbiased "
            "estimate; the other half was chosen because the baseline flagged it.",
            "",
            ood,
            "",
        ]
        for other, a in agreement.items():
            parts.append(
                f"Label agreement with a second annotator ({a['n']} items): Cohen's kappa "
                f"{a['cohen_kappa']:.2f}, raw agreement {a['raw_agreement']:.0%}."
            )
    else:
        parts.append(
            "The collection, sampling, labelling and scoring tools are built and tested "
            "(`src/collect_wiki.py`, `src/label_wiki.py`, `src/evaluate_ood.py`), but no labelled set "
            "has been published yet. Running notebook steps 5–7 fills this section in automatically."
        )
    parts += [
        "",
        "## 6. Reproduce",
        "",
        "```bash",
        "make setup && make data",
        "make baseline      # TF-IDF models, CPU, a few minutes",
        "make transformer   # needs a GPU (Colab T4 ~10 min)",
        "make report",
        "```",
        "",
    ]
    (reports / "RESULTS.md").write_text("\n".join(parts), encoding="utf-8")

    block = [START, "", main_table(runs), ""]
    pending = [n for e, n in NAMES.items() if e not in runs]
    if pending:
        block += [f"_Not run yet: {', '.join(pending)}. See `notebooks/run_pipeline.ipynb`._", ""]
    if ood:
        block += ["**On fresh data (Wikipedia talk pages, hand-labelled):**", "", ood, ""]
    block += [findings(runs), "", "Full report: [`reports/RESULTS.md`](reports/RESULTS.md)", "", END]
    readme_path = Path(readme)
    text = readme_path.read_text(encoding="utf-8")
    if START in text and END in text:
        head, rest = text.split(START, 1)
        text = head + "\n".join(block) + rest.split(END, 1)[1]
        readme_path.write_text(text, encoding="utf-8")
    print(f"wrote {reports / 'RESULTS.md'} and updated {readme_path}")


if __name__ == "__main__":
    build()
