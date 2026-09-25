# Architecture & design decisions

## Pipeline

```
data/raw  (HateXplain dataset.json + post_id_divisions.json, pinned commit)
   │  preprocess.py
   ▼
data/processed/{train,val,test}.csv   post_id, text, label, target_groups
   │  train_baseline.py / train_transformer.py        (one YAML config per experiment)
   ▼
models/<experiment>/                  vectorizer + logreg | mlp.pt | HF checkpoint
   │  predict.py  (same predict_proba for every model)
   ├─► evaluate.py      → outputs/<experiment>/metrics, figures, predictions
   ├─► evaluate_ood.py  → outputs/<experiment>/ood     (Wikipedia set)
   └─► app/app.py       → Gradio demo
outputs/ ──► make_report.py ──► reports/ (committed) + README results section
```

## Decisions

**Official split, original release.** The HuggingFace loader for HateXplain is a
script that recent `datasets` versions refuse to run, and re-splitting the data
makes results incomparable with the paper. The authors' GitHub release includes
the official split, so we download it pinned to a commit.

**Majority labels, undecided posts dropped.** Each post has 3 annotators. About
900 posts have three different labels; like the paper, we drop them rather than
pick one arbitrarily.

**Target groups need 2 of 3 annotators.** One annotator's view of the target is
noisy. A post can target several groups and counts towards each in the fairness
metrics.

**Three model tiers.**
- *TF-IDF + logistic regression*: trains in seconds on CPU and is fully
  interpretable (word contribution = TF-IDF value × class weight). It is the
  floor any other model has to beat.
- *TF-IDF + MLP*: tests whether a non-linear layer over the same features helps.
  `mlp_base` is untuned and `mlp_tuned` changes exactly two settings (dropout,
  class weights), so the comparison isolates their effect. Batches are
  densified one at a time, because the full dense matrix would need ~2 GB.
- *DistilRoBERTa*: context-aware, with the best expected accuracy. It is 40%
  smaller than RoBERTa-base, so it fine-tunes in about 10 minutes on a free T4.
  Class-weighted loss, linear warmup, fp16, and the checkpoint is chosen on
  validation macro F1.

**Model selection on validation macro F1, not loss.** The classes are
imbalanced, and moderation cares about the minority classes, so every model is
selected on macro F1.

**Fairness metrics beyond F1.** Per group we report *hate recall* (attacks that
were missed) and the *false-flag rate on normal posts* (harmless posts about the
group that got flagged). The second is the classic identity-term bias from
Dixon et al. (2018). It is reported only when a group has ≥10 normal posts,
because smaller counts are too noisy.

**Out-of-distribution evaluation instead of more training data.** Adding a few
hundred scraped examples would barely move training. Using them as an
evaluation set answers a more useful question: how much does performance
change on a new platform and a new time period? Sampling is stratified (model-
flagged + random) so there are enough abusive examples to measure, while the
random stratum gives an unbiased estimate. See `DATASHEET_wiki_talk.md`.

**Generated reports.** `make_report.py` writes every number in `README.md` and
`reports/RESULTS.md` from `outputs/`, and copies the metrics and figures into
`reports/` so they are versioned. Hand-typed results drift from the code;
generated ones can't.

## Testing

`tests/` covers labelling logic, split handling, anonymisation and diff parsing
in the collector (including a check that usernames are never requested), the
fairness metrics, and an end-to-end train → evaluate → report run on a tiny
synthetic dataset. CI runs these together with flake8, black and isort. The
torch paths (MLP, transformer) are exercised by the Colab notebook rather than CI.
