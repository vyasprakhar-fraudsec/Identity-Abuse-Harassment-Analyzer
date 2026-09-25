<div align="center">
  <h1>🛡️ Identity Abuse & Targeted Harassment Analyzer</h1>
  <p><b>Detecting hate speech and identity-targeted abuse, measuring who the model fails, and testing it on fresh data it has never seen.</b></p>

[![CI](https://github.com/vyasprakhar-fraudsec/Identity-Abuse-Harassment-Analyzer/actions/workflows/ci.yml/badge.svg)](https://github.com/vyasprakhar-fraudsec/Identity-Abuse-Harassment-Analyzer/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![Dataset](https://img.shields.io/badge/Dataset-HateXplain-purple)](https://github.com/hate-alert/HateXplain)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
</div>

---

## What this project does

Content moderation models are usually judged by one accuracy number. This project asks three harder questions:

1. **How well can we separate hate speech, offensive language and normal speech?** From a TF-IDF baseline up to a fine-tuned transformer, on the official HateXplain split.
2. **Who does the model fail?** Error rates per targeted identity group, including how often *harmless* posts that mention a group get flagged.
3. **Does it hold up on new data?** A small, ethically collected, hand-labelled set of recent Wikipedia talk-page comments, a different platform and a different year from the training data.

## Results

<!-- RESULTS:START -->

| Model | Macro F1 | Accuracy | Normal F1 | Offensive F1 | Hate F1 |
|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | **0.648** | 0.659 | 0.711 | 0.509 | 0.724 |

_Not run yet: TF-IDF + MLP (untuned), TF-IDF + MLP (dropout 0.3, class weights), DistilRoBERTa (fine-tuned), Wikipedia evaluation. See `notebooks/run_pipeline.ipynb`._

- Best model on the HateXplain test set: **TF-IDF + Logistic Regression**, macro F1 0.648.
- Biggest confusion: 96 hate-speech posts predicted as offensive and 107 offensive posts predicted as hate speech.
- Among groups with at least 50 test posts, macro F1 ranges from 0.452 (Refugee) to 0.565 (Women).
- Highest false-flag rate: 77% of normal posts about the **Jewish** group were flagged as abusive (only 13 such posts, so treat as indicative). This is identity-term bias: the model learns the group name itself as a signal of abuse.

Full report: [`reports/RESULTS.md`](reports/RESULTS.md)

<!-- RESULTS:END -->

## How it works

```
HateXplain (pinned release, official split)
  → preprocess: majority-vote labels, majority target groups
  → models: TF-IDF + LogReg │ TF-IDF + MLP │ DistilRoBERTa (fine-tuned)
  → evaluate: per-class metrics, confusion matrix, per-group fairness metrics
  → Wikipedia talk pages (MediaWiki API) → stratified sample → hand labels → out-of-distribution scores
  → make_report: every number in this README is generated from those outputs
```

- **Labels:** majority vote of 3 annotators; posts where all 3 disagree are dropped (as in the paper).
- **Fairness metrics:** a post counts towards every group that ≥2 annotators said it targets. Besides F1, the report tracks *hate recall* (missed attacks) and the *false-flag rate on normal posts* (over-moderation of a group).
- **Explainability:** the demo highlights each word's exact contribution for the linear model.

Design decisions and trade-offs: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## The fresh-data module

`src/collect_wiki.py` builds an evaluation set from English Wikipedia talk pages, collected responsibly:

- **Official MediaWiki API only**, never page scraping. Descriptive User-Agent, one request per second, `maxlag` and back-off, as Wikimedia's API etiquette requires.
- **No personal data**: usernames and page titles are never requested; mentions, signatures, IPs and emails become `<user>`.
- **Only revision ids and labels are committed.** Text is re-fetched locally (`rehydrate`), so anything Wikipedia later deletes disappears here too. Content is CC BY-SA 4.0.
- **Stratified sampling** (half model-flagged, half random), with results reported per stratum, plus Cohen's kappa against a second annotator.

Details: [datasheet](docs/DATASHEET_wiki_talk.md) · [labelling guidelines](docs/LABELING_GUIDELINES.md).

## Quickstart

```bash
git clone https://github.com/vyasprakhar-fraudsec/Identity-Abuse-Harassment-Analyzer
cd Identity-Abuse-Harassment-Analyzer
make setup
make data          # download HateXplain (pinned) + official splits
make baseline      # TF-IDF + logistic regression, CPU, under a minute
make report        # regenerate reports/RESULTS.md and this README's results
make demo          # Gradio app at http://localhost:7860
make test          # test suite
```

GPU steps (transformer, Wikipedia collection, labelling) run end to end in [`notebooks/run_pipeline.ipynb`](notebooks/run_pipeline.ipynb) on a free Colab T4. `make help` lists every target.

## Project structure

```
configs/          one YAML per experiment (data, tfidf_logreg, mlp_base, mlp_tuned, distilroberta, collect_wiki)
src/
  download_hatexplain.py   pinned download of the original release
  preprocess.py            majority labels/targets, official split
  train_baseline.py        TF-IDF + logistic regression or MLP
  train_transformer.py     DistilRoBERTa fine-tuning
  predict.py               one predict_proba interface for every model (+ explanations)
  evaluate.py              metrics, confusion matrix, per-group fairness
  collect_wiki.py          Wikipedia talk-page collector (collect / sample / rehydrate)
  label_wiki.py            terminal labelling tool
  evaluate_ood.py          scores on the new data + inter-annotator agreement
  make_report.py           writes reports/RESULTS.md and the README results
app/app.py        Gradio demo
tests/            pytest suite (runs in CI with lint)
reports/          generated results, figures and metrics (committed)
docs/             architecture, datasheet, labelling guidelines
```

## Limitations

- **English only**, and HateXplain's Twitter/Gab data from 2019–2020 is not representative of all platforms.
- **Group-level numbers are noisy:** several groups have fewer than 100 test posts, and very few *normal* posts mention a group, so false-flag rates rest on small counts (shown in the report).
- **The Wikipedia set is small** (≈500 items) and labelled by one to two people. It shows the direction and rough size of the change on new data, not a precise number.
- **Not a moderation system:** no calibration, no human review loop, not tested against adversarial spelling tricks.

## References

- Mathew et al., *HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection*, AAAI 2021. [arXiv:2012.10289](https://arxiv.org/abs/2012.10289)
- Dixon et al., *Measuring and Mitigating Unintended Bias in Text Classification*, AIES 2018.
- Gebru et al., *Datasheets for Datasets*, CACM 2021.

---

**Prakhar Vyas** · ML for Trust & Safety · MIT License
