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
3. **Is the better model also fairer?** Comparing over-flagging per group between a simple baseline and a fine-tuned transformer.

It also includes an **ethical data-collection module** for building a fresh evaluation set from Wikipedia talk pages (see below).

## Results

<!-- RESULTS:START -->

| Model | Macro F1 | Accuracy | Normal F1 | Offensive F1 | Hate F1 |
|---|---|---|---|---|---|
| TF-IDF + Logistic Regression | 0.648 | 0.659 | 0.711 | 0.509 | 0.724 |
| TF-IDF + MLP (untuned) | 0.633 | 0.660 | 0.722 | 0.457 | 0.720 |
| TF-IDF + MLP (dropout 0.3, class weights) | 0.641 | 0.653 | 0.708 | 0.501 | 0.713 |
| DistilRoBERTa (fine-tuned) | **0.664** | 0.674 | 0.710 | 0.522 | 0.760 |

- Best model on the HateXplain test set: **DistilRoBERTa (fine-tuned)**, macro F1 0.664.
- That is +0.016 macro F1 over the logistic-regression baseline.
- Most common errors: 187 normal posts predicted as offensive, and 137 offensive posts predicted as hate-speech.
- Among groups with at least 50 test posts, macro F1 ranges from 0.480 (African) to 0.635 (Refugee).
- Highest false-flag rate: 85% of normal posts about the **Jewish** group were flagged as abusive (only 13 such posts, so treat as indicative). This is identity-term bias: the model learns the group name itself as a signal of abuse.
- **Accuracy vs. over-flagging:** The more accurate model catches more hate speech in 7 of 7 groups, but wrongly flags harmless posts more often in 5 of 7 groups. Group counts are small, so read this as a signal.

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

`src/collect_wiki.py` builds an evaluation set from English Wikipedia talk pages, a different platform and time period from the training data. It is collected responsibly:

- **Official MediaWiki API only**, never page scraping. Descriptive User-Agent, one request per second, `maxlag` and back-off, as Wikimedia's API etiquette requires.
- **No personal data**: usernames and page titles are never requested; mentions, signatures, IPs and emails become `<user>`.
- **Only revision ids and labels are committed.** Text is re-fetched locally (`rehydrate`), so anything Wikipedia later deletes disappears here too. Content is CC BY-SA 4.0.
- **Stratified sampling** (half model-flagged, half random), with results reported per stratum, plus Cohen's kappa against a second annotator.
- **Status:** collection, labelling and scoring are built and tested; a labelled set hasn't been published yet. Running notebook steps 5–7 adds the results automatically.

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
- **No out-of-distribution score yet:** all results are on HateXplain's own test split. The Wikipedia module exists to measure this but its labelled set hasn't been produced.
- **Not a moderation system:** no calibration, no human review loop, not tested against adversarial spelling tricks.

## References

- Mathew et al., *HateXplain: A Benchmark Dataset for Explainable Hate Speech Detection*, AAAI 2021. [arXiv:2012.10289](https://arxiv.org/abs/2012.10289)
- Dixon et al., *Measuring and Mitigating Unintended Bias in Text Classification*, AIES 2018.
- Gebru et al., *Datasheets for Datasets*, CACM 2021.

---

**Prakhar Vyas** · ML for Trust & Safety · MIT License
