<div align="center">
  <h1>🛡️ Identity Abuse & Targeted Harassment Analyzer</h1>
  <p><b>Applied ML for Trust & Safety — detecting hate speech and identity-based harassment using the HateXplain benchmark.</b></p>
</div>

[![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange?style=flat&logo=pytorch)](https://pytorch.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-green?style=flat&logo=scikitlearn)](https://scikit-learn.org)
[![HateXplain](https://img.shields.io/badge/Dataset-HateXplain-purple?style=flat)](https://huggingface.co/datasets/Hate-speech-CNERG/hatexplain)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow?style=flat)](LICENSE)

---

## Overview

This project builds a text classifier to detect **hate speech**, **offensive content**, and **normal speech** on social media, with a focus on **identity-targeted abuse** (e.g., posts targeting race, religion, gender, or sexual orientation).

It uses the [HateXplain dataset](https://huggingface.co/datasets/Hate-speech-CNERG/hatexplain) — a benchmark specifically designed for explainable hate speech detection — and trains a **TF-IDF + MLP baseline** with class-weighted loss to handle label imbalance. Subgroup fairness analysis is included to surface per-identity performance gaps.

This is a portfolio project focused on **Trust & Safety**, **applied NLP**, and **responsible AI evaluation**.

---

## Why This Matters

Platforms moderating user-generated content face a hard problem: generic toxicity classifiers miss **targeted identity-based abuse** that is contextually harmful but linguistically subtle. HateXplain provides:

- **3-class labels**: hate / offensive / normal
- **Target group annotations**: which identity group is being attacked
- **Human rationales**: which tokens justify the label

This makes it ideal for building moderation systems that are not only accurate but **auditable and bias-aware**.

---

## Project Structure

```
Identity-Abuse-Harassment-Analyzer/
├── configs/
│   └── base_config.yaml        # All hyperparameters and paths in one place
├── reports/
│   └── RESULTS.md              # Full evaluation report: metrics, confusion, subgroup analysis
├── src/
│   ├── download_hatexplain.py  # Downloads dataset via HuggingFace datasets
│   ├── inspect_hatexplain.py   # EDA: label distribution, target groups, text stats
│   ├── preprocess.py           # Cleans text, builds train/val/test CSVs
│   ├── label_maps.py           # Label encoding utilities
│   ├── train_baseline.py       # TF-IDF + MLP training with class weighting
│   ├── evaluate.py             # Classification report, confusion matrix, subgroup F1
│   └── utils.py                # Config loading, seed setting, dir management
├── LICENSE
├── requirements.txt
└── README.md
```

---

## Model Architecture

**Baseline: TF-IDF → MLP Classifier**

```
Input text
  → TF-IDF Vectorizer (max_features=30,000, unigrams + bigrams)
  → Linear(30000 → 256)
  → ReLU
  → Dropout(0.1)
  → Linear(256 → 3)
  → CrossEntropyLoss with class weights
```

- **Why TF-IDF + MLP first?** Establishes an interpretable, fast-to-train baseline before moving to transformer fine-tuning. A strong baseline is a signal of good ML practice.
- **Class weighting**: Handles the hate/offensive/normal imbalance in HateXplain.
- **Config-driven**: All hyperparameters live in `configs/base_config.yaml` — reproducible and easy to sweep.

---

## Key Results

> Metrics reported on held-out test set (15% split, seed=42). Full report: [`reports/RESULTS.md`](reports/RESULTS.md)

| Variant | Macro F1 | Hate F1 | Notes |
|---------|----------|---------|-------|
| Baseline | 0.6254 | 0.59 | dropout=0.0, no class weighting |
| Tuned v2 | **0.6267** | **0.62** | dropout=0.1, class weighting enabled |

**Key win**: Class weighting lifted hate recall by ~3pp — the most safety-critical class to get right.

### Confusion Matrix Summary (Tuned v2)

```
Predicted →   hate    offensive    normal
Actual hate:   251       112          49
Actual off.:    87       641         163
Actual normal:  52       108         214
```

Dominant error: **hate ↔ offensive** misclassification (expected — lexical overlap is high for bag-of-words features). Hate ↔ normal confusion is low (49 cases), which matters most for safety.

### Subgroup Fairness (Macro F1 by target identity)

| Target Group | Macro F1 | Trend |
|---|---|---|
| African | 0.71 | Best represented in training |
| Muslim | 0.68 | Strong lexical signal |
| Jewish | 0.66 | Moderate |
| Women | 0.65 | Gender-based hate harder to detect |
| LGBTQ+ | 0.61 | Contextual/reclaimed language hurts TF-IDF |
| Asian | 0.60 | Underrepresented |
| Indigenous | 0.55 | Sparse — treat with caution |
| Caucasian | 0.52 | Counter-speech misclassified |

19pp gap between best and worst group — highlights why per-group auditing is essential before any deployment.

> Full error analysis, limitations table, and reproduction steps: [`reports/RESULTS.md`](reports/RESULTS.md)

---

## Quickstart

```bash
# 1. Clone and install
git clone https://github.com/vyasprakhar-fraudsec/Identity-Abuse-Harassment-Analyzer
cd Identity-Abuse-Harassment-Analyzer
pip install -r requirements.txt

# 2. Download and preprocess data
python src/download_hatexplain.py
python src/preprocess.py --config configs/base_config.yaml

# 3. Train
python src/train_baseline.py --config configs/base_config.yaml

# 4. Evaluate (generates confusion matrix, subgroup report, predictions)
python src/evaluate.py --config configs/base_config.yaml
```

All outputs are written to `outputs/` (metrics CSVs, confusion matrix PNG, predictions CSV).

---

## Limitations

- **Ceiling on TF-IDF features**: Bag-of-words representations miss context, sarcasm, and dog-whistle language. Transformer fine-tuning is the clear next step.
- **English-only**: HateXplain is English-centric; cross-lingual generalization is untested.
- **Static dataset**: No online learning or drift detection. Real-world abuse patterns evolve rapidly.
- **Subgroup data sparsity**: Groups with <20 test examples are excluded from subgroup analysis — some minority communities may be underrepresented.
- **No adversarial robustness**: The model is not tested against deliberate obfuscation (e.g., leetspeak, spacing tricks).

---

## Next Steps

- [ ] Fine-tune `bert-base-uncased` or `roberta-base` on HateXplain for a meaningful F1 lift
- [ ] Add SHAP token-level explanations to visualize what drives predictions
- [ ] Build a Gradio demo for interactive inference
- [ ] Benchmark against Perspective API on the same test split
- [ ] Experiment with focal loss to further address class imbalance
- [ ] Explore multilingual extension with `xlm-roberta-base`

---

## Tech Stack

| Component | Tool |
|-----------|------|
| Language | Python 3.10+ |
| ML framework | PyTorch 2.0+ |
| Feature extraction | scikit-learn TfidfVectorizer |
| Data | HuggingFace `datasets` |
| Visualization | matplotlib, seaborn |
| Config management | PyYAML |
| Reproducibility | Fixed seed (42), config-driven |

---

## Dataset

**HateXplain** — Mathew et al., AAAI 2021
- ~20,000 posts from Twitter and Gab
- 3-class labels (hate / offensive / normal)
- Target group annotations (10+ identity categories)
- Human rationale spans for explainability

[HuggingFace Dataset](https://huggingface.co/datasets/Hate-speech-CNERG/hatexplain) | [Paper (arXiv)](https://arxiv.org/abs/2012.10289)

---

**Prakhar Vyas** | Aspiring ML Engineer — Trust & Safety / Applied NLP

*Built with PyTorch · MIT License*
