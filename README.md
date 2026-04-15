# Identity Abuse / Targeted Harassment Analyzer

A portfolio project for junior Trust & Safety, AI Abuse Analysis, and applied ML roles.

## Project goal

This project builds a realistic moderation-oriented text classification pipeline for detecting harmful language related to identity abuse and targeted harassment.

Version 1 focuses on:
- HateXplain as the main dataset
- 3-class classification: normal / offensive / hatespeech
- Clean preprocessing and reproducible train/val/test splits
- A baseline model with realistic evaluation
- Subgroup analysis using target community annotations
- Error analysis for false positives and moderation risks

## Why this project

Many toxicity projects stop at overall accuracy. This project is designed to be more credible by focusing on:
- False positives on identity-related language
- Differences across target communities
- Class imbalance
- Explainability-aware dataset choice
- Moderation-relevant evaluation, not just headline metrics

## Dataset

### Primary dataset: HateXplain
HateXplain includes:
- 3-class labels
- Target community annotations
- Human rationales

### Later extension: Civil Comments
Civil Comments can be used later for:
- Larger-scale robustness checks
- Bias testing on identity mentions
- Additional fairness-oriented evaluation

## Project structure

```text
identity-abuse-harassment-analyzer/
├── configs/
├── data/
├── models/
├── outputs/
├── reports/
└── src/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate   # or Windows equivalent
pip install -r requirements.txt
```

## Run pipeline

```bash
python src/download_hatexplain.py
python src/preprocess.py --config configs/base_config.yaml
python src/train_baseline.py --config configs/base_config.yaml
python src/evaluate.py --config configs/base_config.yaml
```

## Version 1 baseline

Baseline model:
- TF-IDF features
- Small PyTorch MLP classifier

Evaluation:
- Precision, recall, F1
- Confusion matrix
- Per-class breakdown
- Subgroup breakdown by target community where possible

## Safety note

This dataset contains offensive and hateful text. Development should avoid unnecessary manual browsing, and any analysis outputs shared publicly should quote harmful content minimally and responsibly.

## Next steps

- Add stronger model after baseline
- Add rationale-aware analysis
- Test on Civil Comments for robustness and bias
- Write short error analysis report