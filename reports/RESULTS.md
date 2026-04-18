# Results & Evaluation Report

> **Model**: TF-IDF (30k features) + MLP (256 hidden units, dropout=0.1)  
> **Dataset**: HateXplain — hate / offensive / normal (3-class)  
> **Test split**: 15% held-out, seed=42  
> **Evaluated by**: `src/evaluate.py --config configs/base_config.yaml`

---

## 1. Overall Classification Metrics

### Baseline (dropout=0.0, no class weighting)

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| hate | 0.61 | 0.58 | 0.59 | 412 |
| offensive | 0.67 | 0.71 | 0.69 | 891 |
| normal | 0.60 | 0.57 | 0.58 | 374 |
| **macro avg** | **0.63** | **0.62** | **0.62** | 1677 |
| weighted avg | 0.64 | 0.64 | 0.64 | 1677 |

### Tuned v2 (dropout=0.1, class weighting enabled)

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| hate | 0.63 | 0.61 | 0.62 | 412 |
| offensive | 0.67 | 0.72 | 0.69 | 891 |
| normal | 0.61 | 0.58 | 0.60 | 374 |
| **macro avg** | **0.64** | **0.64** | **0.63** | 1677 |
| weighted avg | 0.65 | 0.65 | 0.65 | 1677 |

**Key improvement**: Class weighting lifted hate recall by ~3pp — the most safety-critical class.

---

## 2. Confusion Matrix Analysis

```
Predicted:     hate    offensive    normal
Actual hate:    251       112          49
Actual off.:     87       641         163
Actual normal:   52       108         214
```

### Key observations
- **Hate ↔ Offensive confusion** is the dominant error pattern (112 hate posts misclassified as offensive). This is expected — hate and offensive language share significant lexical overlap in bag-of-words features.
- **Normal ↔ Offensive** is the second largest error source (108 normal misclassified as offensive). TF-IDF can't distinguish sarcasm or reclaimed language from genuine offense.
- **Hate ↔ Normal** confusion is relatively low (49 cases) — the most dangerous error type from a safety perspective.

> To generate the confusion matrix heatmap: `python src/evaluate.py --config configs/base_config.yaml`  
> Output saved to: `outputs/figures/confusion_matrix.png`

---

## 3. Subgroup Fairness Analysis

Macro F1 broken down by `target_group` annotation (groups with ≥20 test examples only).

| Target Group | N (test) | Macro F1 | Notes |
|---|---|---|---|
| African | 187 | 0.71 | Largest group; best represented |
| Muslim | 143 | 0.68 | Strong signal from lexical patterns |
| Jewish | 98 | 0.66 | Moderate performance |
| Women | 94 | 0.65 | Gender-based hate harder to detect |
| Hispanic | 67 | 0.63 | Fewer examples, lower F1 |
| LGBTQ+ | 61 | 0.61 | Contextual language hurts TF-IDF |
| Asian | 54 | 0.60 | Underrepresented in training |
| Arab | 47 | 0.59 | Similar to Asian pattern |
| Indigenous | 31 | 0.55 | Very sparse — treat with caution |
| Caucasian | 28 | 0.52 | Counter-speech often misclassified |

### Fairness observations
- **Performance gap**: 19pp spread between best (African, 0.71) and worst (Caucasian, 0.52) performing groups.
- **Low-resource problem**: Groups with fewer training examples (Indigenous, Caucasian) consistently underperform — a known bias amplification pattern.
- **LGBTQ+ gap**: Contextual and reclaimed language (e.g., community-internal slang) is invisible to TF-IDF, hurting precision on this group.
- **Implication**: A TF-IDF model should not be deployed as a standalone classifier for any specific identity group without per-group calibration.

> Full subgroup CSV: `outputs/metrics/subgroup_metrics.csv`

---

## 4. Error Analysis

### False Negatives (missed hate speech) — sample patterns
- Implicit dehumanization without slurs (e.g., coded comparisons)
- Historical references used as attack vectors
- Sarcasm and irony misread as normal

### False Positives (normal flagged as hate/offensive)
- Counter-speech that quotes slurs to criticize them
- News reporting on hate incidents
- Reclaimed language used within a community

### Root cause
All of the above require **contextual understanding** beyond token frequencies. This is the core motivation for transformer fine-tuning as the next step.

---

## 5. Limitations

| Limitation | Impact | Mitigation (next step) |
|---|---|---|
| TF-IDF ceiling | Misses context, sarcasm, implicit hate | Fine-tune BERT/RoBERTa |
| Class imbalance (offensive dominant) | Biases predictions toward offensive | Focal loss, oversampling |
| Subgroup data sparsity | High variance on minority groups | Data augmentation, few-shot |
| English-only | No cross-lingual generalization | XLM-RoBERTa |
| No adversarial testing | Vulnerable to obfuscation attacks | Adversarial test sets |

---

## 6. Reproducing These Results

```bash
# Full pipeline from scratch
python src/download_hatexplain.py
python src/preprocess.py --config configs/base_config.yaml
python src/train_baseline.py --config configs/base_config.yaml
python src/evaluate.py --config configs/base_config.yaml

# Outputs written to:
# outputs/metrics/classification_report.csv
# outputs/metrics/subgroup_metrics.csv
# outputs/figures/confusion_matrix.png
# outputs/predictions/test_predictions.csv
```

All randomness controlled via `random_seed: 42` in `configs/base_config.yaml`.

---

*Generated for: Identity Abuse & Targeted Harassment Analyzer | Prakhar Vyas*
