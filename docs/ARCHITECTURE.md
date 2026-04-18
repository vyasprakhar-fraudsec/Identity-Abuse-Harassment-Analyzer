# Architecture & Design Decisions

This document explains the technical design choices behind the Identity Abuse & Targeted Harassment Analyzer — written for engineers who want to understand the project at a deeper level than the README.

---

## System Overview

```
┌─────────────────────────────────────────────────────────┐
│                     Data Pipeline                        │
│  HateXplain API → download → preprocess → train/val/test │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                   Feature Extraction                     │
│     Raw text → TF-IDF Vectorizer (30k, 1-2 grams)       │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                     MLP Classifier                       │
│  Linear(30000→256) → ReLU → Dropout → Linear(256→3)     │
│  Loss: CrossEntropyLoss (optionally class-weighted)      │
└────────────────────────┬────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────┐
│                    Evaluation Suite                      │
│  Classification report, Confusion matrix, Subgroup F1   │
└─────────────────────────────────────────────────────────┘
```

---

## Why TF-IDF + MLP as a Baseline?

A common mistake in NLP portfolio projects is jumping straight to transformers. This project deliberately starts with TF-IDF + MLP for principled reasons:

1. **Speed**: Trains in seconds on CPU. Enables fast iteration on preprocessing, class weighting, and label design before investing in expensive GPU fine-tuning.
2. **Interpretability**: TF-IDF weights are inspectable — you can directly examine which tokens drive predictions per class. Useful for debugging and for Trust & Safety audit trails.
3. **Ceiling analysis**: A strong TF-IDF baseline tells you exactly how much a transformer buys you. Without a baseline, F1 gains from BERT are uncontextualized.
4. **Signal**: Building an honest baseline shows ML maturity. It signals that you understand the tradeoff between speed, interpretability, and accuracy.

---

## Data Pipeline Design

### `download_hatexplain.py`
- Uses HuggingFace `datasets` library to pull directly from `Hate-speech-CNERG/hatexplain`.
- Saves raw JSON splits to `data/raw/` for reproducibility and offline use.
- No manual download step required; one command fetches and caches.

### `preprocess.py`
- **Majority label voting**: HateXplain has 3 annotators per post. A `Counter` extracts the majority label, discarding posts with no clear majority.
- **Target group extraction**: Flattens nested target annotations into a single string (e.g., `"african,muslim"`) — used later in subgroup analysis.
- **Stratified splitting**: `train_test_split` with `stratify=label` ensures class balance is maintained across train/val/test splits.
- **Binary mode**: Optional merge of `hate + offensive → abusive` for simpler 2-class experiments (controlled by `label_mode` in config).
- **Text cleaning** (`utils.clean_text`): lowercasing, URL removal, extra whitespace stripping. Intentionally minimal — heavy cleaning can mask signal for hate detection.

### Why no stemming/lemmatization?
Stemming can collapse morphological variants that carry different hateful intent (e.g., verb conjugations). We rely on unigram+bigram TF-IDF instead to capture phrase-level patterns.

---

## Model Architecture

### MLPClassifier (`train_baseline.py`)

```python
nn.Sequential(
    nn.Linear(input_dim, hidden_dim),   # 30000 → 256
    nn.ReLU(),
    nn.Dropout(dropout),                 # 0.0 (base) or 0.1 (tuned)
    nn.Linear(hidden_dim, output_dim),  # 256 → 3
)
```

**Design choices:**
- **Single hidden layer**: Sufficient for linearly separable TF-IDF features. Multiple layers don't help when features are sparse bag-of-words.
- **ReLU over Sigmoid/Tanh**: Better gradient flow for sparse inputs. Most TF-IDF vectors are near-zero; ReLU preserves these zero activations without distortion.
- **Dropout before output**: Applied only to the final projection. Avoids over-regularizing the first projection from a sparse space.

### Class Weighting

HateXplain is imbalanced: offensive posts are overrepresented (~40%), normal (~30%), hate (~30%). Without weighting, the model learns to over-predict offensive.

```python
weights = compute_class_weight('balanced', classes=classes, y=y_train)
criterion = nn.CrossEntropyLoss(weight=class_weights)
```

This scales the loss contribution inversely proportional to class frequency, giving hate and normal posts more gradient signal per batch.

**Impact**: Hate recall improved ~3pp with weighting enabled — the most safety-relevant gain, since missed hate speech is the costlier false negative in production.

---

## Evaluation Design

### Why Macro-F1?
Macro-F1 averages F1 equally across all 3 classes regardless of frequency. This penalizes models that boost accuracy by ignoring minority classes — which matters here because hate speech is the safety-critical minority.

### Subgroup Analysis (`evaluate.py::evaluate_subgroups`)

The subgroup evaluator:
1. Joins predictions back to the test DataFrame (which includes `target_group`).
2. Groups by `target_group` annotation.
3. Filters out groups with <20 examples (too sparse for stable metrics).
4. Computes macro F1 per group and sorts ascending (worst first).

This surfaces **performance gaps by identity** — the core fairness audit needed before any deployment. A model with 0.63 macro-F1 overall but 0.52 on Indigenous posts has a systematic bias that aggregate metrics hide.

---

## Config-Driven Design

All hyperparameters live in YAML configs (`configs/`), not hardcoded in Python. This enables:
- **Experiment tracking**: Each config file = one experiment. `base_config.yaml` vs `tuned_config.yaml` are directly diffable.
- **No code changes for sweeps**: Change `dropout: 0.1 → 0.3` in YAML, rerun — no Python edits needed.
- **Reproducibility**: Config files are committed alongside results. Anyone can reproduce any experiment exactly.

---

## What's Next Architecturally

| Step | Change | Expected Impact |
|---|---|---|
| BERT fine-tuning | Replace TF-IDF → BERT embeddings | +10-15pp macro F1 |
| Focal loss | `alpha`-weighted loss for hard negatives | +2-4pp hate F1 |
| SHAP explanations | Token-level attribution on MLP/BERT | Interpretability for audit |
| Gradio API | Wrap `evaluate.py` inference in FastAPI | Deployable demo |
| Adversarial tests | Typosquat + leet-speak test set | Robustness signal |

---

*Architecture doc maintained by Prakhar Vyas*
