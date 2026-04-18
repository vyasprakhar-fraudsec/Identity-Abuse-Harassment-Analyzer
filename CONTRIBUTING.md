# Contributing to Identity Abuse & Targeted Harassment Analyzer

Thank you for your interest in contributing. This is a portfolio ML project focused on Trust & Safety and responsible NLP — contributions that improve fairness, evaluation quality, or model performance are especially welcome.

---

## Ways to Contribute

### Bug Reports
If you find something that doesn't work (broken pipeline step, incorrect metric, data loading error), please open an issue with:
- What you ran (exact command)
- What you expected
- What happened (error message or wrong output)
- Your Python version and OS

### Improvements Welcome
- Better preprocessing (e.g., handling Unicode slurs, emoji)
- Alternative model architectures (BERT fine-tuning, SVM baseline)
- Additional evaluation metrics (AUC-ROC, per-class calibration curves)
- Broader subgroup analysis (intersectional identities)
- Adversarial test cases for robustness evaluation
- Gradio demo or FastAPI inference endpoint

### Documentation
- Corrections or additions to `docs/ARCHITECTURE.md`
- Clarifying comments in source code
- Additional usage examples in README

---

## Getting Started

```bash
# 1. Fork and clone
git clone https://github.com/YOUR_USERNAME/Identity-Abuse-Harassment-Analyzer
cd Identity-Abuse-Harassment-Analyzer

# 2. Set up environment
make setup

# 3. Run the full pipeline to verify everything works
make all

# 4. Run lint before submitting
make lint
```

---

## Code Standards

- **Style**: Follow [PEP 8](https://pep8.org/). Max line length: 100 characters.
- **Formatting**: Use `black` for formatting (`make format`).
- **Imports**: Sort with `isort` (`make format` handles this).
- **Type hints**: Encouraged but not required for this project stage.
- **Comments**: Explain *why*, not *what*. Code is self-documenting; comments add context.

---

## Config-First Approach

If your contribution changes a hyperparameter or training behaviour:
- Add a new YAML file to `configs/` rather than modifying an existing one.
- Name it descriptively: `bert_config.yaml`, `focal_loss_config.yaml`, etc.
- Add a comment header explaining what changed and why.

This keeps experiments reproducible and auditable.

---

## Responsible AI Guidelines

This project deals with sensitive content (hate speech, targeted harassment). When contributing:

- **No raw hate speech examples** in code, comments, or tests. Use placeholder tokens like `[SLUR]` if needed.
- **Bias awareness**: Any new model or feature should include a subgroup fairness check. Run `src/evaluate.py` and review `subgroup_metrics.csv`.
- **Honest reporting**: Don't inflate metrics. If a change doesn't help, that's useful information.
- **Dataset terms**: HateXplain is for research use. Do not redistribute the raw dataset.

---

## Submitting a Pull Request

1. Create a feature branch: `git checkout -b feat/your-feature-name`
2. Make your changes with clear commit messages (e.g., `feat: add focal loss to train_baseline.py`)
3. Run `make lint` and fix any issues
4. Push and open a PR against `main`
5. Describe what you changed and why in the PR description

---

## Questions?

Open a GitHub Issue with the `question` label. Response time may vary as this is a solo portfolio project.

---

*This project follows a [Code of Conduct](https://www.contributor-covenant.org/version/2/1/code_of_conduct/) based on the Contributor Covenant v2.1.*
