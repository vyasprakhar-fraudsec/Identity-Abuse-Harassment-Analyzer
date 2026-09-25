.PHONY: help setup data baseline mlp transformer wiki-collect wiki-sample label ood report demo test lint format clean

PY = python

help:  ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-14s\033[0m %s\n", $$1, $$2}'

setup:  ## Install dependencies
	pip install -r requirements.txt -r requirements-dev.txt

data:  ## Download HateXplain (pinned) and build the official splits
	$(PY) src/download_hatexplain.py
	$(PY) src/preprocess.py

baseline:  ## TF-IDF + logistic regression (CPU, seconds)
	$(PY) src/train_baseline.py --config configs/tfidf_logreg.yaml
	$(PY) src/evaluate.py --config configs/tfidf_logreg.yaml

mlp:  ## TF-IDF + MLP, untuned and tuned
	$(PY) src/train_baseline.py --config configs/mlp_base.yaml
	$(PY) src/evaluate.py --config configs/mlp_base.yaml
	$(PY) src/train_baseline.py --config configs/mlp_tuned.yaml
	$(PY) src/evaluate.py --config configs/mlp_tuned.yaml

transformer:  ## Fine-tune DistilRoBERTa (GPU recommended)
	$(PY) src/train_transformer.py --config configs/distilroberta.yaml
	$(PY) src/evaluate.py --config configs/distilroberta.yaml

wiki-collect:  ## Collect recent Wikipedia talk-page comments (~25 min, rate-limited)
	$(PY) src/collect_wiki.py collect

wiki-sample:  ## Choose 500 comments to label
	$(PY) src/collect_wiki.py sample

label:  ## Label comments in the terminal (ANNOTATOR=yourname)
	$(PY) src/label_wiki.py --annotator $(ANNOTATOR)

ood:  ## Score all trained models on the labelled Wikipedia set (GOLD=yourname)
	$(PY) src/evaluate_ood.py --gold $(GOLD) --models $$(for c in tfidf_logreg mlp_base mlp_tuned distilroberta; do [ -d models/$$c ] && echo configs/$$c.yaml; done)

report:  ## Regenerate reports/RESULTS.md and the README results from outputs/
	$(PY) src/make_report.py

demo:  ## Launch the Gradio demo locally
	$(PY) app/app.py

test:  ## Run the test suite
	pytest -q

lint:  ## flake8 + black + isort checks
	flake8 src tests app
	black --check src tests app
	isort --check-only src tests app

format:  ## Auto-format
	black src tests app
	isort src tests app

clean:  ## Remove models and outputs (keeps data and reports)
	rm -rf models/ outputs/
