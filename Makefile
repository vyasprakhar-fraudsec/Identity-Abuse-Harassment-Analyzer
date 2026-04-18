.PHONY: help setup download preprocess train train-tuned evaluate evaluate-tuned lint format all clean

CONFIG_BASE = configs/base_config.yaml
CONFIG_TUNED = configs/tuned_config.yaml

help:  ## Show available commands
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

setup:  ## Install all dependencies
	pip install -r requirements.txt

download:  ## Download HateXplain dataset from HuggingFace
	python src/download_hatexplain.py

preprocess:  ## Clean and split dataset into train/val/test CSVs
	python src/preprocess.py --config $(CONFIG_BASE)

train:  ## Train baseline model (TF-IDF + MLP, no class weighting)
	python src/train_baseline.py --config $(CONFIG_BASE)

train-tuned:  ## Train tuned model (dropout=0.1, class weighting enabled)
	python src/train_baseline.py --config $(CONFIG_TUNED)

evaluate:  ## Evaluate baseline model (confusion matrix, subgroup F1, predictions)
	python src/evaluate.py --config $(CONFIG_BASE)

evaluate-tuned:  ## Evaluate tuned model
	python src/evaluate.py --config $(CONFIG_TUNED)

lint:  ## Run flake8 linting
	flake8 src/ --max-line-length=100 --ignore=E203,W503

format:  ## Auto-format with black and isort
	black src/
	isort src/

all: download preprocess train evaluate  ## Run full baseline pipeline end-to-end

all-tuned: download preprocess train-tuned evaluate-tuned  ## Run full tuned pipeline end-to-end

clean:  ## Remove generated outputs (models, outputs) - keeps raw data
	rm -rf models/ outputs/
	@echo "Cleaned models/ and outputs/ directories"
