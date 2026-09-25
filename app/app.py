"""Gradio demo: classify a post and show which words drove the prediction.

Run locally:  python app/app.py
Loads every model in MODEL_CONFIGS whose trained files exist.
"""

import sys
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from label_maps import LABELS  # noqa: E402
from predict import load_predictor  # noqa: E402
from utils import experiment_paths, load_config  # noqa: E402

MODEL_CONFIGS = {
    "DistilRoBERTa (fine-tuned)": "configs/distilroberta.yaml",
    "TF-IDF + Logistic Regression (explainable)": "configs/tfidf_logreg.yaml",
}
DISPLAY = {"normal": "Normal", "offensive": "Offensive", "hatespeech": "Hate speech"}

EXAMPLES = [
    "Thanks for fixing the citation, the article reads much better now.",
    "You are a complete idiot and should stop editing.",
    "Those people are ruining this country and should all be sent back.",
]


def available_models():
    models = {}
    for name, cfg_path in MODEL_CONFIGS.items():
        cfg = load_config(ROOT / cfg_path)
        cfg["data_config"] = str(ROOT / cfg["data_config"])
        if experiment_paths(cfg)["model_dir"].exists():
            models[name] = load_predictor(cfg)
    if not models:
        raise SystemExit("No trained models found. Run `make baseline` first.")
    return models


def main():
    import os

    os.chdir(ROOT)  # model paths are relative to the repo root
    models = available_models()

    def classify(text, model_name):
        if not text.strip():
            return {}, []
        predictor = models[model_name]
        probs = predictor.predict_proba([text])[0]
        scores = {DISPLAY[label]: float(p) for label, p in zip(LABELS, probs)}
        contribs = dict(predictor.explain(text))
        highlighted = []
        for word in text.split():
            key = word.lower().strip(".,!?;:\"'()")
            highlighted.append((word + " ", round(contribs[key], 3) if key in contribs else None))
        return scores, highlighted

    with gr.Blocks(title="Identity Abuse & Harassment Analyzer") as demo:
        gr.Markdown(
            "# Identity Abuse & Harassment Analyzer\n"
            "Classifies text as **normal**, **offensive** or **hate speech** (HateXplain labels). "
            "Word highlights are exact contributions from the linear model. "
            "A research demo, not a moderation tool: see the fairness results in the repo."
        )
        with gr.Row():
            with gr.Column():
                text = gr.Textbox(label="Text", lines=4)
                model = gr.Dropdown(list(models), value=list(models)[0], label="Model")
                button = gr.Button("Classify", variant="primary")
            with gr.Column():
                label = gr.Label(label="Prediction", num_top_classes=3)
                words = gr.HighlightedText(
                    label="Words pushing towards the predicted class (linear model only)",
                    combine_adjacent=False,
                )
        gr.Examples(EXAMPLES, inputs=text)
        button.click(classify, [text, model], [label, words])
        text.submit(classify, [text, model], [label, words])

    demo.launch()


if __name__ == "__main__":
    main()
