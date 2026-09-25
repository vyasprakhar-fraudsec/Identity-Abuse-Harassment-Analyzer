"""One interface for every trained model, used by evaluation, OOD evaluation and the demo.

    predictor = load_predictor("configs/tfidf_logreg.yaml")
    probs = predictor.predict_proba(["some text"])   # shape (n, 3), order = label_maps.LABELS
"""

import numpy as np

from label_maps import LABELS
from utils import experiment_paths, load_config


class TfidfPredictor:
    def __init__(self, config):
        import joblib

        self.config = config
        model_dir = experiment_paths(config)["model_dir"]
        self.vectorizer = joblib.load(model_dir / "vectorizer.joblib")
        self.type = config["model"]["type"]
        if self.type == "logreg":
            self.clf = joblib.load(model_dir / "logreg.joblib")
        else:
            import torch

            from models import MLPClassifier

            m = config["model"]
            self.torch = torch
            self.clf = MLPClassifier(len(self.vectorizer.vocabulary_), m["hidden_dim"], 3, m["dropout"])
            self.clf.load_state_dict(torch.load(model_dir / "mlp.pt", map_location="cpu"))
            self.clf.eval()

    def predict_proba(self, texts):
        X = self.vectorizer.transform(list(texts))
        if self.type == "logreg":
            return self.clf.predict_proba(X)
        out = []
        with self.torch.no_grad():
            for start in range(0, X.shape[0], 512):
                batch = self.torch.tensor(X[start : start + 512].toarray(), dtype=self.torch.float32)
                out.append(self.torch.softmax(self.clf(batch), dim=1).numpy())
        return np.concatenate(out)

    def explain(self, text, top_k=8):
        """Per-token contribution to the predicted class (logistic regression only).

        For a linear model this is exact: contribution = tfidf value x class weight.
        """
        if self.type != "logreg":
            return []
        x = self.vectorizer.transform([text])
        pred = int(self.clf.predict(x)[0])
        vocab = self.vectorizer.get_feature_names_out()
        contribs = [(vocab[j], float(v * self.clf.coef_[pred, j])) for j, v in zip(x.indices, x.data)]
        return sorted(contribs, key=lambda t: -abs(t[1]))[:top_k]


class TransformerPredictor:
    def __init__(self, config):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.config = config
        model_dir = experiment_paths(config)["model_dir"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device)
        self.model.eval()
        self.max_length = config["model"]["max_length"]

    def predict_proba(self, texts, batch_size=64):
        texts = list(texts)
        out = []
        with self.torch.no_grad():
            for start in range(0, len(texts), batch_size):
                enc = self.tokenizer(
                    texts[start : start + batch_size],
                    truncation=True,
                    max_length=self.max_length,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                logits = self.model(**enc).logits
                out.append(self.torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(out)

    def explain(self, text, top_k=8):
        return []


def load_predictor(config_or_path):
    config = load_config(config_or_path) if isinstance(config_or_path, str) else config_or_path
    if config["model"]["type"] == "transformer":
        return TransformerPredictor(config)
    return TfidfPredictor(config)


def predict_labels(predictor, texts):
    probs = predictor.predict_proba(texts)
    return probs.argmax(axis=1), probs


__all__ = ["LABELS", "load_predictor", "predict_labels"]
