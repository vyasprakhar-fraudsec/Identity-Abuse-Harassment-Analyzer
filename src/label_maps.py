"""Label encoding. The order here is the order used everywhere (probabilities, reports)."""

LABELS = ["normal", "offensive", "hatespeech"]
LABEL_TO_ID = {label: i for i, label in enumerate(LABELS)}
ID_TO_LABEL = {i: label for label, i in LABEL_TO_ID.items()}

# "Abusive" = offensive or hate speech. Used for fairness metrics.
ABUSIVE_IDS = {LABEL_TO_ID["offensive"], LABEL_TO_ID["hatespeech"]}
