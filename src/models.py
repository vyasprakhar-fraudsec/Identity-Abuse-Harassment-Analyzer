"""PyTorch model definitions (kept separate so scikit-learn code never imports torch)."""

import torch.nn as nn


class MLPClassifier(nn.Module):
    """TF-IDF features -> hidden layer -> 3 logits."""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.net(x)
