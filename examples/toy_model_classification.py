from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from toy_datasets import Data


class ToyCNN(nn.Module):
    """
    Small CNN for 28x28 grayscale images.
    Takes images in [0, 1], normalizes them internally, and returns logits.
    """

    def __init__(self, mean: float = 0.0, std: float = 1.0, nb_classes: int = 10):
        super().__init__()
        self.register_buffer("mean", torch.tensor(float(mean)))
        self.register_buffer("std", torch.tensor(float(std)))
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3), nn.ReLU(), nn.MaxPool2d(2),
            nn.Flatten(), nn.Linear(32 * 5 * 5, nb_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mean) / self.std)


def train_toy_classification_model(
    train: Data,
    weights_path: str,
    epochs: int = 1,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 0
) -> None:
    """
    Train a ToyCNN and save its weights to `weights_path`.
    Normalization statistics come from the training data only and are saved with the weights.
    """
    torch.manual_seed(seed)
    model = ToyCNN(mean=train.x.mean(), std=train.x.std())
    opt = torch.optim.Adam(model.parameters(), lr)
    for _ in range(epochs):
        perm = torch.randperm(len(train))
        for i in range(0, len(perm), batch_size):
            b = perm[i : i + batch_size]
            opt.zero_grad()
            F.cross_entropy(model(train.x[b]), train.y[b]).backward()
            opt.step()

    path = Path(weights_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_toy_classification_model(weights_path: str) -> ToyCNN:
    """Load a ToyCNN saved by `train_toy_classification_model`, in eval mode."""
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"No weights at {path}; run train_toy_classification_model first.")
    model = ToyCNN()   # placeholder mean/std: overwritten by the saved ones below
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model.eval()
