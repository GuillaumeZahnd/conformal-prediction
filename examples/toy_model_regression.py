from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from toy_datasets import Data


class ToyMLP(nn.Module):
    """
    Small MLP for tabular regression.
    Takes raw features, standardizes them internally (per-feature statistics from the
    training data), and returns one prediction per sample, in the units of the target.
    """
    def __init__(
        self,
        nb_features: int,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
        hidden: int = 64,
    ):
        super().__init__()
        self.register_buffer("mean", torch.zeros(nb_features) if mean is None else mean.clone())
        self.register_buffer("std", torch.ones(nb_features) if std is None else std.clone())
        self.net = nn.Sequential(
            nn.Linear(nb_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mean) / self.std).squeeze(-1)   # (N, nb_features) -> (N,)


def train_toy_regression_model(
    train: Data,
    weights_path: str,
    epochs: int = 20,
    batch_size: int = 128,
    lr: float = 1e-3,
    seed: int = 0,
) -> None:
    """
    Train a ToyMLP with a squared-error loss and save its weights to `weights_path`.
    Normalization statistics come from the training data only and are saved with the weights.
    """
    torch.manual_seed(seed)
    model = ToyMLP(nb_features=train.x.shape[1], mean=train.x.mean(dim=0), std=train.x.std(dim=0))
    opt = torch.optim.Adam(model.parameters(), lr)
    for _ in range(epochs):
        perm = torch.randperm(len(train))
        for i in range(0, len(perm), batch_size):
            b = perm[i : i + batch_size]
            opt.zero_grad()
            F.mse_loss(model(train.x[b]), train.y[b]).backward()
            opt.step()

    path = Path(weights_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)


def load_toy_regression_model(weights_path: str) -> ToyMLP:
    """Load a ToyMLP saved by `train_toy_regression_model`, in eval mode."""
    path = Path(weights_path)
    if not path.exists():
        raise FileNotFoundError(f"No weights at {path}; run train_toy_regression_model first.")
    state = torch.load(path, map_location="cpu")
    model = ToyMLP(nb_features=state["mean"].numel())
    model.load_state_dict(state)
    return model.eval()
