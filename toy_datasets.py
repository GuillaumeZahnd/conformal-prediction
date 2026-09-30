from dataclasses import dataclass
import torch
import torchvision


@dataclass(frozen=True)
class Data:
    x: torch.Tensor
    y: torch.Tensor

    def __post_init__(self):
        if len(self.x) != len(self.y):
            raise ValueError(f"x and y must have the same number of rows, got {len(self.x)} and {len(self.y)}.")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return Data(self.x[idx], self.y[idx])


def load_toy_dataset_fashion_mnist(download_path: str, train: bool) -> Data:
    """
    Load the FashionMNIST toy dataset for classification tasks.

    Args:
        download_path: Directory where the raw files are stored (downloaded if missing).
        train: If True, load the training split (60,000 samples); otherwise the test split (10,000 samples).

    Returns:
        Data with:
            x: float32 tensor of shape (N, 1, 28, 28), pixel values in [0, 1].
            y: int64 tensor of shape (N,), class labels in {0, ..., 9}.
    """
    ds = torchvision.datasets.FashionMNIST(download_path, train=train, download=True)
    x = ds.data.unsqueeze(1).float() / 255  # (N, 28, 28) uint8 -> (N, 1, 28, 28) float32
    y = ds.targets
    return Data(x, y)


def split_dataset(data: Data, frac: float, seed: int) -> tuple[Data, Data]:
    idx = torch.randperm(len(data), generator=torch.Generator().manual_seed(seed))
    n = round(frac * len(data))
    return data[idx[:n]], data[idx[n:]]
