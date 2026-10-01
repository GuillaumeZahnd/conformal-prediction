from dataclasses import dataclass
import torch
import torchvision
from sklearn.datasets import fetch_california_housing


@dataclass(frozen=True)
class Data:
    x: torch.Tensor
    y: torch.Tensor
    class_names: tuple[str, ...] | None = None

    def __post_init__(self):
        if len(self.x) != len(self.y):
            raise ValueError(f"x and y must have the same number of rows, got {len(self.x)} and {len(self.y)}.")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return Data(self.x[idx], self.y[idx], self.class_names)


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
            class_names: ("T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot")
    """
    ds = torchvision.datasets.FashionMNIST(download_path, train=train, download=True)
    x = ds.data.unsqueeze(1).float() / 255  # (N, 28, 28) uint8 -> (N, 1, 28, 28) float32
    y = ds.targets
    class_names = tuple(ds.classes)
    return Data(x, y, class_names)


def load_toy_dataset_california_housing(
    download_path: str, train: bool, test_ratio: float = 0.3, seed: int = 0
) -> Data:
    """
    Load the California housing toy dataset for regression tasks.

    The original dataset has no official split, so a fixed random split is made. The same
    `seed` and `test_ratio` give disjoint train and test parts across the two calls.

    Args:
        download_path: Directory where the raw files are stored (downloaded if missing).
        train: If True, load the training part; otherwise the held-out test part.
        test_ratio: Fraction of the 20,640 samples held out as the test part.
        seed: Seed of the train/test split.

    Returns:
        Data with:
            x: float32 tensor of shape (N, 8), raw features (their scales differ widely).
            y: float32 tensor of shape (N,), median house value in units of $100,000.
            class_names: None (regression).
    """

    x, y = fetch_california_housing(data_home=download_path, return_X_y=True)
    full = Data(torch.from_numpy(x).float(), torch.from_numpy(y).float())
    train_part, test_part = split_dataset(full, ratio=1 - test_ratio, seed=seed)
    return train_part if train else test_part


def split_dataset(data: Data, ratio: float, seed: int) -> tuple[Data, Data]:
    idx = torch.randperm(len(data), generator=torch.Generator().manual_seed(seed))
    n = round(ratio * len(data))
    return data[idx[:n]], data[idx[n:]]
