from dataclasses import dataclass
from typing import TypeVar
import torch
import torchvision
from sklearn.datasets import fetch_california_housing

DatasetTypeVar = TypeVar("DatasetTypeVar", bound="BaseDataset")


@dataclass(frozen=True)
class BaseDataset:
    """Base container for paired feature and target tensors."""

    x: torch.Tensor
    y: torch.Tensor

    def __post_init__(self) -> None:
        if len(self.x) != len(self.y):
            raise ValueError(
                f"x and y must have the same number of rows, got {len(self.x)} and {len(self.y)}."
            )

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self: DatasetTypeVar, idx) -> DatasetTypeVar:
        return type(self)(x=self.x[idx], y=self.y[idx])


@dataclass(frozen=True)
class ClassificationDataset(BaseDataset):
    """Dataset container for multi-class classification tasks."""

    class_names: tuple[str, ...]

    def __getitem__(self, idx) -> "ClassificationDataset":
        return ClassificationDataset(
            x=self.x[idx],
            y=self.y[idx],
            class_names=self.class_names,
        )


@dataclass(frozen=True)
class RegressionDataset(BaseDataset):
    """Dataset container for continuous regression tasks."""


DatasetType = ClassificationDataset | RegressionDataset


def load_toy_dataset_fashion_mnist(download_path: str, train: bool) -> ClassificationDataset:
    """Load the FashionMNIST toy dataset for classification tasks.

    Args:
        download_path: Directory where the raw files are stored (downloaded if missing).
        train: If True, load the training split (60,000 samples); otherwise the test split
            (10,000 samples).

    Returns:
        ``ClassificationDataset`` with:
            - ``x``: ``float32`` tensor of shape ``(N, 1, 28, 28)``, pixel values in ``[0, 1]``.
            - ``y``: ``int64`` tensor of shape ``(N,)``, class labels in ``{0, ..., 9}``.
            - ``class_names``: Tuple of Fashion-MNIST class label strings.
    """
    ds = torchvision.datasets.FashionMNIST(download_path, train=train, download=True)
    x = ds.data.unsqueeze(1).float() / 255.0  # (N, 28, 28) uint8 -> (N, 1, 28, 28) float32
    y = ds.targets
    class_names = tuple(ds.classes)
    return ClassificationDataset(x=x, y=y, class_names=class_names)


def load_toy_dataset_california_housing(
    download_path: str,
    train: bool,
    test_ratio: float = 0.3,
    seed: int = 0,
) -> RegressionDataset:
    """Load the California housing toy dataset for regression tasks.

    The original dataset has no official split, so a fixed random split is made. The same
    ``seed`` and ``test_ratio`` give disjoint train and test parts across the two calls.

    Args:
        download_path: Directory where the raw files are stored (downloaded if missing).
        train: If True, load the training part; otherwise the held-out test part.
        test_ratio: Fraction of the 20,640 samples held out as the test part.
        seed: Seed of the train/test split.

    Returns:
        ``RegressionDataset`` with:
            - ``x``: ``float32`` tensor of shape ``(N, 8)``, raw features (their scales differ widely).
            - ``y``: ``float32`` tensor of shape ``(N,)``, median house value in units of $100,000.
    """
    x, y = fetch_california_housing(data_home=download_path, return_X_y=True)
    full = RegressionDataset(
        x=torch.from_numpy(x).float(),
        y=torch.from_numpy(y).float(),
    )
    train_part, test_part = split_dataset(full, ratio=1.0 - test_ratio, seed=seed)
    return train_part if train else test_part


def split_dataset(data: DatasetTypeVar, ratio: float, seed: int) -> tuple[DatasetTypeVar, DatasetTypeVar]:
    """Split a dataset into two disjoint subsets according to ``ratio``.

    Args:
        data: Dataset instance to split.
        ratio: Fraction of samples placed in the first split; must be in ``[0.0, 1.0]``.
        seed: Random seed used to shuffle sample indices.

    Returns:
        A tuple ``(first_split, second_split)`` of the same dataset type as ``data``.
    """
    if not (0.0 <= ratio <= 1.0):
        raise ValueError(f"ratio must be between 0.0 and 1.0, got {ratio}.")
    idx = torch.randperm(len(data), generator=torch.Generator().manual_seed(seed))
    n = round(ratio * len(data))
    return data[idx[:n]], data[idx[n:]]
