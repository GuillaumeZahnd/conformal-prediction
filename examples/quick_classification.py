import torch

from toy_datasets import (
    ClassificationDataset,
    load_toy_dataset_fashion_mnist,
    split_dataset,
)
from toy_model_classification import (
    ToyCNN,
    load_toy_classification_model,
    train_toy_classification_model,
)
from conformal_prediction import predict_probs


def quick_classification(
    path_to_toy_datasets: str,
    path_to_trained_model: str,
    random_seed: int,
) -> tuple[ClassificationDataset, ClassificationDataset, torch.Tensor, torch.Tensor]:
    """Train a ``ToyCNN`` on Fashion-MNIST and predict class probabilities on split test sets.

    Args:
        path_to_toy_datasets: Local directory path where Fashion-MNIST data is stored or downloaded.
        path_to_trained_model: File path used to save and reload ``ToyCNN`` state weights.
        random_seed: Seed passed to ``split_dataset`` for deterministic test splitting.

    Returns:
        A tuple of ``(calibration_split, test_split, calibration_softmax, test_softmax)``,
        where ``calibration_split`` and ``test_split`` are ``ClassificationDataset`` instances
        (5,000 samples each) and ``calibration_softmax`` / ``test_softmax`` are ``torch.Tensor``
        objects of shape ``(5000, 10)``.
    """
    # Train set:
    # train_dataset.x: float32 tensor of shape (60_000, 1, 28, 28), pixel values in [0, 1]
    # train_dataset.y: int64 tensor of shape (60_000,), class labels in {0, ..., 9}
    train_dataset: ClassificationDataset = load_toy_dataset_fashion_mnist(
        download_path=path_to_toy_datasets, train=True
    )

    # Test set:
    # test_dataset.x: float32 tensor of shape (10_000, 1, 28, 28), pixel values in [0, 1]
    # test_dataset.y: int64 tensor of shape (10_000,), class labels in {0, ..., 9}
    test_dataset: ClassificationDataset = load_toy_dataset_fashion_mnist(
        download_path=path_to_toy_datasets, train=False
    )

    # Train ToyCNN and save its state_dict to disk
    train_toy_classification_model(train_dataset, path_to_trained_model)

    # Load ToyCNN in eval mode
    model: ToyCNN = load_toy_classification_model(path_to_trained_model)

    # Split the 10,000-sample test set into calibration and test splits (5,000 samples each)
    calibration_split, test_split = split_dataset(test_dataset, ratio=0.5, seed=random_seed)

    # Obtain class probability distributions (torch.Size([5000, 10]))
    calibration_softmax: torch.Tensor = predict_probs(model, calibration_split.x)
    test_softmax: torch.Tensor = predict_probs(model, test_split.x)

    return calibration_split, test_split, calibration_softmax, test_softmax
