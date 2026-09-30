import hydra
from omegaconf import DictConfig

from toy_datasets import load_toy_dataset_fashion_mnist, split_dataset
from toy_models import train_toy_classifier, load_toy_classifier
from conformal import calibrate, evaluate


@hydra.main(version_base=None, config_path="config", config_name="config")
def demo_classification_fashionmnist(cfg: DictConfig) -> None:

    # Train set
    # dataset_train.x: float32 tensor of shape (60000, 1, 28, 28), pixel values in [0, 1]
    # dataset_train.y: int64 tensor of shape (60000,), class labels in {0, ..., 9}
    train_dataset = load_toy_dataset_fashion_mnist(download_path=cfg.path_to_toy_datasets, train=True)

    # Test set
    # dataset_test.x: float32 tensor of shape (10000, 1, 28, 28), pixel values in [0, 1]
    # dataset_testy: int64 tensor of shape (10000,), class labels in {0, ..., 9}
    test_dataset = load_toy_dataset_fashion_mnist(download_path=cfg.path_to_toy_datasets, train=False)

    # Train a lightweight classification model for one epoch and save the weights
    train_toy_classifier(train_dataset, cfg.path_to_trained_model)

    # Load the trained classification model
    model = load_toy_classifier(cfg.path_to_trained_model)

    # Split the held-out set into the calibration split and the test split
    calibration_split, test_split = split_dataset(test_dataset, frac=0.5, seed=cfg.random_seed)

    # Calibration step
    calibration = calibrate(model, calibration_split, alpha=cfg.alpha)

    # Test step
    evaluate(model, test_split, calibration.qhat)


if __name__ =="__main__":
    demo_classification_fashionmnist()

