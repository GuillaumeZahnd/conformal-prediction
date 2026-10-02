import hydra
from omegaconf import DictConfig

from toy_datasets import load_toy_dataset_california_housing, split_dataset
from toy_model_regression import train_toy_regression_model, load_toy_regression_model

from conformal_prediction import (
    calibrate_regression,
    evaluate_regression,
    format_regression_calibration,
    format_regression_uncertainty,
)


@hydra.main(version_base=None, config_path="config", config_name="config")
def demo_regression_california_housing(cfg: DictConfig) -> None:

    # Dataset
    train_dataset = load_toy_dataset_california_housing(download_path=cfg.path_to_toy_datasets, train=True)
    test_dataset = load_toy_dataset_california_housing(download_path=cfg.path_to_toy_datasets, train=False)

    # Train a lightweight regression model for one epoch and save the weights
    train_toy_regression_model(train_dataset, cfg.path_to_trained_model)

    # Load the trained model
    model = load_toy_regression_model(cfg.path_to_trained_model)

    # Split the held-out set into the calibration split and the test split
    calibration_split, test_split = split_dataset(test_dataset, ratio=0.5, seed=cfg.random_seed)

    # Calibration step
    calibration = calibrate_regression(model, calibration_split.x, calibration_split.y, alpha=cfg.alpha)
    print(format_regression_calibration(calibration))

    # Test step
    uncertainty = evaluate_regression(model, test_split.x, test_split.y, calibration)
    print(format_regression_uncertainty(uncertainty))


if __name__ =="__main__":
    demo_regression_california_housing()
