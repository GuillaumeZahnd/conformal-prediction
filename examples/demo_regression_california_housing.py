import hydra
from omegaconf import DictConfig

from toy_datasets import load_toy_dataset_california_housing, split_dataset
from toy_model_regression import train_toy_regression_model, load_toy_regression_model
from utils import format_regression_task_report

from conformal_prediction import (
    calibrate_regression,
    evaluate_regression,
    predict_values,
    format_regression_calibration,
    format_regression_uncertainty,
)


@hydra.main(version_base=None, config_path="config", config_name="config")
def demo_regression_california_housing(cfg: DictConfig) -> None:

    # Dataset
    train_dataset = load_toy_dataset_california_housing(download_path=cfg.path_to_toy_datasets, train=True)
    test_dataset = load_toy_dataset_california_housing(download_path=cfg.path_to_toy_datasets, train=False)

    # Train a lightweight regression model for a few epochs and save the weights
    train_toy_regression_model(train_dataset, cfg.path_to_trained_model)

    # Load the trained model
    model = load_toy_regression_model(cfg.path_to_trained_model)

    # Split the held-out set into the calibration split and the test split
    calibration_split, test_split = split_dataset(test_dataset, ratio=0.5, seed=cfg.random_seed)

    # Calibration step
    calibration = calibrate_regression(model, calibration_split.x, calibration_split.y, alpha=cfg.alpha)
    print(format_regression_calibration(calibration), end="\n\n")

    # Test step
    uncertainty = evaluate_regression(model, test_split.x, test_split.y, calibration)
    print(format_regression_uncertainty(uncertainty), end="\n\n")

    # Some more evaluation
    test_predictions = predict_values(model, test_split.x)

    # Print basic regression metrics
    print(format_regression_task_report(test_predictions, test_split.y), end="\n\n")


if __name__ =="__main__":
    demo_regression_california_housing()
