import hydra
from omegaconf import DictConfig

from toy_datasets import load_toy_dataset_fashion_mnist, split_dataset
from toy_model_classification import train_toy_classification_model, load_toy_classification_model
from conformal_classification import calibrate_classification, evaluate_classification, predict_probs
from plot_classification import plot_prediction_set


@hydra.main(version_base=None, config_path="config", config_name="config")
def demo_classification_fashion_mnist(cfg: DictConfig) -> None:

    # Train set
    # dataset_train.x: float32 tensor of shape (60000, 1, 28, 28), pixel values in [0, 1]
    # dataset_train.y: int64 tensor of shape (60000,), class labels in {0, ..., 9}
    # CLASSES
    train_dataset = load_toy_dataset_fashion_mnist(download_path=cfg.path_to_toy_datasets, train=True)

    # Test set
    # dataset_test.x: float32 tensor of shape (10000, 1, 28, 28), pixel values in [0, 1]
    # dataset_testy: int64 tensor of shape (10000,), class labels in {0, ..., 9}
    # CLASSES
    test_dataset = load_toy_dataset_fashion_mnist(download_path=cfg.path_to_toy_datasets, train=False)

    # Train a lightweight classification model for one epoch and save the weights
    train_toy_classification_model(train_dataset, cfg.path_to_trained_model)

    # Load the trained model
    model = load_toy_classification_model(cfg.path_to_trained_model)

    # Split the held-out set into the calibration split and the test split
    calibration_split, test_split = split_dataset(test_dataset, ratio=0.5, seed=cfg.random_seed)

    # Calibration step
    calibration = calibrate_classification(model, calibration_split, alpha=cfg.alpha)

    # Test step
    evaluate_classification(model, test_split, calibration.qhat)

    # Plot one sample
    sample_index = cfg.sample_index_demo if cfg.sample_index_demo < len(test_split) else 0
    x = test_split.x[sample_index]
    y = int(test_split.y[sample_index])
    probs = predict_probs(model, x[None])[0]
    plot_prediction_set(
        image=x,
        probs=probs,
        qhat=calibration.qhat,
        alpha=cfg.alpha,
        sample_index=sample_index,
        true_label=y,
        class_names=test_split.class_names,
    )

if __name__ =="__main__":
    demo_classification_fashion_mnist()
