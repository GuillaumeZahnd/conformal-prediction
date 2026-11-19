import hydra
from omegaconf import DictConfig

from toy_datasets import load_toy_dataset_fashion_mnist, split_dataset
from toy_model_classification import train_toy_classification_model, load_toy_classification_model

from conformal_prediction import (
    predict_probs,
    get_qhat,
    get_prediction_sets
)
from plot_classification_raps import plot_prediction_set

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:

    # Train set
    # dataset_train.x: float32 tensor of shape (60_000, 1, 28, 28), pixel values in [0, 1]
    # dataset_train.y: int64 tensor of shape (60_000,), class labels in {0, ..., 9}
    train_dataset = load_toy_dataset_fashion_mnist(download_path=cfg.path_to_toy_datasets, train=True)

    # Test set
    # dataset_test.x: float32 tensor of shape (10_000, 1, 28, 28), pixel values in [0, 1]
    # dataset_testy: int64 tensor of shape (10_000,), class labels in {0, ..., 9}
    test_dataset = load_toy_dataset_fashion_mnist(download_path=cfg.path_to_toy_datasets, train=False)

    # Train a lightweight classification model for one epoch and save the weights
    train_toy_classification_model(train_dataset, cfg.path_to_trained_model)

    # Load the trained model
    model = load_toy_classification_model(cfg.path_to_trained_model)

    # Split the held-out set into the calibration split and the test split (5_000 samples each)
    calibration_split, test_split = split_dataset(test_dataset, ratio=0.5, seed=cfg.random_seed)

    

    # Calibration phase
    calibration_softmax = predict_probs(model, calibration_split.x)  # torch.Size([5000, 10])
    
    qhat = get_qhat(
        softmax_values=calibration_softmax,
        labels=calibration_split.y,
        k_regularization=cfg.raps.k_regularization,
        lambda_regularization=cfg.raps.lambda_regularization,
        alpha=cfg.raps.alpha,
    )

    # Test phase
    test_softmax = predict_probs(model, test_split.x)  # torch.Size([5000, 10])
    
    prediction_sets = get_prediction_sets(
        softmax_values=test_softmax,
        qhat=qhat,
        k_regularization=cfg.raps.k_regularization,
        lambda_regularization=cfg.raps.lambda_regularization,
        randomize=cfg.raps.randomize,
        prohibit_empty_sets=cfg.raps.prohibit_empty_sets,
    )


    # Display
    index = cfg.sample_index_demo
    plot_prediction_set(
        image=test_split.x[index],
        probs=test_softmax[index],
        in_set=prediction_sets[index],
        qhat=float(qhat),
        alpha=cfg.raps.alpha,
        sample_index=index,
        true_label=int(test_split.y[index].item()),
        class_names=test_split.class_names,
        k_regularization=cfg.raps.k_regularization,
        lambda_regularization=cfg.raps.lambda_regularization,
    )


if __name__ =="__main__":
    main()
