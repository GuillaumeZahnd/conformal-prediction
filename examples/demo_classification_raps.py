import hydra
from omegaconf import DictConfig

from toy_datasets import load_toy_dataset_fashion_mnist, split_dataset
from toy_model_classification import train_toy_classification_model, load_toy_classification_model
from plot_classification import plot_prediction_set
from utils import format_classification_task_report

from conformal_prediction import (
    calibrate_classification,
    evaluate_classification,
    predict_probs,
    format_classification_calibration,
    format_classification_uncertainty,
    regularize,
    get_qhat,
)


@hydra.main(version_base=None, config_path="config", config_name="config")
def demo_classification_fashion_mnist(cfg: DictConfig) -> None:

    # Train set
    # dataset_train.x: float32 tensor of shape (60000, 1, 28, 28), pixel values in [0, 1]
    # dataset_train.y: int64 tensor of shape (60000,), class labels in {0, ..., 9}
    train_dataset = load_toy_dataset_fashion_mnist(download_path=cfg.path_to_toy_datasets, train=True)

    # Test set
    # dataset_test.x: float32 tensor of shape (10000, 1, 28, 28), pixel values in [0, 1]
    # dataset_testy: int64 tensor of shape (10000,), class labels in {0, ..., 9}
    test_dataset = load_toy_dataset_fashion_mnist(download_path=cfg.path_to_toy_datasets, train=False)

    # Train a lightweight classification model for one epoch and save the weights
    train_toy_classification_model(train_dataset, cfg.path_to_trained_model)

    # Load the trained model
    model = load_toy_classification_model(cfg.path_to_trained_model)

    # Split the held-out set into the calibration split and the test split
    calibration_split, test_split = split_dataset(test_dataset, ratio=0.5, seed=cfg.random_seed)

    # ================================================================
    import torch
    # calibration_split, test_split: 5000 samples each

    test_softmax = predict_probs(model, test_split.x)  # torch.Size([5000, 10])
    test_labels = test_split.y  # torch.Size([5000])


    # Set RAPS regularization parameters (larger lambda_regularization and smaller k_regularization leads to smaller sets)
    alpha = 0.1 # 1-alpha is the desired coverage
    k_regularization = 5
    lambda_regularization = 0.01
    rand = True # Set this to True in order to see the coverage upper bound hold
    disallow_zero_sets = False # Set this to False in order to see the coverage upper bound hold


    calibration_softmax = predict_probs(model, calibration_split.x)  # torch.Size([5000, 10])
    calibration_labels = calibration_split.y  # torch.Size([5000])

    qhat = get_qhat(
        softmax_values=calibration_softmax,
        labels=calibration_labels,
        k_regularization=k_regularization,
        lambda_regularization=lambda_regularization,
        alpha=alpha,
    )

    print(qhat)

    # -------- Deploy --------
    n_samples_test, n_classes = test_softmax.shape
    device = test_softmax.device

    test_softmax_sorted_regularized, pi_indices_test = regularize(
        softmax_values=test_softmax,
        k_regularization=k_regularization,
        lambda_regularization=lambda_regularization,
    )

    print(pi_indices_test[:,0])
    print(pi_indices_test[:,1])
    print(pi_indices_test[:,2])


    # Will exceed 1.0 due the addition of the regularization vector
    test_softmax_sorted_regularized_cumsum = test_softmax_sorted_regularized.cumsum(dim=1)  # (n_samples_test, n_classes)


    if rand:
        random_factors = torch.rand((n_samples_test, 1), device=device, dtype=test_softmax_sorted_regularized.dtype)
        cumsum_shift_random_scale = test_softmax_sorted_regularized_cumsum - random_factors * test_softmax_sorted_regularized
        indicators = cumsum_shift_random_scale <= qhat
    else:
        # cumulative sum shifted right by 1 with 0 in front
        cumsum_shift = test_softmax_sorted_regularized_cumsum - test_softmax_sorted_regularized
        indicators = cumsum_shift <= qhat

    print(test_softmax_sorted_regularized_cumsum[:3])
    print(test_softmax_sorted_regularized[:3])
    u = test_softmax_sorted_regularized_cumsum - test_softmax_sorted_regularized
    print(u[:3])
    print(indicators[:3])


    # Enforce (edge cases)
    if disallow_zero_sets:
        indicators[:,0] = True

    prediction_sets = torch.take_along_dim(indicators, pi_indices_test.argsort(dim=1), dim=1)  # (n_samples_test, n_classes)

    print(prediction_sets[:3])

    input("pause")




    # ================================================================

    # Calibration step
    calibration = calibrate_classification(
        model=model,
        x=calibration_split.x,
        y=calibration_split.y,
        alpha=cfg.alpha
    )
    print(format_classification_calibration(calibration), end="\n\n")

    from viz import plot_conformal_calibration
    plot_conformal_calibration(
        calibration_scores=calibration.calibration_scores,
        calibration=calibration,
        title="classification_calibration",
    )

    # Test step
    uncertainty = evaluate_classification(model, test_split.x, test_split.y, calibration)
    print(format_classification_uncertainty(uncertainty), end="\n\n")

    # Some more evaluation
    test_probs = predict_probs(model, test_split.x)

    # Plot one sample
    sample_index = cfg.sample_index_demo if cfg.sample_index_demo < len(test_split) else 0
    plot_prediction_set(
        image=test_split.x[sample_index],
        probs=test_probs[sample_index],
        qhat=calibration.qhat,
        alpha=cfg.alpha,
        sample_index=sample_index,
        true_label=int(test_split.y[sample_index]),
        class_names=test_split.class_names,
    )

    # Print basic classification metrics
    print(format_classification_task_report(test_probs, test_split.y), end="\n\n")


if __name__ =="__main__":
    demo_classification_fashion_mnist()
