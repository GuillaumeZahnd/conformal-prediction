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
    
    calibration_softmax = predict_probs(model, calibration_split.x)  # torch.Size([5000, 10])
    test_softmax = predict_probs(model, test_split.x)  # torch.Size([5000, 10])    

    calibration_labels = calibration_split.y  # torch.Size([5000])
    test_labels = test_split.y  # torch.Size([5000])

 
    import numpy as np
    import math

    """
    n=1000 # number of calibration points
    
    # Set RAPS regularization parameters (larger lambda_regularization and smaller k_regularization leads to smaller sets)
    
    
    disallow_zero_sets = False # Set this to False in order to see the coverage upper bound hold
    rand = True # Set this to True in order to see the coverage upper bound hold
    
    """

    alpha = 0.1 # 1-alpha is the desired coverage
    k_regularization = 5
    lambda_regularization = 0.01
    n_samples, n_classes = calibration_softmax.shape

    # Vector of length `n_classes`, with the first `k_regularization` values set to zero, and the remaining values set to `lambda_regularization`
    regularization_vector = torch.cat([
        torch.zeros(k_regularization),
        torch.full((n_classes - k_regularization,), lambda_regularization)
    ]).unsqueeze(0)
    

    # Indices of calibration_softmax, sorted by decreasing order
    cal_pi = calibration_softmax.argsort(dim=1, descending=True)  # torch.Size([5000, 10])

    # Softmax values, sorted from the largest to the smallest
    calibration_softmax_sorted = torch.take_along_dim(calibration_softmax, cal_pi, dim=-1)  # torch.Size([5000, 10])

    calibration_softmax_sorted_regularized = calibration_softmax_sorted + regularization_vector  # torch.Size([5000, 10])
    print(calibration_softmax_sorted_regularized[0])


    softmax_values_sorted_regularized, pi_indices = regularize(
        softmax_values=calibration_softmax,
        k_regularization=k_regularization,
        lambda_regularization=lambda_regularization,
    )

    input("stop")

    
    # Index of the true label for each sample after the softmax probabilities have been sorted in descending order
    # For instance, an index of zero indicates that the true label is ranked as the strongest softmax value.
    # A typical bincount, for 10 classes, ressembles: tensor([4226,  528,  166,   43,   14,   17,    3,    3,    0,    0])
    index_of_gt_label_sorted = torch.where(cal_pi == calibration_labels.unsqueeze(1))[1]

    print(index_of_gt_label_sorted[100:200])
    print(torch.sum(index_of_gt_label_sorted))

    print(torch.bincount(index_of_gt_label_sorted, minlength=n_classes))


    # See Appendix B (Randomized Predictors) of paper Angelopoulos 2022 - Uncertainty Sets for Image Classifiers using Conformal Prediction    
    device = calibration_softmax_sorted_regularized.device
    row_indices = torch.arange(n_samples, device=device)

    # Part A: cumsum value up to the rank of the ground truth label
    cumsum_matrix = calibration_softmax_sorted_regularized.cumsum(dim=1)    
    part_a = cumsum_matrix[row_indices, index_of_gt_label_sorted]

    # Part B: random uniform number multiplied by the score at the ground truth label
    random_factors = torch.rand(n_samples, device=device, dtype=calibration_softmax_sorted_regularized.dtype)
    part_b = random_factors * calibration_softmax_sorted_regularized[row_indices, index_of_gt_label_sorted]

    calibration_scores = part_a - part_b  # torch.Size([5000])


    # Get the score quantile
    q = math.ceil((n_samples+1) * (1-alpha)) / n_samples
    qhat = torch.quantile(calibration_scores, q, interpolation="higher")  # tensor(0.8940, dtype=torch.float64)
    print(qhat)

    input("-----------------------------")

    # Get the score quantile
    
    # Deploy
    n_val = val_smx.shape[0]
    val_pi = val_smx.argsort(1)[:,::-1]
    val_srt = np.take_along_axis(val_smx,val_pi,axis=1)
    val_srt_reg = val_srt + regularization_vector
    val_srt_reg_cumsum = val_srt_reg.cumsum(axis=1)
    indicators = (val_srt_reg.cumsum(axis=1) - np.random.rand(n_val,1)*val_srt_reg) <= qhat if rand else val_srt_reg.cumsum(axis=1) - val_srt_reg <= qhat
    if disallow_zero_sets: indicators[:,0] = True
    prediction_sets = np.take_along_axis(indicators,val_pi.argsort(axis=1),axis=1)






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