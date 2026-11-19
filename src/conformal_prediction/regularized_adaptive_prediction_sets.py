import torch
import math


def get_qhat(
    softmax_values: torch.Tensor,
    labels: torch.Tensor,
    k_regularization: int,
    lambda_regularization: float,
    alpha: float,
) -> torch.Tensor:

    device = softmax_values.device
    dtype = softmax_values.dtype

    n_samples = softmax_values.shape[0]

    softmax_values_sorted_regularized, pi_indices_calibration = regularize(
        softmax_values=softmax_values,
        k_regularization=k_regularization,
        lambda_regularization=lambda_regularization,
    )

    # Index of the true label for each sample after the softmax probabilities have been sorted in descending order
    # For instance, an index of zero indicates that the true label is ranked as the strongest softmax value.
    # A typical bincount, for 10 classes, ressembles: tensor([4226,  528,  166,   43,   14,   17,    3,    3,    0,    0])
    index_of_gt_label_sorted = torch.where(pi_indices_calibration == labels.unsqueeze(1))[1]  # (n_samples)

    # See Appendix B (Randomized Predictors) of paper Angelopoulos 2022 - Uncertainty Sets for Image Classifiers using Conformal Prediction
    row_indices = torch.arange(n_samples, device=device)

    # Cumsum value up to the rank of the ground truth label
    cumsum_matrix = softmax_values_sorted_regularized.cumsum(dim=1)

    # Random uniform number multiplied by the score at the ground truth label
    random_factors = torch.rand(n_samples, device=device, dtype=dtype)

    scores = \
        cumsum_matrix[row_indices, index_of_gt_label_sorted] - \
        random_factors * softmax_values_sorted_regularized[row_indices, index_of_gt_label_sorted]

    # Get the score quantile
    q = math.ceil((n_samples+1) * (1-alpha)) / n_samples
    qhat = torch.quantile(scores, q, interpolation="higher")  # tensor(0.8940, dtype=torch.float64)

    return qhat


def regularize(
    softmax_values: torch.Tensor,
    k_regularization: int,
    lambda_regularization: float,
) -> tuple[torch.Tensor, torch.Tensor]:

    _, n_classes = softmax_values.shape
    device = softmax_values.device
    dtype = softmax_values.dtype

    k_clamped = max(0, min(k_regularization, n_classes))

    # Indices of softmax_values, sorted by decreasing order
    pi_indices = softmax_values.argsort(dim=1, descending=True)

    # Vector of length `n_classes`, with the first `k_regularization` values set to zero, and the remaining values set to `lambda_regularization`
    regularization_vector = torch.cat(
        [
            torch.zeros(k_clamped, device=device, dtype=dtype),
            torch.full((n_classes - k_clamped,), lambda_regularization, device=device, dtype=dtype),
        ]
    ).unsqueeze(0)

    # Softmax values, sorted from the largest to the smallest
    softmax_values_sorted = torch.take_along_dim(softmax_values, pi_indices, dim=1)
    softmax_values_sorted_regularized = softmax_values_sorted + regularization_vector

    return softmax_values_sorted_regularized, pi_indices
