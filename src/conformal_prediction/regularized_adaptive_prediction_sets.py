import torch


def regularize(
    softmax_values: torch.Tensor,
    k_regularization: int,
    lambda_regularization: float,
) -> tuple[torch.Tensor, torch.Tensor]:

    k_regularization = 5
    lambda_regularization = 0.01
    n_classes = softmax_values.shape[1]

    # Vector of length `n_classes`, with the first `k_regularization` values set to zero, and the remaining values set to `lambda_regularization`
    regularization_vector = torch.cat(
        [torch.zeros(k_regularization),
         torch.full((n_classes - k_regularization,), lambda_regularization)]
    ).unsqueeze(0)

    # Indices of softmax_values, sorted by decreasing order
    pi_indices = softmax_values.argsort(dim=1, descending=True)  # torch.Size([5000, 10])

    # Softmax values, sorted from the largest to the smallest
    softmax_values_sorted = torch.take_along_dim(softmax_values, pi_indices, dim=-1)  # torch.Size([5000, 10])

    softmax_values_sorted_regularized = softmax_values_sorted + regularization_vector  # torch.Size([5000, 10])

    return softmax_values_sorted_regularized, pi_indices
