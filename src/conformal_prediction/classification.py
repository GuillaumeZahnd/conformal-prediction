import torch
import math
from dataclasses import dataclass

from .core import calibrate, Calibration


@dataclass(frozen=True)
class ClassificationUncertainty:
    """Result of a conformal classification evaluation on a labelled test set.

    Attributes
    ----------
    nb_samples : int
        Number of evaluated test samples.
    nb_covered : int
        Number of samples whose prediction set contains the true label.
    coverage : float
        Empirical coverage, ``nb_covered / nb_samples``. The conformal guarantee targets ``1 - alpha``.
    mean_set_size : float
        Mean number of classes in the prediction sets.
    median_set_size : int
        Smallest set size ``N`` such that at least 50% of the samples have a set of ``N`` classes or fewer.
    set_size_q05 : int
        Smallest set size ``N`` such that at least 5% of the samples have a set of ``N`` classes or fewer.
    set_size_q95 : int
        Smallest set size ``N`` such that at least 95% of the samples have a set of ``N`` classes or fewer.
    min_set_size : int
        Smallest observed prediction set size; ``0`` if any set is empty.
    max_set_size : int
        Largest observed prediction set size.
    set_size_counts : tuple of int
        Length ``nb_classes + 1``. Entry ``N`` is the number of samples whose set has exactly ``N`` classes.
    set_size_frequencies : tuple of float
        Length ``nb_classes + 1``. Entry ``N`` is ``set_size_counts[N] / nb_samples``; the entries sum to 1.
    covered_per_set_size : tuple of int
        Length ``nb_classes + 1``. Entry ``N`` is the number of samples with a set of ``N`` classes
        that contain the true label. Entry ``0`` is always ``0``.
    """
    nb_samples: int
    nb_covered: int
    coverage: float
    mean_set_size: float
    median_set_size: int
    set_size_q05: int
    set_size_q95: int
    min_set_size: int
    max_set_size: int
    set_size_counts: tuple[int, ...]
    set_size_frequencies: tuple[float, ...]
    covered_per_set_size: tuple[int, ...]


@torch.no_grad()
def predict_probs(
    model: torch.nn.Module,
    x: torch.Tensor,
    batch_size: int = 256
) -> torch.Tensor:
    """Softmax probabilities, shape (N, K), float64, on CPU.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network classification model.
    x : torch.Tensor
        Input features tensor of shape ``(N, ...)``.
    batch_size : int, optional
        Batch size for inference.

    Returns
    -------
    torch.Tensor
        Predicted softmax probabilities of shape ``(N, K)`` on CPU in float64.
    """
    model.eval()
    device = next(model.parameters()).device
    out = [
        torch.softmax(model(x[i : i + batch_size].to(device)).double(), dim=1).cpu()
        for i in range(0, len(x), batch_size)
    ]
    return torch.cat(out)


def calibrate_classification(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float,
) -> Calibration:
    """Compute the conformal threshold from held-out labelled data using the specified score type.

    Parameters
    ----------
    model : torch.nn.Module
        The classification model.
    x : torch.Tensor
        Calibration input tensor.
    y : torch.Tensor
        Calibration true labels tensor.
    alpha : float
        Miscoverage level in ``(0, 1)``.

    Returns
    -------
    Calibration
        The resulting calibration object with computed threshold ``qhat``.
    """
    calibration_scores = compute_calibration_scores_classification(model, x, y)
    return calibrate(calibration_scores, alpha)


def compute_calibration_scores_classification(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """Compute local average conformity scores for calibration.

    Parameters
    ----------
    model : torch.nn.Module
        The classification model.
    x : torch.Tensor
        Input features tensor.
    y : torch.Tensor
        True labels tensor.

    Returns
    -------
    torch.Tensor
        Calibration scores tensor of shape ``(n_calibration_samples,)`` in ``[0, 1]``.
    """
    n_calibration_samples = len(x)
    probs = predict_probs(model, x)  # (n_calibration_samples, nb_classes), in [0, 1]
    prob_true_class = probs[torch.arange(n_calibration_samples), y]
    # Local Average Conformity: 1 - p(true_class)
    calibration_scores = 1 - prob_true_class  # (n_calibration_samples), in [0, 1]
    return calibration_scores


def evaluate_classification(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    calibration: Calibration,
) -> ClassificationUncertainty:
    """Build prediction sets on the test set, and measure coverage and set size.

    Parameters
    ----------
    model : torch.nn.Module
        The classification model.
    x : torch.Tensor
        Test input features tensor.
    y : torch.Tensor
        Test true labels tensor.
    calibration : Calibration
        Fitted calibration object containing ``qhat``.

    Returns
    -------
    ClassificationUncertainty
        Evaluation results containing coverage statistics and set sizes.
    """

    probs = predict_probs(model, x)  # (nb_test_samples, nb_classes), in [0, 1]

    # Boolean mask of shape (nb_samples, nb_classes). True where a class belongs to the prediction set of a sample.
    prediction_sets = (1 - probs) <= calibration.qhat

    # Boolean tensor of shape (nb_samples,). True where the true label of a sample is inside its prediction set.
    covered = prediction_sets[torch.arange(len(y)), y]

    nb_samples, nb_classes = prediction_sets.shape
    nb_covered = int(covered.sum())

    sizes = prediction_sets.sum(dim=1)  # (nb_samples,)
    counts = torch.bincount(sizes, minlength=nb_classes + 1)  # (nb_classes + 1,)
    frequencies = counts.double() / nb_samples  # (nb_classes + 1,)
    covered_counts = torch.bincount(sizes[covered], minlength=nb_classes + 1)

    # Smallest set size N such that at least a fraction q of the samples have a set of N classes or fewer
    cumulative_counts = counts.cumsum(dim=0)
    q05, q50, q95 = (
        int(torch.searchsorted(cumulative_counts, math.ceil(q * nb_samples))) for q in (0.05, 0.5, 0.95)
    )

    classification_uncertainty = ClassificationUncertainty(
        nb_samples=nb_samples,
        nb_covered=nb_covered,
        coverage=nb_covered / nb_samples,
        mean_set_size=sizes.double().mean().item(),
        median_set_size=q50,
        set_size_q05=q05,
        set_size_q95=q95,
        min_set_size=int(sizes.min()),
        max_set_size=int(sizes.max()),
        set_size_counts=tuple(counts.tolist()),
        set_size_frequencies=tuple(frequencies.tolist()),
        covered_per_set_size=tuple(covered_counts.tolist()),
    )

    return classification_uncertainty
