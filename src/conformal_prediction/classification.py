import torch
import math
from dataclasses import dataclass

from .core import Calibration, conformal_quantile


@dataclass(frozen=True)
class ClassificationUncertainty:
    nb_samples: int
    nb_covered: int
    coverage: float
    mean_set_size: float
    median_set_size: int
    set_size_q05: int
    set_size_q95: int
    min_set_size: int
    max_set_size: int
    set_size_counts: tuple[int, ...]         # index N -> number of samples whose set has N classes
    set_size_frequencies: tuple[float, ...]  # index N -> set_size_counts[N] / nb_samples
    covered_per_set_size: tuple[int, ...]    # index N -> how many of those contain the true label


@torch.no_grad()
def predict_probs(model: torch.nn.Module, x: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    """Softmax probabilities, shape (N, K), float64, on CPU."""
    model.eval()
    device = next(model.parameters()).device
    out = [
        torch.softmax(model(x[i : i + batch_size].to(device)).double(), dim=1).cpu()
        for i in range(0, len(x), batch_size)
    ]
    return torch.cat(out)


def calibrate_classification(model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor, alpha: float) -> Calibration:
    """Compute the conformal threshold from held-out labelled data (LAC score)."""

    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    nb_calibration_samples = len(x)

    probs = predict_probs(model, x)  # (nb_calibration_samples, nb_classes), in [0, 1]

    # Calibration score: 1 - p(true_class)
    prob_true_class = probs[torch.arange(nb_calibration_samples), y]
    calibration_scores = 1 - prob_true_class  # (nb_calibration_samples), in [0, 1]

    qhat = conformal_quantile(calibration_scores, alpha)
    calibration = Calibration(alpha, qhat, nb_calibration_samples)

    return calibration


def evaluate_classification(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    calibration: Calibration,
) -> ClassificationUncertainty:
    """Build prediction sets on the test set, and measure coverage and set size."""

    probs = predict_probs(model, x)  # (nb_test_samples, nb_classes), in [0, 1]

    # Boolean mask of shape (nb_samples, nb_classes). True where a class belongs to the prediction set of a sample.
    prediction_sets = (1 - probs) <= calibration.qhat

    #Boolean tensor of shape (nb_samples,). True where the true label of a sample is inside its prediction set.
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
