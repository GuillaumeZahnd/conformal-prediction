import torch
import math

from conformal_utils import Calibration


def print_classification_uncertainty_report(
    prediction_sets: torch.Tensor,
    covered: torch.Tensor,
) -> None:

    """
    Print the uncertanty report.

    Args:
        prediction_sets: Boolean mask of shape (nb_samples, nb_classes).
            True where a class belongs to the prediction set of a sample.
        covered: Boolean tensor of shape (nb_samples,).
            True where the true label of a sample is inside its prediction set.
    """

    print("\n" + "-"*64)
    print("UNCERTAINTY REPORT")
    print("-"*64)

    # How often the true label was within the prediction set
    nb_samples = len(covered)
    nb_covered = int(covered.sum())
    print(f"Coverage (true label in prediction set): {nb_covered}/{nb_samples} ({nb_covered / nb_samples:.1%})")
    print("")

    # How often prediction sets contain N classes, most frequent first
    counts, frequencies = prediction_set_size_distribution(prediction_sets)
    order = torch.sort(frequencies, descending=True, stable=True).indices
    print(f"{'set size':>8} | {'count':>7} | {'frequency':>9}")
    for n in order.tolist():
        if counts[n] == 0:
            break  # Skip cases that do not occur
        print(f"{n:>8} | {counts[n].item():>7} | {frequencies[n].item():>9.3f}")


def print_classification_calibration_report(calibration: "Calibration") -> None:
    """Print the outcome of the calibration step."""

    print("\n" + "-"*64)
    print("CALIBRATION REPORT")
    print("-"*64)

    print(f"Calibration samples: {calibration.nb_calibration_samples}")

    print(f"alpha: {calibration.alpha}  (target coverage >= {1 - calibration.alpha:.1%})")

    if math.isinf(calibration.qhat):
        print("qhat: inf (the calibration set is too small for this alpha: every class is kept)")
    else:
        print(f"qhat: {calibration.qhat:.4f}")
        print(f"A class is included in the prediction set if its softmax probability is >= {1 - calibration.qhat:.4f}")


def print_classification_task_report(probs: torch.Tensor, targets: torch.Tensor) -> None:
    """
    Print basic classification metrics: overall accuracy, mean confidence, per-class accuracy.

    Args:
        probs: Softmax probabilities of shape (nb_samples, nb_classes).
        targets: Ground-truth class labels of shape (nb_samples,), int64.
    """
    nb_samples, nb_classes = probs.shape
    correct = probs.argmax(dim=1) == targets

    print("\n" + "-"*64)
    print("CLASSIFICATION REPORT")
    print("-"*64)

    nb_correct = int(correct.sum())
    print(f"Accuracy: {nb_correct}/{nb_samples} ({nb_correct / nb_samples:.1%})")
    print(f"Mean confidence (top-1 probability): {probs.max(dim=1).values.mean().item():.3f}")
    print("")

    support = torch.bincount(targets, minlength=nb_classes)
    correct_per_class = torch.bincount(targets[correct], minlength=nb_classes)

    print(f"{'class':>5} | {'support':>7} | {'accuracy':>8}")
    for c in range(nb_classes):
        if support[c] == 0:
            continue  # Skip cases that do not occur
        print(f"{c:>5} | {support[c].item():>7} | {correct_per_class[c].item() / support[c].item():>8.3f}")


def prediction_set_size_distribution(prediction_sets: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Count how often prediction sets contain exactly N classes, for N = 0, ..., nb_classes.

    Args:
        prediction_sets: Boolean mask of shape (nb_samples, nb_classes).

    Returns:
        counts: int64 tensor of shape (nb_classes + 1,).
            counts[N] is the number of samples whose prediction set has exactly N classes.
        frequencies: float64 tensor of shape (nb_classes + 1,).
            frequencies[N] is the fraction of samples whose prediction set has exactly N classes.
    """

    nb_samples, nb_classes = prediction_sets.shape
    sizes = prediction_sets.sum(dim=1)  # (nb_samples,), values in [0, nb_classes]
    counts = torch.bincount(sizes, minlength=nb_classes + 1)
    frequencies = counts.double() / nb_samples

    return counts, frequencies
