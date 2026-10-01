import torch

from conformal_utils import Calibration, conformal_quantile
from toy_datasets import Data
from utils_classification import (
    print_classification_calibration_report, print_classification_uncertainty_report, print_classification_task_report
)


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


def calibrate_classification(model: torch.nn.Module, calibration_dataset: Data, alpha: float) -> Calibration:
    """Compute the conformal threshold from held-out labelled data (LAC score)."""

    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    nb_calibration_samples = len(calibration_dataset)

    probs = predict_probs(model, calibration_dataset.x)  # (nb_calibration_samples, nb_classes), in [0, 1]

    # Calibration score: 1 - p(true_class)
    prob_true_class = probs[torch.arange(nb_calibration_samples), calibration_dataset.y]
    calibration_scores = 1 - prob_true_class  # (nb_calibration_samples), in [0, 1]

    qhat = conformal_quantile(calibration_scores, alpha)
    calibration = Calibration(alpha, qhat, nb_calibration_samples)

    print_classification_calibration_report(calibration=calibration)

    return calibration


def evaluate_classification(model: torch.nn.Module, test: Data, qhat: float) -> None:
    """Build prediction sets on `test` and measure coverage and set size."""

    probs = predict_probs(model, test.x)  # (nb_test_samples, nb_classes), in [0, 1]

    prediction_sets = (1 - probs) <= qhat  # (nb_test_samples, nb_classes), boolean mask

    covered = prediction_sets[torch.arange(len(test)), test.y]

    print_classification_task_report(
        probs=probs,
        targets=test.y
    )

    print_classification_uncertainty_report(
        prediction_sets=prediction_sets,
        covered=covered
    )
