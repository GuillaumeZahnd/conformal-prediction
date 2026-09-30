import math
import torch
from dataclasses import dataclass

from toy_datasets import Data
from utils import print_calibration_report, print_uncertainty_report, print_classification_report


@dataclass(frozen=True)
class Calibration:
    alpha: float
    qhat: float
    nb_calibration_samples: int


@torch.no_grad()
def predict_probs(model: torch.nn.Module, x: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    """Softmax probabilities, shape (N, K), float64, on CPU."""
    model.eval()
    device = next(model.parameters()).device
    out = [torch.softmax(model(x[i : i + batch_size].to(device)).double(), dim=1).cpu()
           for i in range(0, len(x), batch_size)]
    return torch.cat(out)


def calibrate(model: torch.nn.Module, calibration_dataset: Data, alpha: float) -> Calibration:
    """Compute the conformal threshold from held-out labelled data (LAC score)."""

    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    nb_calibration_samples = len(calibration_dataset)

    probs = predict_probs(model, calibration_dataset.x)  # (nb_calibration_samples, nb_classes), in [0, 1]

    # Calibration score: 1 - p(true_class)
    prob_true_class = probs[torch.arange(nb_calibration_samples), calibration_dataset.y]
    calibration_scores = 1 - prob_true_class  # (nb_calibration_samples), in [0, 1]

    k = math.ceil((nb_calibration_samples + 1) * (1 - alpha))
    qhat = float("inf") if k > nb_calibration_samples else calibration_scores.sort().values[k - 1].item()
    calibration = Calibration(alpha, qhat, nb_calibration_samples)

    print_calibration_report(calibration=calibration)

    return calibration


def evaluate(model: torch.nn.Module, test: Data, qhat: float) -> None:
    """Build prediction sets on `test` and measure coverage and set size."""

    probs = predict_probs(model, test.x)  # (nb_test_samples, nb_classes), in [0, 1]

    prediction_sets = (1 - probs) <= qhat  # (nb_test_samples, nb_classes), boolean mask

    covered = prediction_sets[torch.arange(len(test)), test.y]

    print_classification_report(
        probs=probs,
        targets=test.y
    )

    print_uncertainty_report(
        prediction_sets=prediction_sets,
        covered=covered
    )
