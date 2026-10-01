import torch

from conformal_utils import Calibration, conformal_quantile
from toy_datasets import Data
from utils_regression import (
    print_regression_calibration_report, print_regression_task_report, print_regression_uncertainty_report
)


@torch.no_grad()
def predict_values(model: torch.nn.Module, x: torch.Tensor, batch_size: int = 256) -> torch.Tensor:
    """Point predictions, shape (N,), float64, on CPU."""
    model.eval()
    device = next(model.parameters()).device
    out = [
        model(x[i : i + batch_size].to(device)).double().reshape(-1).cpu()
        for i in range(0, len(x), batch_size)
    ]
    return torch.cat(out)


def calibrate_regression(model: torch.nn.Module, calibration_dataset: Data, alpha: float) -> Calibration:
    """Compute the half-width of the conformal intervals from held-out labelled data (absolute-error score)."""

    if not 0 < alpha < 1:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    predictions = predict_values(model, calibration_dataset.x)
    calibration_scores = (calibration_dataset.y.double() - predictions).abs()

    qhat = conformal_quantile(calibration_scores, alpha)
    calibration = Calibration(alpha, qhat, len(calibration_dataset))

    print_regression_calibration_report(calibration=calibration)

    return calibration


def predict_intervals(model: torch.nn.Module, x: torch.Tensor, qhat: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Lower and upper bounds, shape (N,) each. No labels needed."""
    predictions = predict_values(model, x)
    lower_bound = predictions - qhat
    upper_bound = predictions + qhat
    return lower_bound, upper_bound


def evaluate_regression(model: torch.nn.Module, test: Data, calibration: Calibration) -> None:
    """Build prediction intervals on `test` and measure coverage and width."""

    predictions = predict_values(model, test.x)
    lower, upper = predictions - calibration.qhat, predictions + calibration.qhat
    targets = test.y.double()
    covered = (targets >= lower) & (targets <= upper)

    print_regression_task_report(
        predictions=predictions,
        targets=targets
    )

    print_regression_uncertainty_report(
        covered=covered,
        calibration=calibration
    )
